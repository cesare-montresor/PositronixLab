"""
Plugin Manager: Lightweight multiprocessing-based plugin orchestration.

Manages plugin lifecycle, pub/sub messaging, environment variables, and services.
Uses simplified request/response protocol via Channel.
"""

from dataclasses import dataclass
import logging
import multiprocessing as mp
import time
from typing import Any, Dict, List, Optional

from modularnet.plugin.plugin_base import PluginDescriptor, Subscription
from modularnet.plugin.plugin_rpc import Channel, PluginLifeCycleRPC, PluginRPC, RequestEnvGetAll, RequestEnvUnsubscribe, RequestPluginStatus, RequestTopicAnnounce, Response
from modularnet.plugin.plugin_worker import PluginWorker, RequestPluginCall, RequestTopicSubscribe, RequestTopicUnsubscribe
from modularnet.plugin.plugin_message import  CommandType, ProcessStatus, Request, RequestEnvGet, RequestEnvSet, RequestEnvSubscribe, PluginStatus, Subscriber
from modularnet.plugin.environment_manager import EnvironmentManager
import threading


logger = logging.getLogger(__name__)

# Set fork mode for queue sharing
try:
    mp.set_start_method('fork', force=True)
except RuntimeError:
    pass

@dataclass
class PluginInfo:
    plugin_id: str
    process: mp.Process
    rpc: PluginRPC
    descriptor: PluginDescriptor
    options: Dict[str, Any]

class PluginManager:
    """
    Lightweight manager for multiprocessing-based plugin orchestration.
    
    Responsibilities:
    - Plugin lifecycle (load, start, stop, unload)
    - Pub/sub topic routing
    - Environment variable management and subscriptions
    - Service request/response dispatch
    """
    
    def __init__(
        self,
        timeout_request: float = 5.0,
        timeout_start: float = 2.0
    ):
        """
        Initialize plugin manager.
        
        Args:
            timeout_request: Default timeout for sync requests (seconds)
            timeout_start: Timeout for start command (seconds)
        """
        self._plugins: Dict[str, PluginInfo] = {}  # plugin_id -> PluginInfo
        
        self._topic_publishers: Dict[str, str] = {}       # topic -> plugin_id
        self._topic_subscribers: Dict[str, Dict[str, Subscriber]] = {}        # topic -> [queue, queue, ...]
        self._env_subscribers: Dict[str, List[str]] = {}  # env_key -> [plugin_id, ...]
        
        self._env = EnvironmentManager()
        
        self.timeout_request = timeout_request
        self.timeout_start = timeout_start

        threading.Thread(target=self._rpc_loop, daemon=True).start()
        threading.Thread(target=self._topic_listener_loop, daemon=True).start()
    
    def _rpc_loop(self) -> None:
        """Main event loop: wait for requests and topic messages."""
        logger.info(f"Plugin {self.plugin_id} worker started")

        
        try:
            while True:
                req = self.channel.receive(timeout=1.0)
                if req is None:
                    continue
                
                self._handle_request(req)
        except KeyboardInterrupt:
            logger.info(f"Plugin {self.plugin_id} interrupted")
        except Exception as e:
            logger.error(f"Plugin {self.plugin_id} error: {e}", exc_info=True)
        finally:
            self.shutdown()

    


    # ========================================================================
    # RPC Responder: Handle requests from plugins
    # ========================================================================
    
    def _init_plugin_rpc(self, plugin_id: str) -> None:
        """Initialize RPC channels for a plugin."""
        if plugin_id not in self._plugins:
            raise ValueError(f"Plugin {plugin_id} not found")
        
        info = self._plugins[plugin_id]
        plugin_rpc = info.rpc

        """
        #PLUGIN_START = "plugin_start"
        #PLUGIN_STOP = "plugin_stop"
        #PLUGIN_STATUS = "plugin_status"
        #PLUGIN_STATUS_ALL = "plugin_status_all"
        #PLUGIN_ERRORS = "plugin_errors"
        PLUGIN_CALL = "plugin_call"

        ENV_SET = "env_set"
        ENV_GET = "env_get"
        ENV_GET_ALL = "env_get_all"
        ENV_SUBSCRIBE = "env_subscribe"
        ENV_UNSUBSCRIBE = "env_unsubscribe"
        #ENV_UPDATE = "env_update"

        TOPIC_SUBSCRIBE = "topic_subscribe"
        TOPIC_UNSUBSCRIBE = "topic_unsubscribe"
        TOPIC_PUBLISH = "topic_publish"
        TOPIC_LIST = "topic_list"
        """

        #TODO: Handy but later
        #plugin_rpc.on_request(CommandType.PLUGIN_START, self._handle_plugin_start)
        #plugin_rpc.on_request(CommandType.PLUGIN_STOP, self._handle_plugin_stop)
        plugin_rpc.on_request(CommandType.PLUGIN_CALL, self._respond_call)  # forward
        #plugin_rpc.on_request(CommandType.PLUGIN_STATUS, self.get_plugin_status) 
        #plugin_rpc.on_request(CommandType.PLUGIN_STATUS_ALL, self.get_all_plugin_status)

        plugin_rpc.on_request(CommandType.ENV_SET, self._respond_env_set)
        plugin_rpc.on_request(CommandType.ENV_GET, self._respond_env_get)
        plugin_rpc.on_request(CommandType.ENV_GET_ALL, self._respond_env_get_all)
        plugin_rpc.on_request(CommandType.ENV_SUBSCRIBE, self._respond_env_subscribe) 
        plugin_rpc.on_request(CommandType.ENV_UNSUBSCRIBE, self._respond_env_unsubscribe)
        
        plugin_rpc.on_request(CommandType.TOPIC_SUBSCRIBE, self._respond_topic_subscribe)
        plugin_rpc.on_request(CommandType.TOPIC_UNSUBSCRIBE, self._respond_topic_unsubscribe)
        plugin_rpc.on_request(CommandType.TOPIC_ANNOUNCE, self._respond_topic_announce)
        plugin_rpc.on_request(CommandType.TOPIC_LIST, self._respond_topic_list)

    # ========================================================================
    # LIFECYCLE: Load, Start, Stop, Unload
    # ========================================================================
    

    def load_plugin(
        self,
        descriptor: PluginDescriptor,
        options: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Load and initialize a plugin (spawn subprocess).
        
        Args:
            descriptor: Plugin descriptor
            options: Initial options for plugin (passed to plugin on start)
        
        Returns:
            True if loaded successfully
        """
        plugin_id = descriptor.name
        options = options or {}
        
        try:
            plugin_rpc = PluginRPC(plugin_id)
            worker_channel = plugin_rpc.channel.Reverse()  # For worker to manager communication
            
            
            # Spawn worker process
            process = mp.Process(
                target=PluginWorker.run,
                args=(plugin_id, descriptor.to_dict(), options, worker_channel),
                name=f"plugin-{plugin_id}"
            )
            process.start()
            
            self._plugins[plugin_id] = PluginInfo(
                plugin_id=plugin_id,
                process=process,
                rpc=plugin_rpc,
                descriptor=descriptor,
                options=options
            )
            
            logger.info(f"Plugin {plugin_id} loaded")
            return True
            
        except Exception as e:
            logger.error(f"Error loading plugin {descriptor}: {e}")
            return False
    
    def start_plugin(self, plugin_id: str) -> bool:
        """
        Start a loaded plugin and register subscriptions.
        
        Args:
            plugin_id: Plugin identifier
        
        Returns:
            True if started successfully
        """
        if plugin_id not in self._plugins:
            logger.error(f"Plugin {plugin_id} not loaded")
            return False
        
        try:
            info = self._plugins[plugin_id]
            pending = info.rpc.plugin_start()
            res = pending.wait(timeout=self.timeout_start)
            if not res:
                logger.warning(f"Plugin {plugin_id} failed to start within {self.timeout_start} seconds")
                return False
            if res.error:
                logger.error(f"Plugin {plugin_id} start error: {res.error}")
                return False

            return res.result
        except Exception as e:
            logger.error(f"Error starting plugin {plugin_id}: {e}")
            return False
    
    def stop_plugin(self, plugin_id: str) -> bool:
        """
        Stop a running plugin.
        
        Args:
            plugin_id: Plugin identifier
        
        Returns:
            True if stopped successfully
        """
        if plugin_id not in self._plugins:
            return False
        
        try:
            info = self._plugins[plugin_id]
            pending = info.rpc.plugin_stop()
            res = pending.wait(timeout=self.timeout_start)
            if not res:
                logger.warning(f"Plugin {plugin_id} failed to stop within {self.timeout_start} seconds")
                return False
            if res.error:
                logger.error(f"Plugin {plugin_id} stop error: {res.error}")
                return False
            
            logger.info(f"Plugin {plugin_id} stop: {res.result}")
            return res.result
        except Exception as e:
            logger.error(f"Error stopping plugin {plugin_id}: {e}")
            return False
    
    def unload_plugin(self, plugin_id: str) -> bool:
        """
        Unload a plugin and terminate its process.
        
        Args:
            plugin_id: Plugin identifier
        
        Returns:
            True if unloaded successfully
        """
        try:
            self.stop_plugin(plugin_id)
            
            if plugin_id in self._plugins:
                info = self._plugins[plugin_id]
                process = info.process
                
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=2)
                    if process.is_alive():
                        process.kill()
                
                # Clean up environment subscriptions
                for env, plugins in self._env_subscribers.items():
                    if plugin_id in plugins:
                        plugins.remove(plugin_id)
                    if len(plugins) == 0:
                        del self._env_subscribers[env]

                # Clean up topic subscriptions
                for topic, plugins in self._topic_subscribers.items():
                    if plugin_id in plugins.keys():
                        del plugins[plugin_id] 
                    if len(plugins) == 0:
                        del self._topic_subscribers[topic]

                # Clean up topic publishers
                for topic, publisher in self._topic_publishers.items():
                    if publisher == plugin_id:
                        del self._topic_publishers[topic]
                
                del self._plugins[plugin_id]
            
            logger.info(f"Plugin {plugin_id} unloaded")
            return True
            
        except Exception as e:
            logger.error(f"Error unloading plugin {plugin_id}: {e}")
            return False
        
    # ========================================================================
    # Utility Methods for Plugin Management
    # ========================================================================

    def publish(self, topic: str, data: Any):
        """
        Publish data to a topic.
        
        Args:
            topic: Topic name
            data: Message data
        
        Returns:
            Number of subscribers that received the message
        """
        
        if topic not in self._topic_subscribers: return True
    
        for subscriber in self._topic_subscribers[topic].values():
            try:
                subscriber.queue.put_nowait(data)
                delivered += 1
            except Exception as e:
                logger.debug(f"Failed to deliver to topic '{topic}': {e}")
        
        return True
     
    def plugin_status(self, plugin_id: str) -> Optional[ProcessStatus]:
        """
        Get status of a plugin.
        
        Args:
            plugin_id: Plugin identifier
        
        Returns:
            ProcessStatus object, or None if not found
        """
        if plugin_id not in self._plugins:
            return None
        
        info = self._plugins[plugin_id]
        process = info.process
        alive = process.is_alive()
        pid = process.pid
        worker_status = None
        
        if alive:
            try:
                pending = info.rpc.plugin_status()
                status_dict = pending.wait(timeout=1.0)
                worker_status = PluginStatus(**status_dict)
            except Exception as e:
                logger.debug(f"Failed to get status for {plugin_id}: {e}")
        
        return ProcessStatus(
            plugin_id=plugin_id,
            process_alive=alive,
            process_pid=pid,
            worker_status=worker_status
        )
    
    def plugin_status_all(self) -> Dict[str, ProcessStatus]:
        """
        Get status of all plugins.
        
        Returns:
            Dictionary of plugin_id -> ProcessStatus
        """
        return {
            plugin_id: self.plugin_status(plugin_id)
            for plugin_id in self._plugins.keys()
        }
    
    def list_plugins(self) -> List[str]:
        """List all loaded plugin IDs."""
        return list(self._plugins.keys())
    
    def start_all_plugins(self) -> None:
        """Start all loaded plugins."""
        for plugin_id in list(self._plugins.keys()):
            self.start_plugin(plugin_id)
    
    def stop_all_plugins(self) -> None:
        """Stop all running plugins."""
        for plugin_id in list(self._plugins.keys()):
            self.stop_plugin(plugin_id)
    
    def shutdown(self) -> None:
        """Shut down the plugin manager and all plugins."""
        self.stop_all_plugins()
        for plugin_id in list(self._plugins.keys()):
            self.unload_plugin(plugin_id)
    
    

    # ========================================================================
    # PLUGIN Lifecycle
    # ========================================================================
  


    def _respond_call(self, rpc: PluginRPC, req: RequestPluginCall) -> Any:
        
        if req.plugin_id not in self._plugins:
            raise ValueError(f"Plugin {req.plugin_id} not found")
        
        forward_rpc = self._plugins[req.plugin_id].rpc
        remote_result = self._forward_request(forward_rpc, req)
        
        return remote_result
    
    # ========================================================================
    # ENVIRONMENT: Variables and Subscriptions
    # ========================================================================
    
    def _respond_env_set(self, rpc: PluginRPC, req: RequestEnvSet) -> None:
        """  """
        self._env.set(req.name, req.value)
        
        # Notify subscribed plugins
        if req.name in self._env_subscribers:
            for plugin_id in self._env_subscribers[req.name]:
                if plugin_id in self._plugins:
                    rpc = self._plugins[plugin_id].rpc
                    rpc.env_update(req.name)
        
    
    def _respond_env_get(self, rpc: PluginRPC, req: RequestEnvGet) -> Any:
        """
        Get an environment variable.
        
        Args:
            key: Variable key
            default: Default value if not found
        
        Returns:
            Variable value
        """
        return self._env.get(req.name)
    
    def _respond_env_get_all(self, rpc: PluginRPC, req: RequestEnvGetAll) -> Dict[str, Any]:
        """
        Get all environment variables.
        
        Returns:
            Dictionary of all variables
        """
        return self._env.get_all()
    
    
    def _respond_env_subscribe(self, rpc: PluginRPC, req: RequestEnvSubscribe) -> bool:
        """
        Subscribe a plugin to an environment variable.
        """
        plugin_id = req.sender
        var_name = req.name

        if plugin_id not in self._plugins:
            return False

        if var_name not in self._env_subscribers:
            self._env_subscribers[var_name] = []

        if plugin_id not in self._env_subscribers[var_name]:
            self._env_subscribers[var_name].append(plugin_id)

        return True

    def _respond_env_unsubscribe(self, rpc: PluginRPC, req: RequestEnvUnsubscribe) -> bool:
        """
        Unsubscribe a plugin from an environment variable.
        """
        if req.sender not in self._plugins:
            return False

        if req.name in self._env_subscribers and req.sender in self._env_subscribers[req.name]:
            self._env_subscribers[req.name].remove(req.sender)
            if not self._env_subscribers[req.name]:
                del self._env_subscribers[req.name]
        return True

    # ========================================================================
    # TOPIC PUB/SUB
    # ========================================================================
    

    def _respond_topic_subscribe(self, rpc: PluginRPC, req: RequestTopicSubscribe) -> bool:
        """
        Subscribe a plugin to a topic.
        Allow subscription also for non (yet, hopefully) existing topics.
        """
        
        sub  = req.subscriber
        topic = sub.topic
        plugin_from_id = req.sender
        plugin_to_id = sub.plugin_id

        if plugin_from_id not in self._plugins: return False
        if plugin_to_id not in self._plugins: return False

        forward_rpc = self._plugins[plugin_to_id].rpc
        remote_result = forward_rpc.forward(req)

        if not remote_result: return False

        if topic not in self._topic_subscribers:
            self._topic_subscribers[topic] = []

        if plugin_from_id not in self._topic_subscribers[topic]:
            self._topic_subscribers[topic].append(plugin_from_id)

        
        return True


    def _respond_topic_unsubscribe(self, rpc: PluginRPC, req: RequestTopicUnsubscribe) -> bool:
        """
        Unsubscribe a plugin from a topic.
        
        Args:
            plugin_id: Plugin identifier
            topic: Topic name
        
        Returns:
            True if unsubscription successful
        """
        sub  = req.subscriber
        topic = sub.topic
        src_id = sub.plugin_id

        if src_id not in self._plugins: return False
        if topic not in self._topic_publishers: return False

        dst_id = self._topic_publishers[topic]

        forward_rpc = self._plugins[dst_id].rpc
        remote_result = forward_rpc.forward(req)

        if not remote_result: return False
        

        if src_id in self._topic_subscribers[topic]:
            self._topic_subscribers[topic].remove(src_id)
            if not self._topic_subscribers[topic]:
                del self._topic_subscribers[topic]


        return True

    def _respond_topic_announce(self, rpc: PluginRPC, req: RequestTopicAnnounce) -> int:
        """
        Announce a new topic.
        
        """
        self._topic_publishers[req.topic] = req.sender

        if req.topic not in self._topic_subscribers:
            self._topic_subscribers[req.topic] = []
        subs = list(self._topic_subscribers[req.topic])

        return subs

    def _respond_topic_list(self) -> List[str]:
        """
        List all topics.
        
        Returns:
            List of all topics
        """
        return list(self._topic_subscribers.keys())