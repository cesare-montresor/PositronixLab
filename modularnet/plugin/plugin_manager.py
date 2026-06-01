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

from modularnet.plugin.plugin_base import PluginDescriptor
from modularnet.plugin.plugin_rpc import Channel, PluginLifeCycleRPC, PluginRPC, Response
from modularnet.plugin.plugin_worker import PluginWorker
from modularnet.plugin.plugin_message import  Commands, ProcessStatus, WorkerStatus
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
        self._topic_subscriptions: Dict[str, List[Any]] = {}        # topic -> [queue, queue, ...]
        self._env_subscriptions: Dict[str, List[str]] = {}  # env_key -> [plugin_id, ...]
        
        self._env = EnvironmentManager()
        
        self.timeout_request = timeout_request
        self.timeout_start = timeout_start
        
        # Manager-level requests from plugins (for manager proxy)
        self._manager_req_queue = mp.Queue()
        self._manager_resp_queues: Dict[str, mp.Queue] = {}  # plugin_id -> resp_queue
        
        # Start manager service handler thread
        
        self._manager_service_thread = threading.Thread(
            target=self._handle_manager_requests,
            daemon=True
        )
        self._manager_service_thread.start()
    
    # ========================================================================
    # LIFECYCLE: Load, Start, Stop, Unload
    # ========================================================================
    
    def _init_rpc_for_plugin(self, plugin_id: str) -> None:
        """Initialize RPC channels for a plugin."""
        if plugin_id not in self._plugins:
            raise ValueError(f"Plugin {plugin_id} not found")
        
        info = self._plugins[plugin_id]
        plugin_rpc = info.rpc
        
        plugin_rpc.on_request(Commands.PLUGIN_START, self._handle_plugin_start)
        plugin_rpc.on_request(Commands.PLUGIN_STOP, self._handle_plugin_start)
        plugin_rpc.on_request(Commands.PLUGIN_CALL, self.call_service)
        plugin_rpc.on_request(Commands.PLUGIN_STATUS, self.get_plugin_status)
        plugin_rpc.on_request(Commands.PLUGIN_STATUS_ALL, self.get_all_plugin_status)

        plugin_rpc.on_request(Commands.ENV_SET, self.env_set)
        plugin_rpc.on_request(Commands.ENV_GET, self.env_get)
        plugin_rpc.on_request(Commands.ENV_GET_ALL, self.env_get_all)

        plugin_rpc.on_request(Commands.TOPIC_PUBLISH, self.publish)





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
                for env, plugin_ids in self._env_subscriptions.items():
                    if plugin_id in plugin_ids:
                        plugin_ids.remove(plugin_id) #TODO: notify*
                    if len(plugin_ids) == 0:
                        del self._env_subscriptions[env]

                # Clean up topic subscriptions
                for topic, plugin_ids in self._topic_subscriptions.items():
                    if plugin_id in plugin_ids:
                        plugin_ids.remove(plugin_id) #TODO: notify*
                    if len(plugin_ids) == 0:
                        del self._topic_subscriptions[topic]

                
                
                del self._plugins[plugin_id]
            
            logger.info(f"Plugin {plugin_id} unloaded")
            return True
            
        except Exception as e:
            logger.error(f"Error unloading plugin {plugin_id}: {e}")
            return False
    
    
    # ========================================================================
    # PUB/SUB: Publish and Subscribe
    # ========================================================================
    
    def publish(self, topic: str, data: Any) -> int:
        """
        Publish data to a topic.
        
        Args:
            topic: Topic name
            data: Message data
        
        Returns:
            Number of subscribers that received the message
        """
        delivered = 0
        
        if topic in self._topic_subscriptions:
            for queue in self._topic_subscriptions[topic]:
                try:
                    queue.put_nowait(data)
                    delivered += 1
                except Exception as e:
                    logger.debug(f"Failed to deliver to topic '{topic}': {e}")
        
        return delivered
    
    # ========================================================================
    # ENVIRONMENT: Variables and Subscriptions
    # ========================================================================
    
    def env_get(self, key: str, value: Any) -> None:
        """
        Set an environment variable and notify subscribers.
        
        Args:
            key: Variable key
            value: New value
        """
        self._env.set(key, value)
        
        # Notify subscribed plugins
        if key in self._env_subscriptions:
            for plugin_id in self._env_subscriptions[key]:
                if plugin_id in self._plugins:
                    channel = self._plugins[plugin_id]["channel"]
                    channel.send("env_update", key)
    
    def env_set(self, key: str, default: Any = None) -> Any:
        """
        Get an environment variable.
        
        Args:
            key: Variable key
            default: Default value if not found
        
        Returns:
            Variable value
        """
        return self._env.get(key, default)
    
    def env_get_all(self) -> Dict[str, Any]:
        """
        Get all environment variables.
        
        Returns:
            Dictionary of all variables
        """
        return self._env.get_all()
    
    
    # ========================================================================
    # SERVICES: Invoke methods on plugins
    # ========================================================================
    
    def call_service(
        self,
        plugin_id: str,
        method_name: str,
        *args,
        **kwargs
    ) -> Any:
        
        if plugin_id not in self._plugins:
            raise ValueError(f"Plugin {plugin_id} not found")
        
        channel: Channel = self._plugins[plugin_id]["channel"]
        return channel.request(
            method_name,
            *args,
            timeout=self.timeout_request,
            **kwargs
        )
    
    # ========================================================================
    # STATUS AND MONITORING
    # ========================================================================
    
    def get_plugin_status(self, plugin_id: str) -> Optional[ProcessStatus]:
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
        process = info["process"]
        alive = process.is_alive()
        pid = process.pid
        worker_status = None
        
        if alive:
            try:
                status_dict = info["channel"].request("get_status", timeout=1.0)
                worker_status = WorkerStatus(**status_dict)
            except Exception as e:
                logger.debug(f"Failed to get status for {plugin_id}: {e}")
        
        return ProcessStatus(
            plugin_id=plugin_id,
            process_alive=alive,
            process_pid=pid,
            worker_status=worker_status
        )
    
    def get_all_plugin_status(self) -> Dict[str, ProcessStatus]:
        """
        Get status of all plugins.
        
        Returns:
            Dictionary of plugin_id -> ProcessStatus
        """
        return {
            pid: self.get_plugin_status(pid)
            for pid in self._plugins.keys()
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
    
