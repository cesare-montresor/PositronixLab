"""
Plugin Manager: Lightweight multiprocessing-based plugin orchestration.

Manages plugin lifecycle, pub/sub messaging, environment variables, and services.
Uses simplified request/response protocol via Channel.
"""

import logging
import multiprocessing as mp
import time
from typing import Any, Dict, List, Optional

from modularnet.plugin.plugin_descriptor import PluginDescriptor
from modularnet.plugin.plugin_worker import PluginWorker
from modularnet.plugin.plugin_messages import Channel, Request, Response, DataExchange, SubscriptionType, EnvUpdate, ProcessStatus, WorkerStatus
from modularnet.plugin.environment_manager import EnvironmentManager

logger = logging.getLogger(__name__)

# Set fork mode for queue sharing
try:
    mp.set_start_method('fork', force=True)
except RuntimeError:
    pass


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
        self._plugins: Dict[str, Dict[str, Any]] = {}  # plugin_id -> {process, channel, topics}
        self._topics: Dict[str, List[Any]] = {}        # topic -> [queue, queue, ...]
        self._env_subscriptions: Dict[str, List[str]] = {}  # env_key -> [plugin_id, ...]
        
        self.env_manager = EnvironmentManager()
        self._manager = mp.Manager()
        
        self.timeout_request = timeout_request
        self.timeout_start = timeout_start
    
    # ========================================================================
    # LIFECYCLE: Load, Start, Stop, Unload
    # ========================================================================
    
    def load_plugin(
        self,
        descriptor: PluginDescriptor,
        config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Load and initialize a plugin (spawn subprocess).
        
        Args:
            descriptor: Plugin descriptor
            config: Initial configuration
        
        Returns:
            True if loaded successfully
        """
        plugin_id = descriptor.name
        config = config or {}
        
        try:
            cmd_queue = mp.Queue()
            resp_queue = mp.Queue()
            channel = Channel(cmd_queue, resp_queue)
            
            # Spawn worker process
            process = mp.Process(
                target=PluginWorker.run,
                args=(plugin_id, descriptor.to_dict(), config, cmd_queue, resp_queue),
                name=f"plugin-{plugin_id}"
            )
            process.start()
            
            self._plugins[plugin_id] = {
                "process": process,
                "channel": channel,
                "topics": {},  # topic -> queue
                "descriptor": descriptor,
                "config": config
            }
            
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
            channel = info["channel"]
            
            # Small delay to ensure worker is ready
            time.sleep(0.1)
            
            # Get subscriptions from plugin
            data_exchanges = channel.request("get_data_exchanges", timeout=self.timeout_request)
            subscriptions = data_exchanges.get("subscriptions", [])
            
            # Register topic subscriptions
            for sub in subscriptions:
                if sub.sub_type == SubscriptionType.TOPIC:
                    topic_queue = self._manager.Queue()
                    channel.send("register_topic_subscription", sub.name, topic_queue)
                    
                    # Track subscription
                    if sub.name not in self._topics:
                        self._topics[sub.name] = []
                    self._topics[sub.name].append(topic_queue)
                    info["topics"][sub.name] = topic_queue
            
            # Register env var subscriptions
            for sub in subscriptions:
                if sub.sub_type == SubscriptionType.ENV_VAR:
                    if sub.name not in self._env_subscriptions:
                        self._env_subscriptions[sub.name] = []
                    self._env_subscriptions[sub.name].append(plugin_id)
            
            # Send start command
            channel.request("start", timeout=self.timeout_start)
            
            logger.info(f"Plugin {plugin_id} started")
            return True
            
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
            channel = info["channel"]
            channel.send("stop")
            logger.info(f"Plugin {plugin_id} stopped")
            return True
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
                process = info["process"]
                
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=2)
                    if process.is_alive():
                        process.kill()
                
                # Clean up environment subscriptions
                for env_key in list(self._env_subscriptions.keys()):
                    if plugin_id in self._env_subscriptions[env_key]:
                        self._env_subscriptions[env_key].remove(plugin_id)
                    if not self._env_subscriptions[env_key]:
                        del self._env_subscriptions[env_key]
                
                del self._plugins[plugin_id]
            
            logger.info(f"Plugin {plugin_id} unloaded")
            return True
            
        except Exception as e:
            logger.error(f"Error unloading plugin {plugin_id}: {e}")
            return False
    
    def update_plugin_config(self, plugin_id: str, config: Dict[str, Any]) -> bool:
        """
        Update plugin configuration.
        
        Args:
            plugin_id: Plugin identifier
            config: Updated configuration
        
        Returns:
            True if updated successfully
        """
        if plugin_id not in self._plugins:
            return False
        
        try:
            info = self._plugins[plugin_id]
            channel = info["channel"]
            channel.request("update_config", config, timeout=self.timeout_request)
            info["config"].update(config)
            logger.info(f"Plugin {plugin_id} config updated")
            return True
        except Exception as e:
            logger.error(f"Error updating config for {plugin_id}: {e}")
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
        
        if topic in self._topics:
            for queue in self._topics[topic]:
                try:
                    queue.put_nowait(data)
                    delivered += 1
                except Exception as e:
                    logger.debug(f"Failed to deliver to topic '{topic}': {e}")
        
        return delivered
    
    # ========================================================================
    # ENVIRONMENT: Variables and Subscriptions
    # ========================================================================
    
    def set_env_variable(self, key: str, value: Any) -> None:
        """
        Set an environment variable and notify subscribers.
        
        Args:
            key: Variable key
            value: New value
        """
        old_value = self.env_manager.get(key)
        self.env_manager.set(key, value)
        
        # Notify subscribed plugins
        if key in self._env_subscriptions:
            update = EnvUpdate(key=key, value=value, old_value=old_value)
            for plugin_id in self._env_subscriptions[key]:
                if plugin_id in self._plugins:
                    channel = self._plugins[plugin_id]["channel"]
                    channel.send("receive_env_update", update)
    
    def get_env_variable(self, key: str, default: Any = None) -> Any:
        """
        Get an environment variable.
        
        Args:
            key: Variable key
            default: Default value if not found
        
        Returns:
            Variable value
        """
        return self.env_manager.get(key, default)
    
    def get_all_env_variables(self) -> Dict[str, Any]:
        """
        Get all environment variables.
        
        Returns:
            Dictionary of all variables
        """
        return self.env_manager.get_all()
    
    # ========================================================================
    # SERVICES: Invoke methods on plugins
    # ========================================================================
    
    def invoke_service(
        self,
        plugin_id: str,
        method_name: str,
        *args,
        **kwargs
    ) -> Any:
        """
        Invoke a service method on a plugin (synchronous).
        
        Args:
            plugin_id: Plugin identifier
            method_name: Method name to invoke
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Method return value
        
        Raises:
            ValueError: If plugin not found
            RuntimeError: If service invocation fails
            TimeoutError: If request times out
        """
        if plugin_id not in self._plugins:
            raise ValueError(f"Plugin {plugin_id} not found")
        
        channel = self._plugins[plugin_id]["channel"]
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
    
    # ========================================================================
    # BACKWARD COMPATIBILITY
    # ========================================================================
    
    def start_processing_loop(self, interval: float = 0.1) -> None:
        """No-op for backward compatibility."""
        pass
    
    def stop_processing_loop(self) -> None:
        """No-op for backward compatibility."""
        pass
