"""
Plugin Manager: Central orchestration for the Ray-based plugin system.

Responsibilities:
1. Plugin lifecycle management (load, start, stop, unload)
2. Pub/sub topic routing
3. Environment variable management and broadcasting
4. Service request/response dispatch
5. Plugin status monitoring
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple
from threading import Thread, Event
import time

# Disable Ray dashboard BEFORE importing Ray
os.environ["RAY_DISABLE_DASHBOARD"] = "1"


import ray
from ray.util.queue import Queue

from .plugin_descriptor import PluginDescriptor
from .plugin_handler import PluginHandler
from .environment_manager import EnvironmentManager, EnvUpdate

logger = logging.getLogger(__name__)


@ray.remote
class PluginOrchestrator:
    """
    Ray actor that orchestrates all plugin management.
    
    Provides:
    - Plugin loading, starting, stopping
    - Pub/sub message routing
    - Environment variable management
    - Service invocation
    - Status monitoring
    """
    
    def __init__(self):
        """Initialize plugin orchestrator."""
        self._plugins: Dict[str, ray.actor.ActorHandle] = {}
        self._topics: Dict[str, List[Queue]] = {}
        self._env_subscriptions: Dict[str, List[str]] = {}  # key -> [plugin_ids]
        self.env_manager = EnvironmentManager()
    
    def register_plugin(self, plugin_id: str, handle: ray.actor.ActorHandle) -> bool:
        """
        Register a plugin actor.
        
        Args:
            plugin_id: Unique plugin identifier
            handle: Ray actor handle
            
        Returns:
            True if successful
        """
        if plugin_id in self._plugins:
            logger.warning(f"Plugin {plugin_id} already registered, overwriting")
        
        self._plugins[plugin_id] = handle
        logger.info(f"Plugin {plugin_id} registered")
        return True
    
    def unregister_plugin(self, plugin_id: str) -> bool:
        """
        Unregister a plugin actor.
        
        Args:
            plugin_id: Plugin identifier
            
        Returns:
            True if successful
        """
        if plugin_id not in self._plugins:
            logger.warning(f"Plugin {plugin_id} not found")
            return False
        
        del self._plugins[plugin_id]
        
        # Clean up env subscriptions
        for key in list(self._env_subscriptions.keys()):
            if plugin_id in self._env_subscriptions[key]:
                self._env_subscriptions[key].remove(plugin_id)
        
        logger.info(f"Plugin {plugin_id} unregistered")
        return True
    
    # --- Pub/Sub Topic Management ---
    
    def subscribe(self, topic: str, max_backlog: int = 100) -> Queue:
        """
        Create a subscription to a pub/sub topic.
        
        Args:
            topic: Topic name
            max_backlog: Maximum queued messages
            
        Returns:
            Queue for receiving messages
        """
        if topic not in self._topics:
            self._topics[topic] = []
        
        sub_queue = Queue(maxsize=max_backlog)
        self._topics[topic].append(sub_queue)
        
        logger.debug(f"New subscription to topic '{topic}'")
        return sub_queue
    
    def publish(self, topic: str, msg: Any) -> int:
        """
        Publish message to a topic.
        
        Args:
            topic: Topic name
            msg: Message (can include ObjectRef for high-bandwidth data)
            
        Returns:
            Number of subscribers that received the message
        """
        if topic not in self._topics:
            return 0
        
        delivered = 0
        failed = 0
        
        for q in self._topics[topic]:
            try:
                q.put(msg, block=False)
                delivered += 1
            except Exception:
                # Queue full - drop message (backpressure)
                failed += 1
        
        if failed > 0:
            logger.debug(f"Published to '{topic}': {delivered} delivered, {failed} dropped")
        
        return delivered
    
    # --- Environment Variable Management ---
    
    def subscribe_env(self, key: str, plugin_id: str) -> bool:
        """
        Subscribe plugin to environment variable changes.
        
        Args:
            key: Environment variable key
            plugin_id: Plugin identifier
            
        Returns:
            True if successful
        """
        if key not in self._env_subscriptions:
            self._env_subscriptions[key] = []
        
        if plugin_id not in self._env_subscriptions[key]:
            self._env_subscriptions[key].append(plugin_id)
        
        logger.debug(f"Plugin {plugin_id} subscribed to env var '{key}'")
        return True
    
    def set_env_variable(self, key: str, value: Any) -> None:
        """
        Set environment variable and notify subscribers.
        
        Args:
            key: Variable name
            value: New value
        """
        old_value = self.env_manager.get(key)
        self.env_manager.set(key, value)
        
        # Notify subscribed plugins
        if key in self._env_subscriptions:
            update = EnvUpdate(key=key, value=value, old_value=old_value)
            for plugin_id in self._env_subscriptions[key]:
                if plugin_id in self._plugins:
                    self._plugins[plugin_id].receive_env_update.remote(update)
    
    def get_env_variable(self, key: str, default: Any = None) -> Any:
        """Get environment variable value."""
        return self.env_manager.get(key, default)
    
    def get_all_env_variables(self) -> Dict[str, Any]:
        """Get all environment variables."""
        return self.env_manager.get_all()
    
    # --- Service Invocation ---
    
    def invoke_service(
        self, 
        plugin_id: str, 
        method_name: str, 
        args: tuple = (), 
        kwargs: dict = None
    ) -> Any:
        """
        Invoke a service method on a plugin.
        
        Args:
            plugin_id: Plugin identifier
            method_name: Method name
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            Method return value
        """
        if plugin_id not in self._plugins:
            raise ValueError(f"Plugin {plugin_id} not found")
        
        kwargs = kwargs or {}
        handle = self._plugins[plugin_id]
        result = ray.get(handle.execute_service.remote(method_name, *args, **kwargs))
        return result
    
    # --- Status and Monitoring ---
    
    def get_plugin_status(self, plugin_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific plugin."""
        if plugin_id not in self._plugins:
            return None
        
        handle = self._plugins[plugin_id]
        return ray.get(handle.get_status.remote())
    
    def get_all_plugin_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all plugins."""
        statuses = {}
        for plugin_id, handle in self._plugins.items():
            try:
                status = ray.get(handle.get_status.remote())
                statuses[plugin_id] = status
            except Exception as e:
                logger.error(f"Error getting status for {plugin_id}: {e}")
                statuses[plugin_id] = {"status": "ERROR", "error": str(e)}
        
        return statuses
    
    def get_plugin_errors(self, plugin_id: str) -> List[str]:
        """Get recent errors from a plugin."""
        if plugin_id not in self._plugins:
            return []
        
        handle = self._plugins[plugin_id]
        return ray.get(handle.get_errors.remote())
    
    def list_plugins(self) -> List[str]:
        """List all registered plugin IDs."""
        return list(self._plugins.keys())


class PluginManager:
    """
    High-level plugin manager interface.
    
    Provides user-friendly methods for managing plugins:
    - Load/start/stop/unload plugins
    - Publish/subscribe to topics
    - Get/set environment variables
    - Invoke services
    - Monitor plugin status
    """
    
    def __init__(self, auto_start_ray: bool = True):
        """
        Initialize plugin manager.
        
        Args:
            auto_start_ray: Automatically initialize Ray if not already initialized
        """
        if auto_start_ray and not ray.is_initialized():
            try:
                # Suppress Ray logging to reduce noise
                logging.getLogger("ray").setLevel(logging.WARNING)
                
                # Initialize Ray with minimal config
                # Note: Ray dashboard has pydantic v1 issues on Python 3.14,
                # so we disable it
                ray.init(
                    ignore_reinit_error=True,
                    include_dashboard=False
                )
            except Exception as e:
                logger.error(f"Failed to initialize Ray: {e}")
                raise
        
        self.orchestrator = PluginOrchestrator.remote()
        self._plugin_wrappers: Dict[str, ray.actor.ActorHandle] = {}
        self._processing_thread: Optional[Thread] = None
        self._stop_event = Event()
    
    def load_plugin(
        self,
        descriptor: PluginDescriptor,
        config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Load a plugin from descriptor.
        
        Args:
            descriptor: PluginDescriptor with package/class/version info
            config: Configuration options (overrides descriptor defaults)
            
        Returns:
            True if successful
        """
        plugin_id = descriptor.name
        config = config or {}
        
        try:
            # Create wrapper actor
            wrapper = PluginHandler.remote(plugin_id, self.orchestrator)
            
            # Register with orchestrator
            ray.get(self.orchestrator.register_plugin.remote(plugin_id, wrapper))
            
            # Load plugin into wrapper
            success = ray.get(wrapper.load_plugin.remote(descriptor.to_dict(), config))
            
            if success:
                self._plugin_wrappers[plugin_id] = wrapper
                logger.info(f"Plugin {descriptor} loaded successfully")
                return True
            else:
                logger.error(f"Failed to load plugin {descriptor}")
                ray.get(self.orchestrator.unregister_plugin.remote(plugin_id))
                return False
                
        except Exception as e:
            logger.error(f"Error loading plugin {descriptor}: {e}")
            return False
    
    def unload_plugin(self, plugin_id: str) -> bool:
        """
        Unload a plugin.
        
        Args:
            plugin_id: Plugin identifier
            
        Returns:
            True if successful
        """
        try:
            # Stop if running
            if plugin_id in self._plugin_wrappers:
                wrapper = self._plugin_wrappers[plugin_id]
                ray.get(wrapper.stop.remote())
            
            # Unregister from orchestrator
            ray.get(self.orchestrator.unregister_plugin.remote(plugin_id))
            
            if plugin_id in self._plugin_wrappers:
                del self._plugin_wrappers[plugin_id]
            
            logger.info(f"Plugin {plugin_id} unloaded")
            return True
            
        except Exception as e:
            logger.error(f"Error unloading plugin {plugin_id}: {e}")
            return False
    
    def start_plugin(self, plugin_id: str) -> bool:
        """
        Start a loaded plugin.
        
        Args:
            plugin_id: Plugin identifier
            
        Returns:
            True if successful
        """
        if plugin_id not in self._plugin_wrappers:
            logger.error(f"Plugin {plugin_id} not loaded")
            return False
        
        try:
            wrapper = self._plugin_wrappers[plugin_id]
            success = ray.get(wrapper.start.remote())
            if success:
                logger.info(f"Plugin {plugin_id} started")
            return success
        except Exception as e:
            logger.error(f"Error starting plugin {plugin_id}: {e}")
            return False
    
    def stop_plugin(self, plugin_id: str) -> bool:
        """
        Stop a running plugin.
        
        Args:
            plugin_id: Plugin identifier
            
        Returns:
            True if successful
        """
        if plugin_id not in self._plugin_wrappers:
            return False
        
        try:
            wrapper = self._plugin_wrappers[plugin_id]
            success = ray.get(wrapper.stop.remote())
            if success:
                logger.info(f"Plugin {plugin_id} stopped")
            return success
        except Exception as e:
            logger.error(f"Error stopping plugin {plugin_id}: {e}")
            return False
    
    def update_plugin_config(self, plugin_id: str, config: Dict[str, Any]) -> bool:
        """
        Update plugin configuration.
        
        Args:
            plugin_id: Plugin identifier
            config: New configuration
            
        Returns:
            True if successful
        """
        if plugin_id not in self._plugin_wrappers:
            return False
        
        try:
            wrapper = self._plugin_wrappers[plugin_id]
            success = ray.get(wrapper.update_config.remote(config))
            if success:
                logger.info(f"Plugin {plugin_id} configuration updated")
            return success
        except Exception as e:
            logger.error(f"Error updating config for {plugin_id}: {e}")
            return False
    
    # --- Pub/Sub ---
    
    def publish(self, topic: str, data: Any) -> int:
        """
        Publish data to a topic.
        
        Args:
            topic: Topic name
            data: Data to publish (for large data, use ObjectRef)
            
        Returns:
            Number of subscribers that received the message
        """
        return ray.get(self.orchestrator.publish.remote(topic, data))
    
    def publish_object_ref(self, topic: str, data: Any) -> int:
        """
        Publish large data via ObjectRef (zero-copy).
        
        Args:
            topic: Topic name
            data: Data to publish (numpy array, tensor, etc.)
            
        Returns:
            Number of subscribers
        """
        data_ref = ray.put(data)
        return ray.get(self.orchestrator.publish.remote(
            topic, 
            {"data_ref": data_ref}
        ))
    
    # --- Environment Variables ---
    
    def set_env_variable(self, key: str, value: Any) -> None:
        """
        Set an environment variable (broadcast to all subscribers).
        
        Args:
            key: Variable name
            value: Value
        """
        ray.get(self.orchestrator.set_env_variable.remote(key, value))
    
    def get_env_variable(self, key: str, default: Any = None) -> Any:
        """Get an environment variable."""
        return ray.get(self.orchestrator.get_env_variable.remote(key, default))
    
    def get_all_env_variables(self) -> Dict[str, Any]:
        """Get all environment variables."""
        return ray.get(self.orchestrator.get_all_env_variables.remote())
    
    # --- Services ---
    
    def invoke_service(
        self, 
        plugin_id: str, 
        method_name: str, 
        *args, 
        **kwargs
    ) -> Any:
        """
        Invoke a service method on a plugin.
        
        Args:
            plugin_id: Plugin identifier
            method_name: Method name
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Method result
        """
        return ray.get(self.orchestrator.invoke_service.remote(
            plugin_id, 
            method_name, 
            args, 
            kwargs
        ))
    
    # --- Monitoring ---
    
    def get_plugin_status(self, plugin_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a plugin."""
        return ray.get(self.orchestrator.get_plugin_status.remote(plugin_id))
    
    def get_all_plugin_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all plugins."""
        return ray.get(self.orchestrator.get_all_plugin_status.remote())
    
    def get_plugin_errors(self, plugin_id: str) -> List[str]:
        """Get recent errors from a plugin."""
        return ray.get(self.orchestrator.get_plugin_errors.remote(plugin_id))
    
    def list_plugins(self) -> List[str]:
        """List all loaded plugin IDs."""
        return ray.get(self.orchestrator.list_plugins.remote())
    
    # --- Processing Loop ---
    
    def start_processing_loop(self, interval: float = 0.1) -> None:
        """
        Start background thread that processes messages.
        
        Args:
            interval: Processing interval in seconds
        """
        if self._processing_thread:
            logger.warning("Processing loop already running")
            return
        
        self._stop_event.clear()
        self._processing_thread = Thread(
            target=self._process_loop,
            args=(interval,),
            daemon=True
        )
        self._processing_thread.start()
        logger.info("Plugin processing loop started")
    
    def stop_processing_loop(self) -> None:
        """Stop the background processing thread."""
        if not self._processing_thread:
            return
        
        self._stop_event.set()
        self._processing_thread.join(timeout=5.0)
        self._processing_thread = None
        logger.info("Plugin processing loop stopped")
    
    def _process_loop(self, interval: float) -> None:
        """Background loop that processes messages for all plugins."""
        while not self._stop_event.is_set():
            try:
                for plugin_id, wrapper in self._plugin_wrappers.items():
                    try:
                        ray.get(wrapper.process_messages.remote(max_messages=10))
                    except Exception as e:
                        logger.error(f"Error processing messages for {plugin_id}: {e}")
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(interval)
    
    def shutdown(self) -> None:
        """Shutdown the plugin manager."""
        self.stop_processing_loop()
        
        # Stop all plugins
        for plugin_id in list(self._plugin_wrappers.keys()):
            self.unload_plugin(plugin_id)
        
        logger.info("Plugin manager shutdown complete")
