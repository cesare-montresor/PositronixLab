"""
Base Plugin Interface: Abstract base class for all plugins.

Defines the lifecycle, subscription, and service interfaces that all
plugins must implement or can override.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable
from enum import Enum


class PluginState(Enum):
    """Plugin lifecycle events."""
    STARTED = "started"
    UPDATED = "updated"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class Subscription:
    """Describes a subscription: pub/sub topic or environment variable."""
    name: str  # Topic name (pub/sub) or env var key
    callback: str  # Method name on plugin
    publish_to: Optional[str] = None  # Optional: republish results (pub/sub only)


class BasePlugin(ABC):
    """
    Abstract base class for all ModularNet plugins.
    
    Plugins run in isolated Ray actors and communicate via:
    1. Control channel: Load/unload/configure commands from manager
    2. Data plane: Pub/sub topics for high-bandwidth data exchange (via pub/sub callbacks)
    3. State plane: Shared environment variables with change notifications
    4. Services: Sync request/response calls
    
    Plugins can actively communicate with each other:
    - self.publish(topic, data)  # Publish messages to other plugins
    - self.set_env_variable(key, value)  # Set shared environment variables
    - self.get_env_variable(key)  # Get environment variable value
    - self.invoke_service(plugin_id, method, *args)  # Call other plugins
    
    Subclasses must implement:
    - on_start: Initialize plugin resources
    - on_stop: Clean up resources
    
    Optional overrides:
    - on_update: Handle configuration updates
    - on_error: Handle errors
    - get_subscriptions: Define pub/sub subscriptions
    - get_env_subscriptions: Define env var change subscriptions
    - get_services: Define RPC methods
    """
    
    def __init__(self, plugin_id: str, config: Optional[Dict[str, Any]] = None, orchestrator_handle: Optional[Any] = None):
        """
        Initialize plugin.
        
        Args:
            plugin_id: Unique identifier for this plugin instance
            config: Configuration options dictionary
            orchestrator_handle: Optional Ray handle to orchestrator for inter-plugin communication
        """
        self.plugin_id = plugin_id
        self.config = config or {}
        self._is_running = False
        self._orchestrator = orchestrator_handle  # Internal: Ray handle for communication
    
    @abstractmethod
    def on_start(self) -> None:
        """
        Called when plugin is started.
        
        Initialize resources, open connections, start background tasks, etc.
        Raise exception on failure.
        """
        pass
    
    @abstractmethod
    def on_stop(self) -> None:
        """
        Called when plugin is stopped.
        
        Clean up resources, close connections, stop background tasks, etc.
        """
        pass
    
    def on_update(self, config: Dict[str, Any]) -> None:
        """
        Called when plugin configuration is updated.
        
        Override to handle dynamic config changes. Default does nothing.
        
        Args:
            config: Updated configuration dictionary
        """
        self.config = config
    
    def on_error(self, error: Exception, context: str = "") -> None:
        """
        Called when an error occurs in the plugin.
        
        Override for custom error handling. Default does nothing.
        
        Args:
            error: The exception that occurred
            context: Context about where/when error occurred
        """
        pass
    
    def get_subscriptions(self) -> List[Subscription]:
        """
        Define pub/sub topic subscriptions.
        
        Return list of topics this plugin wants to subscribe to.
        When messages arrive, the specified callback method is called.
        
        Returns:
            List of subscriptions
        """
        return []
    
    def get_env_subscriptions(self) -> List[Subscription]:
        """
        Define environment variable subscriptions.
        
        Return list of env vars this plugin wants to watch for changes.
        When a var changes, the specified callback method is called.
        
        Returns:
            List of subscriptions
        """
        return []
    
    def get_services(self) -> List[str]:
        """
        Define service methods available for RPC calls.
        
        Return list of method names that can be called via services.
        These methods should be sync and return serializable results.
        
        Returns:
            List of method names
        """
        return []
    
    @property
    def is_running(self) -> bool:
        """Whether plugin is currently running."""
        return self._is_running
    
    # ========================================================================
    # Inter-Plugin Communication Methods
    # ========================================================================
    
    def publish(self, topic: str, data: Any) -> int:
        """
        Publish data to a topic (plugin → other plugins).
        
        Args:
            topic: Topic name
            data: Data to publish
            
        Returns:
            Number of subscribers that received the message
        """
        if not self._orchestrator:
            raise RuntimeError("Plugin communication not available (orchestrator not initialized)")
        try:
            import ray
            return ray.get(self._orchestrator.publish.remote(topic, data))
        except Exception as e:
            raise RuntimeError(f"Failed to publish to topic '{topic}': {e}")
    
    def publish_object_ref(self, topic: str, data: Any) -> int:
        """
        Publish large data via ObjectRef (zero-copy).
        
        Args:
            topic: Topic name
            data: Data to publish (numpy array, tensor, etc.)
            
        Returns:
            Number of subscribers
        """
        if not self._orchestrator:
            raise RuntimeError("Plugin communication not available (orchestrator not initialized)")
        try:
            import ray
            data_ref = ray.put(data)
            return ray.get(self._orchestrator.publish.remote(
                topic,
                {"data_ref": data_ref}
            ))
        except Exception as e:
            raise RuntimeError(f"Failed to publish object ref to '{topic}': {e}")
    
    def set_env_variable(self, key: str, value: Any) -> None:
        """
        Set environment variable (broadcast to all subscribers).
        
        Args:
            key: Variable name
            value: New value
        """
        if not self._orchestrator:
            raise RuntimeError("Plugin communication not available (orchestrator not initialized)")
        try:
            import ray
            ray.get(self._orchestrator.set_env_variable.remote(key, value))
        except Exception as e:
            raise RuntimeError(f"Failed to set env variable '{key}': {e}")
    
    def get_env_variable(self, key: str, default: Any = None) -> Any:
        """
        Get environment variable value.
        
        Args:
            key: Variable name
            default: Default value if not found
            
        Returns:
            Variable value or default
        """
        if not self._orchestrator:
            raise RuntimeError("Plugin communication not available (orchestrator not initialized)")
        try:
            import ray
            return ray.get(self._orchestrator.get_env_variable.remote(key, default))
        except Exception as e:
            raise RuntimeError(f"Failed to get env variable '{key}': {e}")
    
    def invoke_service(
        self,
        plugin_id: str,
        method_name: str,
        *args,
        **kwargs
    ) -> Any:
        """
        Invoke a service method on another plugin.
        
        Args:
            plugin_id: Target plugin identifier
            method_name: Method name to invoke
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Method return value
        """
        if not self._orchestrator:
            raise RuntimeError("Plugin communication not available (orchestrator not initialized)")
        try:
            import ray
            return ray.get(self._orchestrator.invoke_service.remote(
                plugin_id,
                method_name,
                args,
                kwargs
            ))
        except Exception as e:
            raise RuntimeError(
                f"Failed to invoke service '{method_name}' on plugin '{plugin_id}': {e}"
            )
