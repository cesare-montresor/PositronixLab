"""
Base Plugin Interface: Abstract base class for all plugins.

Defines the lifecycle, subscription, and service interfaces that all
plugins must implement or can override.
"""

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
from enum import Enum


from modularnet.common.utils import camel_to_kebab
from modularnet.plugin import PluginDescriptor


class PluginState(Enum):
    """Plugin lifecycle states."""
    INITIALIZED = "initialized"
    LOADED = "loaded"
    RUNNING = "running"
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
    
    Plugins run in isolated processes and communicate via:
    1. Control channel: Load/unload/configure commands from manager
    2. Data plane: Pub/sub topics for high-bandwidth data exchange (via pub/sub callbacks)
    3. State plane: Shared environment variables with change notifications
    4. Services: Sync request/response calls
    
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

    @classmethod
    def Descriptor(cls, version:str, author:str, description:str=None, name:str=None, options: Dict[str, Any] = None) -> PluginDescriptor:
        return PluginDescriptor(
            version=version,
            author=author,
            options=options or {},
            name=name or camel_to_kebab(cls.__name__),
            package=f"{cls.__module__}",
            class_name=cls.__name__,
            description=description or f"{cls.__doc__}",
            
        )

    def __init__(self, plugin_id: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize plugin.
        
        Args:
            plugin_id: Unique identifier for this plugin instance
            config: Configuration options dictionary
        """
        self.plugin_id = plugin_id
        self.config = config or {}
        self._is_running = False
    
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
