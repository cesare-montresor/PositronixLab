"""
Environment State System: Manages shared environment variables across plugins.

Provides a thread-safe system for:
1. Reading/writing global environment variables
2. Subscribing to variable change notifications
3. Broadcasting updates to all interested plugins
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Callable, Optional
from threading import Lock
import logging

logger = logging.getLogger(__name__)


@dataclass
class EnvUpdate:
    """Represents an environment variable update."""
    key: str
    value: Any
    old_value: Any = None


class EnvironmentManager:
    """
    Thread-safe manager for shared environment variables.
    
    All plugins have access to the same environment through this manager.
    Supports change notifications via callbacks.
    """
    
    def __init__(self):
        """Initialize environment manager."""
        self._env: Dict[str, Any] = {}
        self._lock = Lock()
        self._subscribers: Dict[str, List[Callable[[EnvUpdate], None]]] = {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get environment variable value.
        
        Args:
            key: Variable name
            default: Default value if key not found
            
        Returns:
            Variable value or default
        """
        with self._lock:
            return self._env.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set environment variable and notify subscribers.
        
        Args:
            key: Variable name
            value: New value
        """
        with self._lock:
            old_value = self._env.get(key)
            self._env[key] = value
            subscribers = self._subscribers.get(key, [])
        
        # Notify subscribers (outside lock to avoid deadlocks)
        if subscribers:
            update = EnvUpdate(key=key, value=value, old_value=old_value)
            for callback in subscribers:
                try:
                    callback(update)
                except Exception as e:
                    logger.error(f"Error in env change callback for '{key}': {e}")
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all environment variables.
        
        Returns:
            Dictionary of all variables
        """
        with self._lock:
            return self._env.copy()
    
    def set_all(self, env_dict: Dict[str, Any]) -> None:
        """
        Set multiple environment variables.
        
        Args:
            env_dict: Dictionary of key-value pairs
        """
        for key, value in env_dict.items():
            self.set(key, value)
    
    def subscribe(self, key: str, callback: Callable[[EnvUpdate], None]) -> Callable:
        """
        Subscribe to changes on an environment variable.
        
        Args:
            key: Variable name to watch
            callback: Function called when variable changes
            
        Returns:
            Unsubscribe function
        """
        with self._lock:
            if key not in self._subscribers:
                self._subscribers[key] = []
            self._subscribers[key].append(callback)
        
        # Return unsubscribe function
        def unsubscribe():
            with self._lock:
                if key in self._subscribers:
                    self._subscribers[key].remove(callback)
        
        return unsubscribe
    
    def clear(self) -> None:
        """Clear all environment variables and subscribers."""
        with self._lock:
            self._env.clear()
            self._subscribers.clear()
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in environment."""
        with self._lock:
            return key in self._env
