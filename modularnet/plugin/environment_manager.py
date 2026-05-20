"""
Environment State System: Manages shared environment variables across plugins.

Simple thread-safe system for reading/writing global environment variables
and broadcasting updates to interested plugins.
"""

from dataclasses import dataclass
from typing import Any, Dict
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
    """Thread-safe manager for shared environment variables."""
    
    def __init__(self):
        """Initialize environment manager."""
        self._env: Dict[str, Any] = {}
        self._lock = Lock()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get environment variable value."""
        with self._lock:
            return self._env.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set environment variable."""
        with self._lock:
            self._env[key] = value
    
    def get_all(self) -> Dict[str, Any]:
        """Get all environment variables."""
        with self._lock:
            return self._env.copy()
    
    def set_all(self, env_dict: Dict[str, Any]) -> None:
        """Set multiple environment variables."""
        for key, value in env_dict.items():
            self.set(key, value)
    
    def clear(self) -> None:
        """Clear all environment variables."""
        with self._lock:
            self._env.clear()
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in environment."""
        with self._lock:
            return key in self._env
