"""
Plugin Descriptor: Serializable metadata for dynamic plugin loading.

Describes a plugin's identity, location, and configuration options.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional
import json


@dataclass
class PluginDescriptor:
    """
    Serializable descriptor for plugin metadata and instantiation.
    
    Attributes:
        name: Unique plugin identifier
        version: Semantic version (e.g., "1.0.0")
        package: Full package path (e.g., "modularnet.plugins.my_plugin")
        class_name: Plugin class name in the package
        description: Human-readable description
        author: Plugin author
        options: Default configuration options for the plugin
        dependencies: List of required plugin names
    """
    name: str
    version: str
    package: str
    class_name: str
    description: str = ""
    author: str = ""
    options: Dict[str, Any] = field(default_factory=dict)
    dependencies: list = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert descriptor to dictionary (JSON-serializable)."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Serialize descriptor to JSON."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PluginDescriptor":
        """Create descriptor from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> "PluginDescriptor":
        """Create descriptor from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def __str__(self) -> str:
        return f"{self.name}@{self.version} ({self.package}.{self.class_name})"
    
    def __repr__(self) -> str:
        return f"PluginDescriptor({self})"
