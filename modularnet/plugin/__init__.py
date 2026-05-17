"""
ModularNet Plugin System: Ray-based modular plugin architecture.

This package provides a lean, high-performance plugin system built on Ray
for distributed, multi-process plugin execution with pub/sub, environment
variables, and service invocation.

Main Components:
- PluginDescriptor: Serializable plugin metadata
- BasePlugin: Abstract base class for plugins
- PluginManager: High-level plugin management interface
- PluginOrchestrator: Low-level Ray orchestrator
"""

from .plugin_descriptor import PluginDescriptor
from .plugin_base import BasePlugin, PluginState, Subscription
from .environment_manager import EnvironmentManager, EnvUpdate
from .plugin_manager import PluginManager, PluginOrchestrator

__all__ = [
    "PluginDescriptor",
    "BasePlugin",
    "PluginState",
    "Subscription",
    "EnvironmentManager",
    "EnvUpdate",
    "PluginManager",
    "PluginOrchestrator",
]
