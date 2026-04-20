"""
Backward-compatible aliases for the unified plugin system.

These provide convenient imports for the main Plugin and PluginManager classes
that combine pub/sub and environment variable functionality.
"""

from modularnet.plugin.plugin_env import PluginEnv, PluginManagerEnv
from modularnet.plugin.plugin_pubsub import PluginManagerPubSub, PluginPubSub


class PluginManager(PluginManagerPubSub, PluginManagerEnv):
    """
    Unified plugin manager with full pub/sub and environment variable support.
    
    Combines:
    - PluginManagerPubSub: unified pub/sub with automatic transport detection
    - PluginManagerEnv: environment variable registry and synchronization
    """
    pass


class Plugin(PluginPubSub, PluginEnv):
    """
    Unified plugin base class with full pub/sub and environment variable support.
    
    Combines:
    - PluginPubSub: unified pub/sub with automatic transport detection
    - PluginEnv: environment variable get/set/watch capabilities
    """
    pass

