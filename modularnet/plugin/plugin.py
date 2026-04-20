
from torch import P

from modularnet.plugin.plugin_env import PluginEnv, PluginManagerEnv
from modularnet.plugin.plugin_pubsub_old import PluginManagerPubSub, PluginPubSub


class PluginManager(PluginManagerPubSub, PluginManagerEnv):
    """Backward-compatible alias for PluginManagerPubSub with full pub/sub support."""
    pass
            



class Plugin(PluginPubSub, PluginEnv):
    """Backward-compatible alias for PluginManagerPubSub with full pub/sub support."""
    pass
            

