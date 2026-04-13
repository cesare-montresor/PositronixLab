

from modularnet.plugin.plugin_base import PluginBase
from modularnet.plugin.plugin_topic import PluginTopicManager


class PluginManager:
    def __init__(self):
        self.plugins = {}
        self.topics = PluginTopicManager()

    def register_plugin(self, name, plugin:PluginBase):
        plugin

        self.plugins[name] = plugin

    def get_plugin(self, name):
        return self.plugins.get(name)

    def list_plugins(self):
        return list(self.plugins.keys())
    
    def get_topic(self, topic_name: str):
        return self.topics.get(topic_name)
