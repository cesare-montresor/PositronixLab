from abc import ABC, abstractmethod



class PluginBase(ABC):
    def __init__(self, name: str, version: str, plugin_version = None ):
        self.name = name
        self.version = version
        self.topics = []
        
    def subscribe_topic(self, topic_name: str):
        self.topics.append(topic_name)
        
    def manifest(self):
        return {
            "name": self.name,
            "version": self.version
        }


class PluginGuest(ABC,PluginBase):
    def __init__(self, name: str, version: str):
        super().__init__(name, version)
        
    @abstractmethod
    def execute_plugin(self, plugin_name, *args, **kwargs):
        pass
