from typing import Callable

from torch import multiprocessing as mp

from modularnet.plugin.plugin_base import PluginBase



class PluginTopic:
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.queue = mp.Queue()
        self.subscribers = []
    
    def subscribe(self, callback: Callable):
        self.subscribers.append(callback)

    def publish(self, data):
        self.queue.put(data)



class PluginTopicManager:
    def __init__(self):
        self.topics = {}

    def remove(self, name: str):
        if name in self.topics:
            del self.topics[name]

    def get(self, name: str) -> PluginTopic | None:
        topic = self.topics.get(name, None)
        if topic is None:
            self.topics[name] = PluginTopic(name)
        
        return self.topics.get(name)
    
    def subscribe(self, topic_name: str, callback: Callable):
        topic = self.get(topic_name)
        topic.subscribe(callback)

    def publish(self, topic_name: str, data):
        topic = self.get(topic_name)
        topic.publish(data)