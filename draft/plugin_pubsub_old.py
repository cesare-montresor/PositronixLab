from torch import multiprocessing as mp
import time
from typing import Callable, Dict


from modularnet.plugin.plugin_base import PluginBase
from modularnet.plugin.plugin_manager import PluginManagerBase
from modularnet.task.task import Task

class PluginTopic:
    """Manages a single pub/sub topic with multiprocessing queues."""
    def __init__(self, name: str):
        self.name = name
        self.subscribers: Dict[str, mp.Queue] = {}  # plugin_name -> queue
        self.publisher_queue = mp.Queue()  # Broadcasted to all subscribers
    
    def add_subscriber(self, plugin_name: str, queue: mp.Queue):
        """Register a subscriber queue for a plugin."""
        self.subscribers[plugin_name] = queue
    
    def remove_subscriber(self, plugin_name: str):
        """Unregister a subscriber."""
        if plugin_name in self.subscribers:
            del self.subscribers[plugin_name]
    
    def broadcast(self, data):
        """Put data in all subscriber queues (non-blocking)."""
        for queue in self.subscribers.values():
            try:
                queue.put_nowait(data)
            except:
                pass  # Queue full, skip


class PluginPubSub(PluginBase):
    """Base class for plugins with pub/sub capabilities."""
    def __init__(self, options: dict={}, name: str = None, fatal_errors: bool = False):
        super().__init__(options, name, fatal_errors)
        
        self.subscriptions: Dict[str, Callable] = {}  # topic_name -> callback
        self.reception_queues: Dict[str, mp.Queue] = {}  # topic_name -> queue
        self.task_receiver = None

    def set_manager(self, manager: PluginManagerPubSub):
        self.manager = manager
        
    def subscribe(self, topic_name: str, callback: Callable):
        """Subscribe to a topic with a callback function.
        
        The callback will be called with the message data whenever
        a message is published to this topic.
        """
        self.subscriptions[topic_name] = callback

    def publish(self, topic_name: str, data):
        """Publish data to a topic (non-blocking)."""
        if self.manager:
            self.manager.publish_to_topic(topic_name, data)
    
    def run_main(self, options: dict={}, **kwargs):
        """Override to set up receiver thread in the plugin process."""
        self.options = options
        if 'fatal_errors' in kwargs:
            self.fatal_errors = kwargs['fatal_errors']
        
        # Setup reception queues in the plugin process
        if self.manager:
            for topic_name in self.subscriptions.keys():
                queue = self.manager.get_topic_queue(topic_name, self.name)
                self.reception_queues[topic_name] = queue
        
        # Start receiver thread in this process
        self.task_receiver = Task.create(target=self._receive_messages, mode="thread")
        self.task_receiver.start()
        
        # Run the main loop
        self.onStart()
        
        while not self.stop_event.is_set():
            try:
                self.onUpdate()
            except Exception as e:
                print(f"{self.__class__.__name__} encountered an error: {e}")
                import traceback
                traceback.print_exc()
                if self.fatal_errors:
                    self.stop_event.set()
                    raise e
            time.sleep(0.001)  # Prevent busy waiting
        
        self.onStop()
        if self.task_receiver:
            self.task_receiver.join()
    
    def _receive_messages(self):
        """Continuously receive and dispatch messages from subscription queues."""
        while not self.stop_event.is_set():
            for topic_name, callback in self.subscriptions.items():
                if topic_name in self.reception_queues:
                    queue = self.reception_queues[topic_name]
                    try:
                        data = queue.get(timeout=0.1)
                        callback(data)
                    except:
                        pass  # Timeout or empty queue
            time.sleep(0.01)




class PluginManagerPubSub(PluginManagerBase):
    """Manager for registering plugins and handling pub/sub topics."""
    def __init__(self):
        super().__init__()
        self.topics: Dict[str, PluginTopic] = {}
        
        self._mp_manager = None  # Multiprocessing Manager for shared state
        self._shared_state = None  # Shared state dict: {topic_name: {plugin_name: queue}}
    
    def _init_shared_state(self):
        """Initialize shared multiprocessing state if not already done."""
        if self._mp_manager is None:
            self._mp_manager = mp.Manager()
            self._shared_state = self._mp_manager.dict()
    
    def register_plugin(self, plugin: PluginBase, name: str = None):
        """Register a plugin with the manager."""
        if name is None:
            name = plugin.name
        self.plugins[name] = plugin
        
        # Initialize shared state
        self._init_shared_state()
        
        # For PubSub plugins, register subscriptions
        if isinstance(plugin, PluginPubSub):
            for topic_name in plugin.subscriptions.keys():
                self._register_topic_subscription(plugin.name, topic_name)
        
        # Pass manager to plugin
        plugin.set_manager(self)
    
    def _register_topic_subscription(self, plugin_name: str, topic_name: str):
        """Register a subscription and create a queue for it."""
        if topic_name not in self._shared_state:
            self._shared_state[topic_name] = self._mp_manager.dict()
        
        if plugin_name not in self._shared_state[topic_name]:
            queue = self._mp_manager.Queue()
            self._shared_state[topic_name][plugin_name] = queue
    
    def get_topic(self, topic_name: str) -> PluginTopic:
        """Get or create a topic."""
        if topic_name not in self.topics:
            self.topics[topic_name] = PluginTopic(topic_name)
        return self.topics[topic_name]
    
    def get_topic_queue(self, topic_name: str, plugin_name: str) -> mp.Queue:
        """Get or create a reception queue for a plugin on a topic."""
        self._init_shared_state()
        
        if topic_name not in self._shared_state:
            self._shared_state[topic_name] = self._mp_manager.dict()
        
        if plugin_name not in self._shared_state[topic_name]:
            queue = self._mp_manager.Queue()
            self._shared_state[topic_name][plugin_name] = queue
            
        return self._shared_state[topic_name][plugin_name]
    
    def publish_to_topic(self, topic_name: str, data):
        """Publish data to all subscribers of a topic."""
        self._init_shared_state()
        
        if topic_name in self._shared_state:
            for queue in self._shared_state[topic_name].values():
                try:
                    queue.put(data, block=False)
                except:
                    pass  # Queue full or other error

