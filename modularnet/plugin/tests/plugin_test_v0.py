import pytest
import time
from torch import multiprocessing as mp

from modularnet.plugin.plugin import PluginManager, Plugin


class PublisherPlugin(Plugin):
    """Example plugin that publishes messages to a topic."""
    def __init__(self):
        super().__init__(name="PublisherPlugin")
        self.counter = 0
    
    def onUpdate(self):
        """Called repeatedly by the plugin's main loop."""
        self.publish("test_topic", f"Message {self.counter}")
        self.counter += 1
        time.sleep(0.5)  # Publish every 500ms


class SubscriberPlugin(Plugin):
    """Example plugin that subscribes to messages."""
    def __init__(self, shared_list):
        super().__init__(name="SubscriberPlugin")
        self.shared_list = shared_list
        # Subscribe to the topic with a callback
        self.subscribe("test_topic", self.on_message_received)
    
    def on_message_received(self, data):
        """Callback when a message is received."""
        print(f"SubscriberPlugin received: {data}")
        self.shared_list.append(data)
    
    def onUpdate(self):
        """Called repeatedly by the plugin's main loop."""
        time.sleep(0.1)  # Just idle


def test_plugin_pubsub():
    """Test pub/sub communication between plugins."""
    # Use multiprocessing-safe list to track messages
    with mp.Manager() as manager:
        shared_messages = manager.list()
        
        # Create manager and plugins
        pm = PluginManager()
        pub_plugin = PublisherPlugin()
        sub_plugin = SubscriberPlugin(shared_messages)

        # Register plugins
        pm.register_plugin(pub_plugin)
        pm.register_plugin(sub_plugin)

        # Start all plugins
        pm.start_all_plugins()
        
        # Let them run for a bit
        time.sleep(2.5)
        
        # Stop all plugins
        pm.stop_all_plugins()
        
        # Verify that subscriber received messages
        received_count = len(shared_messages)
        assert received_count > 0, f"Subscriber should have received messages (got {received_count})"
        print(f"✓ Subscriber received {received_count} messages")
        for msg in list(shared_messages):
            print(f"  - {msg}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

