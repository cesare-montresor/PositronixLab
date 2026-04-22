"""
Comprehensive tests for plugin_pubsub.py - Pub/Sub communication and tensor channels.
"""

import pytest
import time
import torch
import torch.nn as nn
from torch import multiprocessing as mp

from modularnet.plugin.plugin import PluginManager, Plugin
from modularnet.plugin.plugin_pubsub import (
    PluginPubSub, ActivationTap, TensorDeliveryMode
)


# ============================================================================
# Basic Pub/Sub Tests
# ============================================================================

class PublisherPlugin(Plugin):
    """Example plugin that publishes messages to a topic."""
    def __init__(self, shared_list=None):
        super().__init__(name="PublisherPlugin")
        self.counter = 0
        self.shared_list = shared_list
    
    def onUpdate(self):
        """Called repeatedly by the plugin's main loop."""
        self.publish("test_topic", f"Message {self.counter}")
        self.counter += 1
        if self.counter >= 5:
            self.stop_event.set()
        time.sleep(0.1)


class SubscriberPlugin(Plugin):
    """Example plugin that subscribes to messages."""
    def __init__(self, shared_list):
        super().__init__(name="SubscriberPlugin")
        self.shared_list = shared_list
        self.subscribe("test_topic", self.on_message_received)
    
    def on_message_received(self, data):
        """Callback when a message is received."""
        self.shared_list.append(data)
    
    def onUpdate(self):
        """Called repeatedly by the plugin's main loop."""
        time.sleep(0.05)


class TestBasicPubSub:
    """Test basic pub/sub communication."""
    
    def test_plugin_pubsub_basic(self):
        """Test basic pub/sub communication between plugins."""
        with mp.Manager() as manager:
            shared_messages = manager.list()
            
            pm = PluginManager()
            pub_plugin = PublisherPlugin(shared_list=shared_messages)
            sub_plugin = SubscriberPlugin(shared_messages)

            pm.register_plugin(pub_plugin)
            pm.register_plugin(sub_plugin)

            pm.start_all_plugins()
            pm.stop_all_plugins()
            
            received_count = len(shared_messages)
            assert received_count > 0, f"Subscriber should have received messages (got {received_count})"
    
    def test_multiple_subscribers(self):
        """Test multiple subscribers on same topic."""
        with mp.Manager() as manager:
            shared1 = manager.list()
            shared2 = manager.list()
            
            pm = PluginManager()
            pub = PublisherPlugin()
            sub1 = SubscriberPlugin(shared1)
            sub2 = SubscriberPlugin(shared2)
            
            pm.register_plugin(pub)
            pm.register_plugin(sub1)
            pm.register_plugin(sub2)
            
            pm.start_all_plugins()
            pm.stop_all_plugins()
            
            # Both subscribers should have received
            assert len(shared1) > 0
            assert len(shared2) > 0
    
    def test_multiple_topics(self):
        """Test multiple independent topics."""
        with mp.Manager() as manager:
            messages = manager.list()
            
            class MultiPublisher(Plugin):
                def __init__(self):
                    super().__init__(name="MultiPub")
                    self.counter = 0
                
                def onUpdate(self):
                    self.publish("topic_a", f"A{self.counter}")
                    self.publish("topic_b", f"B{self.counter}")
                    self.counter += 1
                    if self.counter >= 3:
                        self.stop_event.set()
                    time.sleep(0.05)
            
            class MultiSubscriber(Plugin):
                def __init__(self, topic, shared):
                    super().__init__(name=f"Sub_{topic}")
                    self.topic = topic
                    self.shared = shared
                    self.subscribe(topic, self.on_msg)
                
                def on_msg(self, data):
                    self.shared.append((self.topic, data))
                
                def onUpdate(self):
                    time.sleep(0.05)
            
            pm = PluginManager()
            pm.register_plugin(MultiPublisher())
            pm.register_plugin(MultiSubscriber("topic_a", messages))
            pm.register_plugin(MultiSubscriber("topic_b", messages))
            
            pm.start_all_plugins()
            pm.stop_all_plugins()
            
            topics_a = [item for item in messages if item[0] == "topic_a"]
            topics_b = [item for item in messages if item[0] == "topic_b"]
            
            assert len(topics_a) > 0
            assert len(topics_b) > 0


# ============================================================================
# Tensor Pub/Sub Tests
# ============================================================================

class TestTensorPubSub:
    """Test tensor-specific pub/sub."""
    
    def test_tensor_publish_subscribe(self):
        """Test publishing and subscribing to tensors."""
        with mp.Manager() as manager:
            received = manager.list()
            
            class TensorPublisher(Plugin):
                def __init__(self):
                    super().__init__(name="TensorPub")
                    self.counter = 0
                
                def onUpdate(self):
                    tensor = torch.full((5, 5), float(self.counter))
                    self.publish_tensor("tensor_topic", tensor)
                    self.counter += 1
                    if self.counter >= 3:
                        self.stop_event.set()
                    time.sleep(0.05)
            
            class TensorSubscriber(Plugin):
                def __init__(self, shared):
                    super().__init__(name="TensorSub")
                    self.shared = shared
                    self.subscribe_tensor("tensor_topic", self.on_tensor,
                                        mode=TensorDeliveryMode.LATEST)
                
                def on_tensor(self, tensor):
                    self.shared.append(float(tensor[0, 0]))
                
                def onUpdate(self):
                    time.sleep(0.05)
            
            pm = PluginManager()
            pm.register_plugin(TensorPublisher())
            pm.register_plugin(TensorSubscriber(received))
            
            pm.start_all_plugins()
            pm.stop_all_plugins()
            
            assert len(received) > 0
    
    def test_tensor_lazy_allocation(self):
        """Test lazy tensor buffer allocation."""
        with mp.Manager() as manager:
            received = manager.list()
            
            class LazyPublisher(Plugin):
                def __init__(self):
                    super().__init__(name="LazyPub")
                    self.counter = 0
                
                def onUpdate(self):
                    # Don't specify shape - let it be lazy
                    tensor = torch.randn(10, self.counter + 5)
                    try:
                        self.publish_tensor("lazy_topic", tensor)
                        received.append("success")
                    except:
                        received.append("failed")
                    
                    self.counter += 1
                    if self.counter >= 2:
                        self.stop_event.set()
                    time.sleep(0.05)
            
            class LazySubscriber(Plugin):
                def __init__(self, shared):
                    super().__init__(name="LazySub")
                    self.shared = shared
                    # Lazy allocation: shape=None
                    self.subscribe_tensor("lazy_topic", self.on_tensor,
                                        shape=None,
                                        mode=TensorDeliveryMode.LATEST)
                
                def on_tensor(self, tensor):
                    self.shared.append(tensor.shape)
                
                def onUpdate(self):
                    time.sleep(0.05)
            
            pm = PluginManager()
            pm.register_plugin(LazyPublisher())
            pm.register_plugin(LazySubscriber(received))
            
            pm.start_all_plugins()
            pm.stop_all_plugins()
    
    def test_tensor_sync_mode(self):
        """Test SYNC mode tensor delivery."""
        with mp.Manager() as manager:
            events = manager.list()
            
            class SyncPublisher(Plugin):
                def __init__(self, shared):
                    super().__init__(name="SyncPub")
                    self.shared = shared
                    self.counter = 0
                
                def onUpdate(self):
                    tensor = torch.full((3, 3), float(self.counter))
                    start = time.time()
                    self.publish_tensor("sync_topic", tensor)
                    elapsed = time.time() - start
                    # SYNC mode might block - record timing
                    self.counter += 1
                    if self.counter >= 2:
                        self.stop_event.set()
                    time.sleep(0.05)
            
            class SyncSubscriber(Plugin):
                def __init__(self, shared):
                    super().__init__(name="SyncSub")
                    self.shared = shared
                    self.subscribe_tensor("sync_topic", self.on_tensor,
                                        mode=TensorDeliveryMode.SYNC)
                
                def on_tensor(self, tensor):
                    self.shared.append("received")
                    channel = self.get_tensor_channel("sync_topic")
                    if channel:
                        channel.consumed()  # Signal done
                
                def onUpdate(self):
                    time.sleep(0.05)
            
            pm = PluginManager()
            pm.register_plugin(SyncPublisher(events))
            pm.register_plugin(SyncSubscriber(events))
            
            pm.start_all_plugins()
            pm.stop_all_plugins()
            
            # Should have received messages
            assert any("received" in str(item) for item in events)
    
    def test_tensor_latest_mode(self):
        """Test LATEST mode tensor delivery."""
        with mp.Manager() as manager:
            received = manager.list()
            
            class FastPublisher(Plugin):
                def __init__(self):
                    super().__init__(name="FastPub")
                    self.counter = 0
                
                def onUpdate(self):
                    tensor = torch.full((5, 5), float(self.counter))
                    self.publish_tensor("fast_topic", tensor)
                    self.counter += 1
                    if self.counter >= 100:
                        self.stop_event.set()
                    # No sleep - publish fast
            
            class SlowSubscriber(Plugin):
                def __init__(self, shared):
                    super().__init__(name="SlowSub")
                    self.shared = shared
                    self.subscribe_tensor("fast_topic", self.on_tensor,
                                        mode=TensorDeliveryMode.LATEST)
                
                def on_tensor(self, tensor):
                    self.shared.append(float(tensor[0, 0]))
                
                def onUpdate(self):
                    time.sleep(0.05)  # Slow reader
            
            pm = PluginManager()
            pm.register_plugin(FastPublisher())
            pm.register_plugin(SlowSubscriber(received))
            
            pm.start_all_plugins()
            pm.stop_all_plugins()
            
            # In LATEST mode, slow subscriber may skip frames
            # but should still get some
            assert len(received) > 0
            assert len(received) < 100  # Skipped frames


# ============================================================================
# Mixed Data Type Tests
# ============================================================================

class TestMixedDataTypes:
    """Test publishing different data types through pub/sub."""
    
    def test_publish_dict(self):
        """Test publishing dict."""
        with mp.Manager() as manager:
            received = manager.list()
            
            class DictPublisher(Plugin):
                def onUpdate(self):
                    data = {"loss": 0.5, "acc": 0.95}
                    self.publish("metrics", data)
                    self.stop_event.set()
            
            class DictSubscriber(Plugin):
                def __init__(self, shared):
                    super().__init__(name="DictSub")
                    self.shared = shared
                    self.subscribe("metrics", lambda d: self.shared.append(d["loss"]))
            
            pm = PluginManager()
            pm.register_plugin(DictPublisher(name="DictPub"))
            pm.register_plugin(DictSubscriber(received))
            
            pm.start_all_plugins()
            pm.stop_all_plugins()
            
            assert len(received) > 0
            assert 0.5 in received
    
    def test_publish_list(self):
        """Test publishing list."""
        with mp.Manager() as manager:
            received = manager.list()
            
            class ListPublisher(Plugin):
                def onUpdate(self):
                    data = [1, 2, 3, 4, 5]
                    self.publish("numbers", data)
                    self.stop_event.set()
            
            class ListSubscriber(Plugin):
                def __init__(self, shared):
                    super().__init__(name="ListSub")
                    self.shared = shared
                    self.subscribe("numbers", lambda d: self.shared.append(len(d)))
            
            pm = PluginManager()
            pm.register_plugin(ListPublisher(name="ListPub"))
            pm.register_plugin(ListSubscriber(received))
            
            pm.start_all_plugins()
            pm.stop_all_plugins()
            
            assert len(received) > 0
            assert 5 in received
    
    def test_publish_module(self):
        """Test publishing neural network module."""
        with mp.Manager() as manager:
            received = manager.list()
            
            class ModulePublisher(Plugin):
                def onUpdate(self):
                    model = nn.Linear(10, 5)
                    self.publish("model", model)
                    self.stop_event.set()
            
            class ModuleSubscriber(Plugin):
                def __init__(self, shared):
                    super().__init__(name="ModuleSub")
                    self.shared = shared
                    self.subscribe("model", self.on_model)
                
                def on_model(self, model):
                    if isinstance(model, nn.Module):
                        self.shared.append("received_module")
            
            pm = PluginManager()
            pm.register_plugin(ModulePublisher(name="ModulePub"))
            pm.register_plugin(ModuleSubscriber(received))
            
            pm.start_all_plugins()
            pm.stop_all_plugins()
            
            assert any("received_module" in str(item) for item in received)


# ============================================================================
# ActivationTap Tests
# ============================================================================

class TestActivationTap:
    """Test ActivationTap for forward hooks."""
    
    def test_activation_tap_basic(self):
        """Test basic activation tap."""
        with mp.Manager() as manager:
            received = manager.list()
            
            class TapSubscriber(Plugin):
                def __init__(self, shared):
                    super().__init__(name="TapSub")
                    self.shared = shared
                    self.subscribe_tensor("layer_out", self.on_activation,
                                        mode=TensorDeliveryMode.LATEST)
                
                def on_activation(self, activation):
                    self.shared.append(activation.shape)
            
            pm = PluginManager()
            pm.register_plugin(TapSubscriber(received))
            pm.start_all_plugins()
            
            # Create model and attach hook
            model = nn.Sequential(
                nn.Linear(10, 20),
                nn.ReLU(),
                nn.Linear(20, 5)
            )
            
            tap = ActivationTap("layer_out", pm, mode=TensorDeliveryMode.LATEST)
            handle = model[0].register_forward_hook(tap.hook)
            
            # Run forward pass
            with torch.no_grad():
                for _ in range(3):
                    x = torch.randn(2, 10)
                    model(x)
                    time.sleep(0.1)
            
            handle.remove()
            pm.stop_all_plugins()
            
            # Should have captured activations
            assert len(received) > 0


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestPubSubErrorHandling:
    """Test error handling in pub/sub."""
    
    def test_callback_exception_handling(self):
        """Test that callback exceptions don't crash plugin."""
        with mp.Manager() as manager:
            events = manager.list()
            
            class FaultySubscriber(Plugin):
                def __init__(self, shared):
                    super().__init__(name="FaultySub")
                    self.shared = shared
                    self.counter = 0
                    self.subscribe("data", self.faulty_callback)
                
                def faulty_callback(self, data):
                    self.counter += 1
                    if self.counter == 1:
                        raise ValueError("Intentional error")
                    self.shared.append("success")
                
                def onUpdate(self):
                    time.sleep(0.05)
            
            class Publisher(Plugin):
                def __init__(self):
                    super().__init__(name="Pub")
                    self.counter = 0
                
                def onUpdate(self):
                    self.publish("data", self.counter)
                    self.counter += 1
                    if self.counter >= 5:
                        self.stop_event.set()
                    time.sleep(0.05)
            
            pm = PluginManager()
            pm.register_plugin(Publisher())
            pm.register_plugin(FaultySubscriber(events))
            
            pm.start_all_plugins()
            pm.stop_all_plugins()
            
            # Plugin should have recovered and processed more messages
            assert any("success" in str(item) for item in events)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

