"""
Integration tests for the complete ModularNet plugin system.
Tests all components working together: PluginBase, PluginPubSub, PluginEnv, and Transport.
"""

import pytest
import time
import torch
import torch.nn as nn
from torch import multiprocessing as mp

from modularnet.plugin.plugin import Plugin, PluginManager
from modularnet.plugin.plugin_pubsub import ActivationTap, TensorDeliveryMode


# ============================================================================
# Full System Integration Tests
# ============================================================================

class TestFullIntegration:
    """Test complete system with multiple components."""
    
    def test_trainer_with_monitoring(self):
        """Test a complete training scenario with pub/sub and env variables."""
        with mp.Manager() as manager:
            results = manager.list()
            
            class TrainerPlugin(Plugin):
                def __init__(self, shared):
                    super().__init__(name="Trainer")
                    self.shared = shared
                    self.loss = 1.0
                    self.env_declare("learning_rate", default=1e-3, watch=True,
                                   on_change=self.on_lr_changed)
                    self.lr = 1e-3
                
                def on_lr_changed(self, old_lr, new_lr):
                    self.lr = new_lr
                    self.shared.append(f"lr_changed_{new_lr}")
                
                def onUpdate(self):
                    self.loss *= 0.95  # Simulate training
                    self.publish("loss", self.loss)
                    
                    if self.loss < 0.1:
                        self.stop_event.set()
                    time.sleep(0.05)
            
            class MonitorPlugin(Plugin):
                def __init__(self, shared):
                    super().__init__(name="Monitor")
                    self.shared = shared
                    self.subscribe("loss", self.on_loss)
                    self.env_declare("learning_rate")
                    self.loss_history = []
                
                def on_loss(self, loss):
                    self.loss_history.append(loss)
                    self.shared.append(f"loss_{loss:.4f}")
                
                def onUpdate(self):
                    lr = self.env_get("learning_rate")
                    if len(self.loss_history) % 5 == 0 and len(self.loss_history) > 0:
                        # Could adjust LR based on loss
                        pass
                    time.sleep(0.05)
            
            pm = PluginManager()
            pm.env_register("learning_rate", default=1e-3)
            
            pm.register_plugin(TrainerPlugin(results))
            pm.register_plugin(MonitorPlugin(results))
            
            pm.start_all_plugins()
            time.sleep(0.3)
            pm.env_set("learning_rate", 5e-4)
            pm.stop_all_plugins()
            
            assert any("loss_" in str(item) for item in results)
            assert any("lr_changed" in str(item) for item in results)
    
    def test_data_pipeline(self):
        """Test a data processing pipeline with multiple stages."""
        with mp.Manager() as manager:
            pipeline_events = manager.list()
            
            class DataProducerPlugin(Plugin):
                def __init__(self, shared):
                    super().__init__(name="Producer")
                    self.shared = shared
                    self.batch_count = 0
                
                def onUpdate(self):
                    batch = torch.randn(32, 784)
                    self.publish("raw_data", batch)
                    self.batch_count += 1
                    self.shared.append(f"produced_{self.batch_count}")
                    if self.batch_count >= 3:
                        self.stop_event.set()
                    time.sleep(0.05)
            
            class ProcessorPlugin(Plugin):
                def __init__(self, shared):
                    super().__init__(name="Processor")
                    self.shared = shared
                    self.subscribe_tensor("raw_data", self.process,
                                        mode=TensorDeliveryMode.LATEST)
                    self.count = 0
                
                def process(self, tensor):
                    # Normalize
                    normalized = (tensor - tensor.mean()) / (tensor.std() + 1e-5)
                    self.publish("processed_data", normalized)
                    self.count += 1
                    self.shared.append(f"processed_{self.count}")
                
                def onUpdate(self):
                    time.sleep(0.05)
            
            class ConsumerPlugin(Plugin):
                def __init__(self, shared):
                    super().__init__(name="Consumer")
                    self.shared = shared
                    self.subscribe_tensor("processed_data", self.consume,
                                        mode=TensorDeliveryMode.LATEST)
                    self.count = 0
                
                def consume(self, tensor):
                    # Could save to disk or accumulate stats
                    self.count += 1
                    self.shared.append(f"consumed_{self.count}")
                
                def onUpdate(self):
                    time.sleep(0.05)
            
            pm = PluginManager()
            pm.register_plugin(DataProducerPlugin(pipeline_events))
            pm.register_plugin(ProcessorPlugin(pipeline_events))
            pm.register_plugin(ConsumerPlugin(pipeline_events))
            
            pm.start_all_plugins()
            pm.stop_all_plugins()
            
            assert any("produced_" in str(item) for item in pipeline_events)
            assert any("processed_" in str(item) for item in pipeline_events)
            assert any("consumed_" in str(item) for item in pipeline_events)
    
    def test_neural_network_monitoring(self):
        """Test monitoring neural network training with activation taps."""
        with mp.Manager() as manager:
            events = manager.list()
            
            class ActivationMonitor(Plugin):
                def __init__(self, shared):
                    super().__init__(name="Monitor")
                    self.shared = shared
                    self.subscribe_tensor("fc1_out", self.on_fc1,
                                        mode=TensorDeliveryMode.LATEST)
                
                def on_fc1(self, activation):
                    mean_val = float(activation.mean())
                    self.shared.append(f"fc1_mean_{mean_val:.4f}")
                
                def onUpdate(self):
                    time.sleep(0.1)
            
            pm = PluginManager()
            pm.register_plugin(ActivationMonitor(events))
            pm.start_all_plugins()
            
            # Create model
            model = nn.Sequential(
                nn.Linear(10, 20),
                nn.ReLU(),
                nn.Linear(20, 5)
            )
            
            # Attach activation tap
            tap = ActivationTap("fc1_out", pm, mode=TensorDeliveryMode.LATEST)
            handle = model[0].register_forward_hook(tap.hook)
            
            # Run forward passes
            with torch.no_grad():
                for i in range(3):
                    x = torch.randn(4, 10)
                    model(x)
                    time.sleep(0.1)
            
            handle.remove()
            pm.stop_all_plugins()
            
            # Should have captured activations
            assert any("fc1_mean_" in str(item) for item in events)
    
    def test_multiple_independent_pipelines(self):
        """Test multiple independent plugin pipelines running simultaneously."""
        with mp.Manager() as manager:
            results_a = manager.list()
            results_b = manager.list()
            
            class PipelineA_Producer(Plugin):
                def __init__(self, shared):
                    super().__init__(name="A_Prod")
                    self.shared = shared
                    self.counter = 0
                
                def onUpdate(self):
                    self.publish("pipeline_a", f"A_{self.counter}")
                    self.counter += 1
                    if self.counter >= 3:
                        self.stop_event.set()
                    time.sleep(0.05)
            
            class PipelineA_Consumer(Plugin):
                def __init__(self, shared):
                    super().__init__(name="A_Cons")
                    self.shared = shared
                    self.subscribe("pipeline_a", lambda d: self.shared.append(d))
            
            class PipelineB_Producer(Plugin):
                def __init__(self, shared):
                    super().__init__(name="B_Prod")
                    self.shared = shared
                    self.counter = 0
                
                def onUpdate(self):
                    self.publish("pipeline_b", f"B_{self.counter}")
                    self.counter += 1
                    if self.counter >= 3:
                        self.stop_event.set()
                    time.sleep(0.05)
            
            class PipelineB_Consumer(Plugin):
                def __init__(self, shared):
                    super().__init__(name="B_Cons")
                    self.shared = shared
                    self.subscribe("pipeline_b", lambda d: self.shared.append(d))
            
            pm = PluginManager()
            pm.register_plugin(PipelineA_Producer(results_a))
            pm.register_plugin(PipelineA_Consumer(results_a))
            pm.register_plugin(PipelineB_Producer(results_b))
            pm.register_plugin(PipelineB_Consumer(results_b))
            
            pm.start_all_plugins()
            pm.stop_all_plugins()
            
            # Both pipelines should have run independently
            assert len(results_a) > 0
            assert len(results_b) > 0
            assert all("A_" in str(item) for item in results_a)
            assert all("B_" in str(item) for item in results_b)


# ============================================================================
# Complex Scenario Tests
# ============================================================================

class TestComplexScenarios:
    """Test complex, realistic scenarios."""
    
    def test_model_training_with_distributed_logging(self):
        """Test model training with multiple loggers and monitors."""
        with mp.Manager() as manager:
            events = manager.list()
            
            class Trainer(Plugin):
                def __init__(self, shared):
                    super().__init__(name="Trainer")
                    self.shared = shared
                    self.env_declare("batch_size", default=32, watch=True)
                    self.epoch = 0
                
                def onUpdate(self):
                    loss = 1.0 / (self.epoch + 1)
                    acc = min(0.99, 1.0 - 1.0 / (self.epoch + 1))
                    
                    self.publish("metrics", {
                        "loss": loss,
                        "acc": acc,
                        "epoch": self.epoch
                    })
                    
                    self.epoch += 1
                    if self.epoch >= 3:
                        self.stop_event.set()
                    time.sleep(0.05)
            
            class LossLogger(Plugin):
                def __init__(self, shared):
                    super().__init__(name="LossLogger")
                    self.shared = shared
                    self.subscribe("metrics", self.log_loss)
                
                def log_loss(self, metrics):
                    if isinstance(metrics, dict) and "loss" in metrics:
                        self.shared.append(f"loss_logged_{metrics['loss']:.4f}")
            
            class AccLogger(Plugin):
                def __init__(self, shared):
                    super().__init__(name="AccLogger")
                    self.shared = shared
                    self.subscribe("metrics", self.log_acc)
                
                def log_acc(self, metrics):
                    if isinstance(metrics, dict) and "acc" in metrics:
                        self.shared.append(f"acc_logged_{metrics['acc']:.4f}")
            
            pm = PluginManager()
            pm.env_register("batch_size", default=32)
            
            pm.register_plugin(Trainer(events))
            pm.register_plugin(LossLogger(events))
            pm.register_plugin(AccLogger(events))
            
            pm.start_all_plugins()
            pm.stop_all_plugins()
            
            loss_logs = [item for item in events if "loss_logged_" in str(item)]
            acc_logs = [item for item in events if "acc_logged_" in str(item)]
            
            assert len(loss_logs) > 0
            assert len(acc_logs) > 0
    
    def test_hierarchical_processing(self):
        """Test hierarchical data processing."""
        with mp.Manager() as manager:
            trace = manager.list()
            
            class Stage1(Plugin):
                def __init__(self, shared):
                    super().__init__(name="Stage1")
                    self.shared = shared
                    self.counter = 0
                
                def onUpdate(self):
                    data = torch.randn(10)
                    self.publish("stage1_output", data)
                    self.counter += 1
                    if self.counter >= 2:
                        self.stop_event.set()
                    time.sleep(0.05)
            
            class Stage2(Plugin):
                def __init__(self, shared):
                    super().__init__(name="Stage2")
                    self.shared = shared
                    self.subscribe_tensor("stage1_output", self.process,
                                        mode=TensorDeliveryMode.LATEST)
                
                def process(self, data):
                    # Double it
                    result = data * 2
                    self.publish("stage2_output", result)
                    self.shared.append("stage2_processed")
            
            class Stage3(Plugin):
                def __init__(self, shared):
                    super().__init__(name="Stage3")
                    self.shared = shared
                    self.subscribe_tensor("stage2_output", self.process,
                                        mode=TensorDeliveryMode.LATEST)
                
                def process(self, data):
                    # Add 1
                    result = data + 1
                    self.publish("final_output", result)
                    self.shared.append("stage3_processed")
            
            pm = PluginManager()
            pm.register_plugin(Stage1(trace))
            pm.register_plugin(Stage2(trace))
            pm.register_plugin(Stage3(trace))
            
            pm.start_all_plugins()
            pm.stop_all_plugins()
            
            assert any("stage2_processed" in str(item) for item in trace)
            assert any("stage3_processed" in str(item) for item in trace)


# ============================================================================
# Stress Tests
# ============================================================================

class TestStressScenarios:
    """Test system under stress conditions."""
    
    def test_high_message_throughput(self):
        """Test handling high message throughput."""
        with mp.Manager() as manager:
            count = manager.list([0])
            
            class FastPublisher(Plugin):
                def __init__(self, shared_count):
                    super().__init__(name="FastPub")
                    self.counter = 0
                    self.shared_count = shared_count
                
                def onUpdate(self):
                    # Publish rapidly
                    for i in range(100):
                        self.publish("fast_topic", self.counter)
                        self.counter += 1
                    if self.counter >= 10000:
                        self.stop_event.set()
            
            class FastSubscriber(Plugin):
                def __init__(self, shared_count):
                    super().__init__(name="FastSub")
                    self.shared_count = shared_count
                    self.subscribe("fast_topic", self.on_msg)
                
                def on_msg(self, data):
                    self.shared_count[0] += 1
            
            pm = PluginManager()
            pm.register_plugin(FastPublisher(count))
            pm.register_plugin(FastSubscriber(count))
            
            pm.start_all_plugins()
            pm.stop_all_plugins()
            
            # Should have handled many messages
            assert count[0] > 100
    
    def test_many_topics(self):
        """Test system with many independent topics."""
        with mp.Manager() as manager:
            events = manager.list()
            
            num_topics = 10
            
            class MultiTopicPublisher(Plugin):
                def __init__(self, shared, num_topics):
                    super().__init__(name="MultiPub")
                    self.shared = shared
                    self.num_topics = num_topics
                    self.counter = 0
                
                def onUpdate(self):
                    for topic_id in range(self.num_topics):
                        self.publish(f"topic_{topic_id}", self.counter)
                    self.counter += 1
                    if self.counter >= 2:
                        self.stop_event.set()
                    time.sleep(0.05)
            
            class MultiTopicSubscriber(Plugin):
                def __init__(self, shared, topic_id):
                    super().__init__(name=f"Sub_{topic_id}")
                    self.shared = shared
                    self.topic_id = topic_id
                    self.subscribe(f"topic_{topic_id}", 
                                  lambda d: self.shared.append(f"topic_{topic_id}"))
            
            pm = PluginManager()
            pm.register_plugin(MultiTopicPublisher(events, num_topics))
            for topic_id in range(num_topics):
                pm.register_plugin(MultiTopicSubscriber(events, topic_id))
            
            pm.start_all_plugins()
            pm.stop_all_plugins()
            
            # All topics should have delivered messages
            for topic_id in range(num_topics):
                assert any(f"topic_{topic_id}" in str(item) for item in events)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
