# Plugin System: Practical Examples

This document provides real-world usage examples for the ModularNet plugin system.

## Example 1: Data Processing Pipeline

Multiple plugins processing data through a pipeline:

```python
from modularnet.plugin import BasePlugin, PluginDescriptor, PluginManager
from typing import Dict, Any, List

# Plugin 1: Data Source - generates data
class DataSourcePlugin(BasePlugin):
    def on_start(self):
        self.counter = 0
    
    def on_stop(self):
        pass
    
    def get_services(self) -> List[str]:
        return ["generate_batch"]
    
    def generate_batch(self, size: int = 10) -> Dict[str, Any]:
        """Generate a batch of data."""
        self.counter += 1
        return {
            "batch_id": self.counter,
            "data": list(range(size)),
            "size": size
        }

# Plugin 2: Data Processor - processes and republishes
class ProcessorPlugin(BasePlugin):
    def on_start(self):
        self.processed_count = 0
    
    def on_stop(self):
        pass
    
    def get_subscriptions(self):
        from modularnet.plugin import SubscriptionDescriptor
        return [
            SubscriptionDescriptor(
                topic="raw_data",
                callback="process",
                publish_to="processed_data"
            )
        ]
    
    def process(self, data: Dict[str, Any], env: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw data and republish."""
        self.processed_count += 1
        
        # Use learning_rate from environment if available
        lr = env.get("learning_rate", 0.001)
        
        return {
            "batch_id": data["batch_id"],
            "processed_data": [x * 2 for x in data["data"]],  # Simple processing
            "learning_rate": lr,
            "processed_at": self.processed_count
        }
    
    def get_services(self) -> List[str]:
        return ["get_statistics"]
    
    def get_statistics(self) -> Dict[str, int]:
        return {"processed_count": self.processed_count}

# Plugin 3: Data Consumer - receives and stores results
class ConsumerPlugin(BasePlugin):
    def on_start(self):
        self.results = []
    
    def on_stop(self):
        pass
    
    def get_subscriptions(self):
        from modularnet.plugin import SubscriptionDescriptor
        return [
            SubscriptionDescriptor(
                topic="processed_data",
                callback="consume"
            )
        ]
    
    def consume(self, data: Dict[str, Any], env: Dict[str, Any]):
        """Consume processed data."""
        self.results.append(data)
    
    def get_services(self) -> List[str]:
        return ["get_results"]
    
    def get_results(self) -> List[Dict[str, Any]]:
        return self.results.copy()

# Usage
if __name__ == "__main__":
    import time
    
    manager = PluginManager()
    
    # Load all plugins
    for name, cls in [
        ("source", "DataSourcePlugin"),
        ("processor", "ProcessorPlugin"),
        ("consumer", "ConsumerPlugin")
    ]:
        descriptor = PluginDescriptor(
            name=name,
            version="1.0.0",
            package=__name__,
            class_name=cls
        )
        manager.load_plugin(descriptor)
        manager.start_plugin(name)
    
    # Start processing loop
    manager.start_processing_loop(interval=0.05)
    
    # Generate batches
    for i in range(5):
        batch = manager.invoke_service("source", "generate_batch", 10)
        manager.publish("raw_data", batch)
        time.sleep(0.1)
    
    # Let processing complete
    time.sleep(1.0)
    
    # Get results
    results = manager.invoke_service("consumer", "get_results")
    print(f"Processed {len(results)} batches")
    for r in results:
        print(f"  Batch {r['batch_id']}: {r['processed_data']}")
    
    # Cleanup
    manager.stop_processing_loop()
    manager.shutdown()
```

## Example 2: Model Training with Dynamic Configuration

Plugin that trains a model with environment variable configuration:

```python
from modularnet.plugin import BasePlugin, EnvironmentSubscriptionDescriptor, EnvUpdate
from typing import Dict, Any, List
import numpy as np

class ModelTrainerPlugin(BasePlugin):
    """Plugin that trains a model with dynamic hyperparameter updates."""
    
    def __init__(self, plugin_id: str, config: Dict[str, Any] = None):
        super().__init__(plugin_id, config)
        self.learning_rate = 0.001
        self.training_enabled = True
        self.model_weights = np.random.randn(10)
        self.epoch = 0
        self.loss_history = []
    
    def on_start(self):
        print(f"Starting trainer with lr={self.learning_rate}")
    
    def on_stop(self):
        print(f"Stopped trainer after {self.epoch} epochs")
    
    def get_env_subscriptions(self) -> List[EnvironmentSubscriptionDescriptor]:
        """Subscribe to training parameters."""
        return [
            EnvironmentSubscriptionDescriptor("learning_rate", "on_lr_change"),
            EnvironmentSubscriptionDescriptor("training_enabled", "on_training_toggle"),
        ]
    
    def on_lr_change(self, update: EnvUpdate):
        """Handle learning rate changes."""
        self.learning_rate = update.value
        print(f"Learning rate: {update.old_value} → {update.value}")
    
    def on_training_toggle(self, update: EnvUpdate):
        """Handle training on/off."""
        self.training_enabled = update.value
        status = "enabled" if update.value else "disabled"
        print(f"Training {status}")
    
    def get_subscriptions(self):
        from modularnet.plugin import SubscriptionDescriptor
        return [
            SubscriptionDescriptor(
                topic="train_batch",
                callback="train_step"
            )
        ]
    
    def train_step(self, batch: Dict[str, Any], env: Dict[str, Any]):
        """Process one training batch."""
        if not self.training_enabled:
            return
        
        self.epoch += 1
        
        # Simulate training loss calculation
        loss = np.mean((self.model_weights - batch["target"]) ** 2)
        self.loss_history.append(loss)
        
        # Update weights (simulated)
        gradient = 2 * (self.model_weights - batch["target"])
        self.model_weights -= self.learning_rate * gradient
        
        if self.epoch % 100 == 0:
            print(f"Epoch {self.epoch}: loss={loss:.6f}")
    
    def get_services(self) -> List[str]:
        return ["get_loss_history", "get_weights", "reset"]
    
    def get_loss_history(self) -> List[float]:
        return self.loss_history.copy()
    
    def get_weights(self) -> np.ndarray:
        return self.model_weights.copy()
    
    def reset(self):
        """Reset training state."""
        self.model_weights = np.random.randn(10)
        self.epoch = 0
        self.loss_history = []
        print("Training state reset")

# Usage
if __name__ == "__main__":
    import time
    
    manager = PluginManager()
    
    # Load trainer
    descriptor = PluginDescriptor(
        name="trainer",
        version="1.0.0",
        package=__name__,
        class_name="ModelTrainerPlugin",
        options={"initial_lr": 0.001}
    )
    
    manager.load_plugin(descriptor)
    manager.start_plugin("trainer")
    
    # Set initial environment
    manager.set_env_variable("learning_rate", 0.001)
    manager.set_env_variable("training_enabled", True)
    
    # Start processing
    manager.start_processing_loop(interval=0.01)
    
    # Generate training batches
    for i in range(500):
        batch = {
            "id": i,
            "target": np.random.randn(10)
        }
        manager.publish("train_batch", batch)
    
    # Monitor and update hyperparameters
    time.sleep(1.0)
    
    print("\n--- Reducing learning rate ---")
    manager.set_env_variable("learning_rate", 0.0001)
    
    for i in range(500, 1000):
        batch = {"id": i, "target": np.random.randn(10)}
        manager.publish("train_batch", batch)
    
    time.sleep(1.0)
    
    # Pause training
    print("\n--- Pausing training ---")
    manager.set_env_variable("training_enabled", False)
    
    time.sleep(0.5)
    
    # Resume
    print("\n--- Resuming training ---")
    manager.set_env_variable("training_enabled", True)
    
    for i in range(1000, 1500):
        batch = {"id": i, "target": np.random.randn(10)}
        manager.publish("train_batch", batch)
    
    time.sleep(1.0)
    
    # Get results
    history = manager.invoke_service("trainer", "get_loss_history")
    print(f"\nTraining complete: {len(history)} batches")
    print(f"Initial loss: {history[0]:.6f}")
    print(f"Final loss: {history[-1]:.6f}")
    
    # Cleanup
    manager.stop_processing_loop()
    manager.shutdown()
```

## Example 3: Multi-Worker Distributed Inference

Multiple inference plugins consuming a task queue:

```python
from modularnet.plugin import BasePlugin, SubscriptionDescriptor
from typing import Dict, Any, List
import numpy as np

class InferenceWorker(BasePlugin):
    """Inference worker that processes tasks from a queue."""
    
    def __init__(self, plugin_id: str, config: Dict[str, Any] = None):
        super().__init__(plugin_id, config)
        self.model_id = config.get("model_id", "default") if config else "default"
        self.tasks_processed = 0
    
    def on_start(self):
        print(f"Worker {self.plugin_id} started with model {self.model_id}")
    
    def on_stop(self):
        print(f"Worker {self.plugin_id} processed {self.tasks_processed} tasks")
    
    def get_subscriptions(self) -> List[SubscriptionDescriptor]:
        return [
            SubscriptionDescriptor(
                topic="inference_tasks",
                callback="infer",
                publish_to="inference_results"
            )
        ]
    
    def infer(self, task: Dict[str, Any], env: Dict[str, Any]) -> Dict[str, Any]:
        """Process inference task."""
        self.tasks_processed += 1
        
        # Simulate inference
        task_id = task["id"]
        data = task["data"]
        result = {"prediction": np.sum(data) * 2}  # Simplified "model"
        
        return {
            "task_id": task_id,
            "worker_id": self.plugin_id,
            "result": result,
            "completed": True
        }
    
    def get_services(self) -> List[str]:
        return ["get_task_count"]
    
    def get_task_count(self) -> int:
        return self.tasks_processed

# Usage
if __name__ == "__main__":
    import time
    
    manager = PluginManager()
    
    # Load multiple workers
    num_workers = 3
    for i in range(num_workers):
        descriptor = PluginDescriptor(
            name=f"worker_{i}",
            version="1.0.0",
            package=__name__,
            class_name="InferenceWorker",
            options={"model_id": "model_v1"}
        )
        manager.load_plugin(descriptor)
        manager.start_plugin(f"worker_{i}")
    
    # Start processing
    manager.start_processing_loop(interval=0.01)
    
    # Submit inference tasks
    num_tasks = 100
    for i in range(num_tasks):
        task = {
            "id": i,
            "data": np.random.randn(128)
        }
        manager.publish("inference_tasks", task)
    
    # Wait for processing
    time.sleep(2.0)
    
    # Collect results
    results_by_worker = {}
    for i in range(num_workers):
        worker_id = f"worker_{i}"
        count = manager.invoke_service(worker_id, "get_task_count")
        results_by_worker[worker_id] = count
    
    print(f"Processed {num_tasks} tasks:")
    for worker_id, count in results_by_worker.items():
        print(f"  {worker_id}: {count} tasks")
    
    # Cleanup
    manager.stop_processing_loop()
    manager.shutdown()
```

## Example 4: Monitoring and Logging

Plugin that monitors other plugins:

```python
from modularnet.plugin import BasePlugin, EnvironmentSubscriptionDescriptor
from typing import Dict, Any, List
from datetime import datetime

class MonitorPlugin(BasePlugin):
    """Monitor plugin that tracks system metrics."""
    
    def __init__(self, plugin_id: str, config: Dict[str, Any] = None):
        super().__init__(plugin_id, config)
        self.metrics = {
            "startup_time": None,
            "cpu_samples": [],
            "memory_samples": [],
            "events": []
        }
    
    def on_start(self):
        self.metrics["startup_time"] = datetime.now()
        self.log_event("started")
    
    def on_stop(self):
        self.log_event("stopped")
    
    def get_env_subscriptions(self) -> List[EnvironmentSubscriptionDescriptor]:
        return [
            EnvironmentSubscriptionDescriptor("system_cpu", "on_cpu_change"),
            EnvironmentSubscriptionDescriptor("system_memory", "on_memory_change"),
        ]
    
    def on_cpu_change(self, update):
        self.metrics["cpu_samples"].append({
            "value": update.value,
            "timestamp": datetime.now()
        })
        self.log_event(f"CPU: {update.value:.1f}%")
    
    def on_memory_change(self, update):
        self.metrics["memory_samples"].append({
            "value": update.value,
            "timestamp": datetime.now()
        })
        self.log_event(f"Memory: {update.value:.1f}%")
    
    def log_event(self, message: str):
        """Log an event."""
        self.metrics["events"].append({
            "timestamp": datetime.now(),
            "message": message
        })
        print(f"[{datetime.now().isoformat()}] {message}")
    
    def get_services(self) -> List[str]:
        return ["get_metrics", "get_uptime"]
    
    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics
    
    def get_uptime(self) -> float:
        if self.metrics["startup_time"]:
            return (datetime.now() - self.metrics["startup_time"]).total_seconds()
        return 0.0

# Usage
if __name__ == "__main__":
    import time
    
    manager = PluginManager()
    
    # Load monitor
    descriptor = PluginDescriptor(
        name="monitor",
        version="1.0.0",
        package=__name__,
        class_name="MonitorPlugin"
    )
    
    manager.load_plugin(descriptor)
    manager.start_plugin("monitor")
    
    # Simulate system metrics being published
    manager.set_env_variable("system_cpu", 25.0)
    manager.set_env_variable("system_memory", 45.0)
    
    time.sleep(0.5)
    
    manager.set_env_variable("system_cpu", 35.0)
    manager.set_env_variable("system_memory", 50.0)
    
    time.sleep(0.5)
    
    manager.set_env_variable("system_cpu", 20.0)
    manager.set_env_variable("system_memory", 48.0)
    
    # Get metrics
    metrics = manager.invoke_service("monitor", "get_metrics")
    uptime = manager.invoke_service("monitor", "get_uptime")
    
    print(f"\nUptime: {uptime:.2f}s")
    print(f"Events logged: {len(metrics['events'])}")
    print(f"CPU samples: {len(metrics['cpu_samples'])}")
    print(f"Memory samples: {len(metrics['memory_samples'])}")
    
    # Cleanup
    manager.shutdown()
```

## Tips and Patterns

### Pattern 1: Plugin Discovery

```python
def load_all_plugins_from_package(manager, package_name):
    """Auto-discover and load all plugins in a package."""
    import importlib
    import inspect
    
    module = importlib.import_module(package_name)
    
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and issubclass(obj, BasePlugin) and obj != BasePlugin:
            # Found a plugin class
            descriptor = PluginDescriptor(
                name=name.lower(),
                version="1.0.0",
                package=package_name,
                class_name=name
            )
            manager.load_plugin(descriptor)
```

### Pattern 2: Configuration Files

```python
import json

def load_config(filepath: str):
    """Load configuration from JSON file."""
    with open(filepath) as f:
        return json.load(f)

# config.json:
# {
#   "plugins": [
#     {
#       "name": "plugin_a",
#       "version": "1.0.0",
#       "package": "my_app.plugins",
#       "class_name": "PluginA",
#       "config": {"param": "value"}
#     }
#   ]
# }

config = load_config("config.json")
for plugin_cfg in config["plugins"]:
    descriptor = PluginDescriptor(**plugin_cfg)
    manager.load_plugin(descriptor, plugin_cfg.get("config"))
```

### Pattern 3: Graceful Shutdown

```python
import signal

def setup_graceful_shutdown(manager):
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(sig, frame):
        print("\nShutting down...")
        manager.shutdown()
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
```
