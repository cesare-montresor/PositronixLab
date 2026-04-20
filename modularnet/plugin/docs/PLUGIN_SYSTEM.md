# ModularNet Plugin System (ROS-like)

A transparent, efficient multiprocessing-based pub/sub plugin system inspired by ROS.

## Features

- **Process Isolation**: Each plugin runs in its own process
- **Pub/Sub Communication**: Topics-based message passing using multiprocessing queues
- **Transparent API**: Simple `subscribe()` and `publish()` methods
- **Efficient**: Non-blocking queue operations with automatic message distribution
- **Scalable**: Central manager coordinates all plugin communications

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Main Process                        │
│              PluginManager (coordinator)             │
│   ┌─────────────────────────────────────────────┐   │
│   │  Shared Multiprocessing State               │   │
│   │  - Topics: {topic_name: {plugin: queue}}    │   │
│   └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
      │                    │                    │
      ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│Plugin Process│    │Plugin Process│    │Plugin Process│
│  Publisher   │    │ Subscriber   │    │Subscriber    │
│              │    │              │    │              │
│ publish()───────→ receive() ─→callback()            │
│              │    │              │    │              │
└──────────────┘    └──────────────┘    └──────────────┘
```

## Usage Example

### 1. Create Publisher Plugin

```python
from modularnet.plugin.plugin import Plugin
import time

class SensorPublisher(Plugin):
    def __init__(self):
        super().__init__(name="SensorPublisher")
        self.value = 0
    
    def onUpdate(self):
        # Publish sensor data every loop iteration
        self.publish("sensor/data", self.value)
        self.value += 1
        time.sleep(0.1)
```

### 2. Create Subscriber Plugin

```python
class DataProcessor(Plugin):
    def __init__(self):
        super().__init__(name="DataProcessor")
        # Subscribe to topic with callback
        self.subscribe("sensor/data", self.process_data)
        self.count = 0
    
    def process_data(self, data):
        # Called whenever a message is published
        print(f"Processing: {data}")
        self.count += 1
    
    def onUpdate(self):
        time.sleep(0.1)
```

### 3. Use the Manager

```python
from modularnet.plugin.plugin import PluginManager
import time

# Create manager
manager = PluginManager()

# Register plugins
publisher = SensorPublisher()
processor = DataProcessor()

manager.register_plugin(publisher)
manager.register_plugin(processor)

# Start all plugins (each runs in its own process)
manager.start_all_plugins()

# Let them run
time.sleep(10)

# Stop all plugins
manager.stop_all_plugins()
```

## Key Classes

### PluginBase (`modularnet.plugin.plugin_base`)
Base class for all plugins. Manages plugin lifecycle with multiprocessing support.

- `start()`: Start the plugin in a new process
- `stop()`: Stop the plugin
- `onStart()`: Override to initialize when plugin starts
- `onUpdate()`: Override to implement plugin logic (called repeatedly)
- `onStop()`: Override to cleanup when plugin stops

### PluginPubSub (`modularnet.plugin.plugin_pubsub`)
Extends PluginBase with pub/sub capabilities using multiprocessing queues for inter-process communication.

- `subscribe(topic_name, callback)`: Subscribe to a topic with a callback function
- `publish(topic_name, data)`: Publish data to a topic (non-blocking)

### PluginManagerBase (`modularnet.plugin.plugin_manager`)
Base manager for registering and controlling plugins.

- `register_plugin(plugin, name=None)`: Register a plugin
- `start_plugin(name)`: Start a specific plugin
- `stop_plugin(name)`: Stop a specific plugin
- `start_all_plugins()`: Start all registered plugins
- `stop_all_plugins()`: Stop all registered plugins
- `get_plugin(name)`: Get a plugin by name
- `list_plugins()`: List all registered plugin names

### PluginManagerPubSub (`modularnet.plugin.plugin_pubsub`)
Extends PluginManagerBase with pub/sub topic management for coordinating plugin communications.

### Backward-Compatible Aliases (`modularnet.plugin.plugin`)
For convenience, the following aliases are available:

- `Plugin` = `PluginPubSub` (preferred: use `PluginPubSub` from `plugin_pubsub` module)
- `PluginManager` = `PluginManagerPubSub` (preferred: use `PluginManagerPubSub` from `plugin_pubsub` module)

## Best Practices

1. **Lightweight Callbacks**: Callbacks are called in a receiver thread, so keep them fast
2. **Error Handling**: Exceptions in `onUpdate()` are caught and printed (set `fatal_errors=True` to propagate)
3. **Resource Cleanup**: Override `onStop()` to clean up resources
4. **Message Size**: Keep messages small and serializable (avoid large objects)
5. **Topic Naming**: Use hierarchical naming like `component/sensor/data`

## Performance Considerations

- Each plugin runs in its own process: true parallelism on multi-core systems
- Queues are non-blocking with automatic dropout on overflow
- Receiver thread polls every 0.01s for maximum latency of ~10ms
- Uses torch.multiprocessing for PyTorch compatibility

## Advanced Topics

### Error Handling

```python
from modularnet.plugin.plugin import Plugin

class RobustPlugin(Plugin):
    def __init__(self):
        super().__init__(name="RobustPlugin", fatal_errors=False)
        # fatal_errors=False (default): errors are logged but don't stop plugin
        # fatal_errors=True: errors cause plugin to stop
    
    def onUpdate(self):
        # If this raises and fatal_errors=False, it's logged and plugin continues
        risky_operation()
```

### Custom Lifecycle

```python
from modularnet.plugin.plugin import Plugin

class CustomPlugin(Plugin):
    def onStart(self):
        super().onStart()
        # Initialize resources here
        self.connection = connect_to_database()
    
    def onStop(self):
        # Cleanup resources here
        self.connection.close()
        super().onStop()
    
    def onUpdate(self):
        # Main plugin logic
        pass
```
