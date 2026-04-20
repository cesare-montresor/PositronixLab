# ModularNet Plugin System (ROS-like)

A transparent, efficient multiprocessing-based pub/sub plugin system with unified data transport inspired by ROS.

## Features

- **Process Isolation**: Each plugin runs in its own process with true parallelism
- **Unified Pub/Sub**: Topics-based message passing with automatic transport detection
- **Efficient Tensor Communication**: Zero-copy shared memory for tensors (SYNC and LATEST modes)
- **Transparent API**: Simple `subscribe()` and `publish()` methods work with any data type
- **Environment Variables**: Share state across plugins with `env_get()`, `env_set()`, and callbacks
- **Scalable**: Central manager coordinates all plugin communications

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Main Process                          │
│            PluginManager (coordinator)                   │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Shared Multiprocessing State                    │   │
│  │  - Topics: {topic_name: {plugin_name: queue}}   │   │
│  │  - Tensor Channels: zero-copy shared memory     │   │
│  │  - Environment Registry: thread-safe var store  │   │
│  └──────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
  │ Plugin Process │  │ Plugin Process │  │ Plugin Process │
  │   Publisher    │  │  Subscriber    │  │  Subscriber    │
  │                │  │                │  │                │
  │publish(data)───────→receive() ─→callback()            │
  │                │  │                │  │                │
  │ (auto-detect   │  │(reads from     │  │ (callback in  │
  │  transport)    │  │ shared memory) │  │  thread)      │
  └────────────────┘  └────────────────┘  └────────────────┘
```

## Unified Transport System

The plugin system automatically detects the data type being published and selects the optimal transport mechanism:

### Transport Strategies

| Transport | Use Case | Mechanism |
|-----------|----------|-----------|
| **PICKLE** | Small objects, scalars, stateless modules | Standard Python pickling |
| **SHARED_MODULE** | `nn.Module` with parameters | Calls `share_memory_()` on parameters |
| **SHARED_TENSOR** | CPU tensors | Zero-copy shared memory buffer |
| **CUDA_IPC** | CUDA tensors | CUDA Inter-Process Communication |
| **SIGNAL_ONLY** | Large/unserializable objects | Notification only, caller fetches data |

Auto-detection is transparent — just use `publish(data)` and the system handles transport selection.

## Usage Examples

### 1. Basic Pub/Sub Communication

```python
from modularnet.plugin.plugin import Plugin, PluginManager
import time

class Publisher(Plugin):
    def __init__(self):
        super().__init__(name="Publisher")
        self.count = 0
    
    def onUpdate(self):
        # Any data type — transport auto-detected
        self.publish("counter", self.count)
        self.count += 1
        time.sleep(0.1)

class Subscriber(Plugin):
    def __init__(self):
        super().__init__(name="Subscriber")
        self.subscribe("counter", self.on_count)
    
    def on_count(self, value):
        print(f"Received: {value}")
    
    def onUpdate(self):
        time.sleep(0.1)

# Setup and run
manager = PluginManager()
manager.register_plugin(Publisher())
manager.register_plugin(Subscriber())
manager.start_all_plugins()
time.sleep(5)
manager.stop_all_plugins()
```

### 2. Efficient Tensor Communication

For high-throughput tensor data (e.g., activations, gradients), use tensor-specific methods:

```python
class ModelProducer(Plugin):
    def onUpdate(self):
        # Publish tensor with guaranteed zero-copy delivery
        activation = model(input_batch)  # shape: (batch, 512)
        self.publish_tensor("activations", activation)

class Analyzer(Plugin):
    def __init__(self):
        super().__init__(name="Analyzer")
        # Subscribe to tensor with specific mode
        self.subscribe_tensor(
            "activations",
            self.analyze,
            mode=TensorDeliveryMode.LATEST
        )
    
    def analyze(self, tensor):
        # Zero-copy view into shared memory
        mean = tensor.mean()
        print(f"Mean activation: {mean}")
```

### 3. Environment Variables

Share configuration and state across plugins with automatic synchronization:

```python
class ConfigurationPlugin(Plugin):
    def onStart(self):
        # Declare interest in environment variables
        self.env_declare("learning_rate", default=1e-3, watch=True)
        self.env_declare("model", watch=True, signal_only=False)
        
        # Set up callbacks for changes
        self.env_watch("learning_rate", self.on_lr_changed)
    
    def on_lr_changed(self, old, new):
        print(f"Learning rate changed: {old} -> {new}")
    
    def onUpdate(self):
        # Read current value
        lr = self.env_get("learning_rate")
        # Use lr in training...

class MonitorPlugin(Plugin):
    def onUpdate(self):
        # Change from any process
        self.env_set("learning_rate", 3e-4)

# Manager setup
manager = PluginManager()
manager.env_register("learning_rate", default=1e-3)
manager.env_register("model", signal_only=True)  # notify only, no pickle
manager.register_plugin(ConfigurationPlugin())
manager.register_plugin(MonitorPlugin())
```

### 4. Forward Hook Integration

Attach activation hooks to neural network layers for efficient monitoring:

```python
from modularnet.plugin.plugin_pubsub import ActivationTap, TensorDeliveryMode

manager = PluginManager()
# ... register plugins ...

# Tap into layer output
tap = ActivationTap("layer_output", manager, mode=TensorDeliveryMode.LATEST)
model.layer.register_forward_hook(tap.hook)

# Any plugin can now subscribe to "layer_output"
```

## Key Classes

### PluginBase ([plugin_base.py](plugin_base.py))
Base class for all plugins. Manages plugin lifecycle with multiprocessing support.

**Methods:**
- `start()`: Start the plugin in a new process
- `stop()`: Stop the plugin
- `onStart()`: Override to initialize when plugin starts
- `onUpdate()`: Override to implement plugin logic (called repeatedly)
- `onStop()`: Override to cleanup when plugin stops

### PluginPubSub ([plugin_pubsub.py](plugin_pubsub.py))
Extends PluginBase with pub/sub capabilities and automatic transport detection.

**Methods:**
- `subscribe(topic_name, callback)`: Subscribe to a topic with callback
- `subscribe_tensor(topic_name, callback, ...)`: Subscribe to tensor topic with optional shape/dtype/mode
- `publish(topic_name, data)`: Publish any data (transport auto-detected)
- `publish_tensor(topic_name, tensor)`: Publish tensor (guaranteed zero-copy path)
- `get_tensor_channel(topic_name)`: Get TensorChannel for manual control

### PluginEnv ([plugin_env.py](plugin_env.py))
Mixin adding environment variable get/set/watch capabilities to plugins.

**Methods:**
- `env_declare(key, default, watch, signal_only, on_change)`: Declare an env variable
- `env_watch(key, callback)`: Attach callback to env changes
- `env_set(key, value)`: Write env variable
- `env_get(key)`: Read env variable

### PluginManagerBase ([plugin_base.py](plugin_base.py))
Base manager for registering and controlling plugins.

**Methods:**
- `register_plugin(plugin, name)`: Register a plugin
- `start_plugin(name)`: Start specific plugin
- `stop_plugin(name)`: Stop specific plugin
- `start_all_plugins()`: Start all registered plugins
- `stop_all_plugins()`: Stop all registered plugins
- `get_plugin(name)`: Get plugin by name
- `list_plugins()`: List all plugin names

### PluginManagerPubSub ([plugin_pubsub.py](plugin_pubsub.py))
Manager with unified pub/sub topic management and auto-sensing transport.

**Methods:**
- `publish_to_topic(topic_name, data)`: Publish to topic
- `publish_tensor_to_topic(topic_name, tensor, ...)`: Publish tensor
- `get_topic_queue(topic_name, plugin_name)`: Get message queue
- `get_tensor_channel(topic_name, plugin_name)`: Get tensor channel

### PluginManagerEnv ([plugin_env.py](plugin_env.py))
Manager mixin with environment variable registry.

**Methods:**
- `env_register(key, default, signal_only)`: Register env variable
- `env_set(key, value)`: Set from main process
- `env_get(key)`: Get current value

### TensorChannel ([plugin_transport.py](../plugin_transport.py))
Shared-memory slot for efficient tensor communication between processes.

**Features:**
- Lazy or eager allocation mode
- SYNC mode: writer blocks until reader consumes
- LATEST mode: writer never blocks, reader sees newest value
- Zero-copy after first write

**Methods:**
- `write(tensor)`: Write tensor to shared buffer
- `read(timeout)`: Read from buffer (blocking or non-blocking based on mode)
- `consumed()`: Signal reader is done (SYNC mode only)
- `get_version()`: Get monotonic write counter (LATEST mode)

### ActivationTap ([plugin_pubsub.py](../plugin_pubsub.py))
Forward hook for efficient neural network layer monitoring.

```python
tap = ActivationTap("layer_name", manager, dtype=torch.float32, 
                    mode=TensorDeliveryMode.LATEST)
model.layer.register_forward_hook(tap.hook)
```

## Transport Details

### How Transport Auto-Sensing Works

When you call `publish(data)`:

1. **Classification**: The system inspects the data type
2. **Preparation**: Required transformations (e.g., `share_memory_()`) are applied
3. **Distribution**: Data is sent to all subscribers with optimal delivery
4. **Reception**: Subscribers receive actual objects (not just handles)

```python
# Automatic transport selection examples:
publish(42)                    # → PICKLE
publish(model)                 # → SHARED_MODULE (has parameters)
publish(loss_tensor)           # → SHARED_TENSOR (CPU) or CUDA_IPC (CUDA)
publish(BytesIO(large_data))   # → SIGNAL_ONLY (if huge)
```

### TensorDeliveryMode

- **SYNC**: Every frame guaranteed delivery. Writer blocks until reader consumes.
  - Use for: critical data, low-frequency streams, when frame skipping is unacceptable
  
- **LATEST**: Writer never blocks. Reader always sees newest value.
  - Use for: high-frequency streams (video, activations), when latency matters more than completeness

```python
# SYNC mode (default)
self.subscribe_tensor("critical", callback, mode=TensorDeliveryMode.SYNC)

# LATEST mode (skip frames OK)
self.subscribe_tensor("fast_stream", callback, mode=TensorDeliveryMode.LATEST)
```

## Best Practices

1. **Lightweight Callbacks**: Callbacks run in a receiver thread, keep them fast
2. **Error Handling**: Exceptions in `onUpdate()` are caught. Set `fatal_errors=True` to propagate
3. **Resource Cleanup**: Override `onStop()` to clean up resources
4. **Message Size**: Keep messages serializable (avoid large custom objects)
5. **Topic Naming**: Use hierarchical names like `component/sensor/temperature`
6. **Tensor Efficiency**: Use `publish_tensor()` and `subscribe_tensor()` for large tensors
7. **Environment Variables**: Use for slow-changing configuration, not high-frequency data

## Performance Considerations

- Each plugin runs in its own process: true parallelism on multi-core systems
- Message queues are non-blocking with automatic dropout on overflow
- Tensor channels: zero-copy after first buffer allocation
- Receiver thread polls every 10ms for maximum latency
- Uses torch.multiprocessing for PyTorch compatibility

## Advanced Configuration

### Custom Tensor Topic with Shape Hint

```python
# Pre-allocate buffer if you know output shape
self.subscribe_tensor(
    "layer_output",
    self.callback,
    shape=(batch_size, 512),    # pre-allocate
    dtype=torch.float32,
    mode=TensorDeliveryMode.SYNC
)
```

### Signal-Only Mode

```python
# For large objects that can't be pickled
manager.env_register("dataset", signal_only=True)

# Subscribers get notification but retrieve themselves
def on_dataset_available(old, new):
    # new is None in signal_only mode — call env_get() yourself
    dataset = plugin.env_get("dataset")
```

### Error Handling

```python
class RobustPlugin(Plugin):
    def __init__(self):
        # fatal_errors=False: errors are logged, plugin continues
        # fatal_errors=True: errors cause plugin to stop
        super().__init__(name="RobustPlugin", fatal_errors=False)
    
    def onUpdate(self):
        try:
            self.process_data()
        except Exception as e:
            print(f"Recoverable error: {e}")  # logged automatically too
```

## Troubleshooting

### "Queue full" warnings
- Subscribers are slow; drop is expected. Use `LATEST` mode if frame skipping OK
- Or increase timeout when subscribing

### Tensor shape mismatch
- Once a lazy TensorChannel allocates, shape is fixed
- Publisher must always send same shape or specify eager mode with shape hint

### Import errors in child processes
- Ensure all plugin imports work independently
- Child processes can't access parent's memory (except shared memory)
- Test with: `plugin.start()` and let it run separately

## Examples

See [plugin_test_v0.py](../tests/plugin_test_v0.py) for a complete working example with pub/sub testing.


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
