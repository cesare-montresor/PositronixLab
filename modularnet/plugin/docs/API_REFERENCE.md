# ModularNet Plugin System - API Reference

Complete API documentation for all plugin system components.

## Quick Reference

### Class Hierarchy

```
PluginBase
├── PluginPubSub
│   └── PluginEnv
└── Plugin (= PluginPubSub + PluginEnv)

PluginManagerBase
├── PluginManagerPubSub
│   └── PluginManagerEnv
└── PluginManager (= PluginManagerPubSub + PluginManagerEnv)

Transport & Channels
├── TensorChannel
├── ActivationTap
├── _Transport (constants)
├── TensorDeliveryMode (constants)
└── _classify() (function)
```

## Core Classes

### `PluginBase`

Base class for all plugins.

```python
class PluginBase:
    def __init__(
        self,
        options: dict = {},
        name: str = None,
        fatal_errors: bool = False,
        logging = None
    )
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `options` | dict | {} | Configuration passed to plugin |
| `name` | str | class name | Plugin identifier |
| `fatal_errors` | bool | False | Terminate on exception in onUpdate |
| `logging` | Logger | auto | Logger instance to use |

#### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `onStart()` | → None | Override: called once on plugin start |
| `onUpdate()` | → None | Override: called repeatedly in main loop |
| `onStop()` | → None | Override: called once on plugin stop |
| `start(wait=False)` | → None | Launch plugin in new process |
| `stop(wait=True)` | → None | Stop plugin (optionally wait for exit) |
| `manifest()` | → dict | Return {name, version} metadata |
| `set_manager(manager)` | → None | Register manager (called by PluginManager) |

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | str | Plugin identifier |
| `version` | str | Plugin version |
| `options` | dict | Configuration |
| `logger` | Logger | Logger instance |
| `manager` | PluginManagerBase | Manager reference |
| `task_main` | Task | Multiprocessing task handle |
| `start_event` | Event | Start synchronization event |
| `stop_event` | Event | Stop flag |

### `PluginManagerBase`

Base manager for plugin registration and control.

```python
class PluginManagerBase:
    def __init__(self)
```

#### Methods

| Method | Signature | Return | Description |
|--------|-----------|--------|-------------|
| `register_plugin(plugin, name=None)` | (PluginBase, str?) → None | Registers plugin |
| `unregister_plugin(plugin, name=None)` | (PluginBase, str?) → None | Unregisters plugin |
| `start_plugin(name)` | (str) → None | Start specific plugin |
| `stop_plugin(name)` | (str) → None | Stop specific plugin |
| `start_all_plugins()` | () → None | Start all registered plugins |
| `stop_all_plugins()` | () → None | Stop all plugins |
| `get_plugin(name)` | (str) → PluginBase? | Get plugin by name (or None) |
| `list_plugins()` | () → List[str] | Get all plugin names |

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `plugins` | Dict[str, PluginBase] | Registered plugins by name |

### `PluginPubSub`

Extends PluginBase with pub/sub capabilities.

```python
class PluginPubSub(PluginBase):
    def __init__(
        self,
        options: dict = {},
        name: str = None,
        fatal_errors: bool = False
    )
```

#### Pub/Sub Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `subscribe(topic, callback)` | (str, Callable) → None | Subscribe to topic |
| `subscribe_tensor(topic, callback, shape=None, dtype=float32, mode=LATEST)` | (str, Callable, Tuple?, dtype, str) → None | Subscribe to tensor topic with optional pre-allocation |
| `publish(topic, data)` | (str, Any) → None | Publish any data type |
| `publish_tensor(topic, tensor)` | (str, Tensor) → None | Publish tensor (zero-copy) |
| `get_tensor_channel(topic)` | (str) → TensorChannel? | Get tensor channel for manual control |

#### Callback Signatures

```python
# Regular subscription
def on_message(data: Any) → None:
    pass

# Tensor subscription
def on_tensor(tensor: torch.Tensor) → None:
    pass

# In SYNC mode, must call consumed():
def on_tensor_sync(tensor: torch.Tensor) → None:
    # Process tensor
    channel = self.get_tensor_channel("topic")
    if channel:
        channel.consumed()  # Unblock writer
```

### `PluginManagerPubSub`

Manager with pub/sub support.

```python
class PluginManagerPubSub(PluginManagerBase):
    def __init__(self)
```

#### Methods (in addition to base)

| Method | Signature | Return | Description |
|--------|-----------|--------|-------------|
| `publish_to_topic(topic, data)` | (str, Any) → None | Publish from main process |
| `publish_tensor_to_topic(topic, tensor, lazy=False, dtype=float32, mode=LATEST)` | (str, Tensor, bool, dtype, str) → None | Publish tensor from main |
| `get_topic(topic)` | (str) → PluginTopic | Get or create topic |
| `get_topic_queue(topic, plugin_name)` | (str, str) → Queue | Get reception queue for plugin |
| `get_tensor_channel(topic, plugin_name)` | (str, str) → TensorChannel? | Get tensor channel |

### `PluginEnv`

Mixin adding environment variables to plugins.

```python
class PluginEnv(PluginBase):
    def __init__(self, *args, **kwargs)
```

#### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `env_declare(key, default=None, watch=False, signal_only=False, on_change=None)` | (str, Any?, bool, bool, Callable?) → None | Declare env variable interest |
| `env_watch(key, callback)` | (str, Callable) → None | Attach callback to variable |
| `env_set(key, value)` | (str, Any) → None | Set env variable value |
| `env_get(key)` | (str) → Any | Get env variable value |

#### Callback Signatures

```python
# Watch callback
def on_change(old_value: Any, new_value: Any) → None:
    pass

# In signal_only mode:
# old_value and new_value are None; call env_get() to fetch
```

### `PluginManagerEnv`

Manager with environment variable registry.

```python
class PluginManagerEnv(PluginManagerBase):
    def __init__(self)
```

#### Methods (in addition to base)

| Method | Signature | Return | Description |
|--------|-----------|--------|-------------|
| `env_register(key, default=None, signal_only=False)` | (str, Any?, bool) → self | Register env variable (returns self for chaining) |
| `env_set(key, value)` | (str, Any) → None | Set value from main process |
| `env_get(key)` | (str) → Any | Get current value |

### `Plugin`

Unified plugin with both pub/sub and env variable support.

```python
class Plugin(PluginPubSub, PluginEnv):
    # Inherits all methods from both mixins
    pass
```

### `PluginManager`

Unified manager with both pub/sub and env support.

```python
class PluginManager(PluginManagerPubSub, PluginManagerEnv):
    # Inherits all methods from both mixins
    pass
```

## Transport & Channels

### `TensorChannel`

Low-level shared memory tensor channel.

```python
class TensorChannel:
    def __init__(
        self,
        shape: Optional[Tuple] = None,
        dtype: torch.dtype = torch.float32,
        mode: str = TensorDeliveryMode.LATEST
    )
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shape` | Tuple? | None | Buffer shape (None = lazy allocation) |
| `dtype` | torch.dtype | float32 | Tensor data type |
| `mode` | str | LATEST | SYNC or LATEST delivery |

#### Methods

| Method | Signature | Return | Description |
|--------|-----------|--------|-------------|
| `write(tensor)` | (Tensor) → bool | Writes tensor to buffer, blocks in SYNC mode |
| `read(timeout=None)` | (float?) → Tensor? | Reads from buffer, blocks in SYNC mode |
| `consumed()` | () → None | Signal read done (SYNC mode only) |
| `get_version()` | () → int | Get write counter (LATEST mode only) |

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `shape` | Tuple? | Allocated buffer shape |
| `dtype` | torch.dtype | Tensor data type |
| `mode` | str | Delivery mode |
| `tensor` | Tensor? | Shared memory tensor |

### `ActivationTap`

Forward hook for capturing layer activations.

```python
class ActivationTap:
    def __init__(
        self,
        topic_name: str,
        manager: PluginManagerPubSub,
        dtype: torch.dtype = torch.float32,
        mode: str = TensorDeliveryMode.LATEST
    )
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `topic_name` | str | - | Topic to publish activations to |
| `manager` | PluginManagerPubSub | - | Plugin manager instance |
| `dtype` | torch.dtype | float32 | Tensor data type |
| `mode` | str | LATEST | SYNC or LATEST delivery |

#### Methods

| Method | Signature | Return | Description |
|--------|-----------|--------|-------------|
| `hook(module, input, output)` | (Module, tuple, Tensor/tuple) → None | Forward hook (register with `register_forward_hook()`) |

### Constants

#### `TensorDeliveryMode`

```python
class TensorDeliveryMode:
    SYNC = "sync"      # Writer blocks until reader consumes
    LATEST = "latest"  # Writer never blocks, reader gets newest
```

#### `_Transport`

```python
class _Transport:
    PICKLE = "pickle"              # Default serialization
    SHARED_MODULE = "shared_module"  # nn.Module with parameters
    SHARED_TENSOR = "shared_tensor"  # CPU tensor (zero-copy)
    CUDA_IPC = "cuda_ipc"          # CUDA tensor
    SIGNAL_ONLY = "signal_only"    # Notification only
```

### Functions

#### `_classify(value) → str`

Auto-detects transport strategy for a value.

```python
transport = _classify(42)  # "pickle"
transport = _classify(torch.randn(5))  # "shared_tensor" or "cuda_ipc"
transport = _classify(nn.Linear(10, 5))  # "shared_module"
```

## Common Patterns

### Pattern: Subscribe in __init__, Use in onUpdate

```python
class ConsumerPlugin(Plugin):
    def __init__(self):
        super().__init__(name="Consumer")
        self.subscribe("my_topic", self.on_message)
        self.last_message = None
    
    def on_message(self, data):
        self.last_message = data  # Callback in receiver thread
    
    def onUpdate(self):
        if self.last_message:
            print(f"Got: {self.last_message}")
        time.sleep(0.1)
```

### Pattern: Environment Variables with Callbacks

```python
class AdaptivePlugin(Plugin):
    def __init__(self):
        super().__init__(name="Adaptive")
        self.env_declare("learning_rate", default=1e-3, watch=True,
                        on_change=self.on_lr_changed)
        self.lr = 1e-3
    
    def on_lr_changed(self, old_lr, new_lr):
        self.lr = new_lr
        print(f"LR updated: {old_lr} → {new_lr}")
    
    def onUpdate(self):
        # Use self.lr which is updated via callback
        pass
```

### Pattern: Tensor Topic with Shape Inference

```python
class DataProcessor(Plugin):
    def __init__(self):
        super().__init__(name="Processor")
        # shape=None means lazy allocation
        self.subscribe_tensor("features", self.process,
                            shape=None,  # Infer from first tensor
                            mode=TensorDeliveryMode.LATEST)
    
    def process(self, tensor):
        # Callback receives zero-copy view
        print(f"Tensor shape: {tensor.shape}")
```

### Pattern: Synchronous Processing

```python
class Logger(Plugin):
    def __init__(self):
        super().__init__(name="Logger")
        self.subscribe_tensor("frames", self.log_frame,
                            mode=TensorDeliveryMode.SYNC)  # Wait for consumer
    
    def log_frame(self, frame):
        save_to_disk(frame)
        # Must call consumed() to unblock producer
        channel = self.get_tensor_channel("frames")
        if channel:
            channel.consumed()
    
    def onUpdate(self):
        time.sleep(0.1)
```

### Pattern: Manager Publishing from Main Process

```python
def main():
    manager = PluginManager()
    
    # Setup
    manager.env_register("config", default={})
    plugin = MyPlugin()
    manager.register_plugin(plugin)
    manager.start_all_plugins()
    
    # Publish from main process
    time.sleep(1)
    manager.publish_to_topic("control", {"shutdown": True})
    
    time.sleep(1)
    manager.stop_all_plugins()
```

## Error Handling

### Timeout Handling

```python
channel = self.get_tensor_channel("slow_producer")
if channel:
    tensor = channel.read(timeout=1.0)  # Wait up to 1 second
    if tensor is None:
        print("Timeout waiting for tensor")
```

### Callback Error Handling

```python
class SafeConsumer(Plugin):
    def __init__(self):
        super().__init__(name="Safe")
        self.subscribe("data", self.safe_callback)
    
    def safe_callback(self, data):
        try:
            process_data(data)
        except Exception as e:
            print(f"Error: {e}")  # Don't crash plugin
```

### Plugin Error Handling

```python
class RobustPlugin(Plugin):
    def __init__(self):
        super().__init__(name="Robust", fatal_errors=False)
    
    def onUpdate(self):
        try:
            risky_operation()
        except Exception as e:
            print(f"Error: {e}")
            # Plugin continues running
```

## Performance Tips

1. **Use LATEST mode** for high-throughput data you don't need to save
2. **Use SYNC mode** only for logging/recording critical data
3. **Use publish_tensor()** instead of publish() for tensors
4. **Pre-allocate** tensor channels if shape is known
5. **Batch env_set()** calls - don't update per iteration
6. **Use signal_only** for non-pickleable data
7. **Keep callbacks fast** - use threads for slow operations

## Debugging

```python
# List all plugins
print(manager.list_plugins())

# Get plugin reference
plugin = manager.get_plugin("MyPlugin")

# Check if running
import psutil
if plugin.task_main:
    ps = psutil.Process(plugin.task_main.pid)
    print(ps.is_running())

# Enable debug logging
from modularnet.logger import Logger
Logger.set_level("DEBUG")
```

