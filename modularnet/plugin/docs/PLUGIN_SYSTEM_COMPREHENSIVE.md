# ModularNet Plugin System - Comprehensive API Guide

A transparent, efficient multiprocessing-based pub/sub plugin system with unified data transport inspired by ROS.

## Table of Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Transport System](#unified-transport-system)
4. [API Reference](#api-reference)
5. [Usage Examples](#usage-examples)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

## Features

- **Process Isolation**: Each plugin runs in its own process with true parallelism
- **Unified Pub/Sub**: Topics-based message passing with automatic transport detection
- **Efficient Tensor Communication**: Zero-copy shared memory for tensors (SYNC and LATEST modes)
- **Transparent API**: Simple `subscribe()` and `publish()` methods work with any data type
- **Environment Variables**: Share state across plugins with `env_get()`, `env_set()`, and callbacks
- **Scalable**: Central manager coordinates all plugin communications
- **Lazy Initialization**: Allocate buffers on-demand, don't pre-allocate sizes
- **Forward Hooks**: Attach `ActivationTap` to capture layer outputs efficiently

## Architecture

### System Overview

```
┌──────────────────────────────────────────────────────────┐
│                    Main Process                          │
│            PluginManager (coordinator)                   │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Shared Multiprocessing State                    │   │
│  │  - Topics: {topic_name: {plugin_name: queue}}   │   │
│  │  - Tensor Channels: zero-copy shared memory     │   │
│  │  - Environment Registry: thread-safe var store  │   │
│  │  - Dispatch Thread: env variable notifications  │   │
│  └──────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
         │                    │                    │
      ┌──▼──┐             ┌──▼──┐             ┌──▼──┐
      │ P1  │             │ P2  │             │ P3  │
      │ (process)         │ (process)        │ (process)
      │                   │                   │
      │ Recv Thread       │ Recv Thread       │ Recv Thread
      │ Env Watcher       │ Env Watcher       │ Env Watcher
      └─────┘             └─────┘             └─────┘
```

### Plugin Lifecycle

```
                       ┌─────────┐
                       │ Created │
                       └────┬────┘
                            │ start()
                            ▼
                       ┌─────────┐
                       │ Running │◄────────────┐
                       │ onUpdate│             │
                       └────┬────┘             │
                            │                 │
                    ┌────────┴────────┐        │
                    │                 │        │
                    ▼                 ▼        │
            ┌──────────────┐  ┌──────────────┐│
            │  Plugin P1   │  │   Receiver   ││
            │ (in process) │  │ (in thread)  ││
            │              │  │              ││
            │ onStart()    │  │ Dispatch     ││
            │ onUpdate()   │  │ Messages     ││
            │ onStop()     │  │              ││
            └──────────────┘  └──────────────┘│
                    │                         │
                    └─────────┬───────────────┘
                              │ stop()
                              ▼
                          ┌─────────┐
                          │ Stopped │
                          └─────────┘
```

## Unified Transport System

The plugin system automatically detects the data type being published and selects the optimal transport mechanism:

### Transport Strategies

| Transport | Use Case | Mechanism | Overhead |
|-----------|----------|-----------|----------|
| **PICKLE** | Small objects, scalars, stateless modules, dicts, lists | Standard Python pickling | Serialization time |
| **SHARED_MODULE** | `nn.Module` with parameters | Calls `share_memory_()` on parameters; metadata pickled | One-time setup + pickle metadata |
| **SHARED_TENSOR** | CPU tensors | Zero-copy shared memory buffer | Minimal (copy into pre-allocated buffer) |
| **CUDA_IPC** | CUDA tensors | CUDA Inter-Process Communication handle | Minimal (CUDA kernel ops) |
| **SIGNAL_ONLY** | Large/unserializable objects (e.g., custom data loaders) | Notification only; caller uses `env_get()` to fetch data | Just a signal in queue |

### Transport Detection Algorithm

```python
def auto_detect_transport(value):
    if isinstance(value, nn.Module):
        if has_parameters(value):
            return SHARED_MODULE  # .share_memory_() on params
        else:
            return PICKLE  # stateless (pure functional)
    
    if isinstance(value, torch.Tensor):
        if value.is_cuda:
            return CUDA_IPC
        else:
            return SHARED_TENSOR
    
    return PICKLE  # everything else
```

### When to Use Signal-Only

Use `signal_only=True` in `env_register()` or `env_declare()` for:
- Large dataset objects that can't be pickled
- File handles or resources tied to a specific process
- Custom data loaders with internal state

Subscribers will receive notification `(None, None)` and must call `env_get()` to fetch.

## API Reference

### PluginBase

Base class for all plugins. Manages plugin lifecycle with multiprocessing support.

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

**Parameters:**
- `options`: Configuration dictionary passed to the plugin
- `name`: Plugin identifier (defaults to class name)
- `fatal_errors`: If True, any exception in `onUpdate()` terminates the plugin
- `logging`: Logger instance (defaults to auto-created logger)

**Lifecycle Methods (override these):**
- `onStart()`: Called once when plugin starts in its process
- `onUpdate()`: Called repeatedly in a loop (1ms sleep between iterations)
- `onStop()`: Called once when plugin is stopping

**Public Methods:**
- `start(wait=False)`: Launch plugin in a new process
- `stop(wait=True)`: Send stop signal and optionally wait for shutdown
- `manifest()`: Return plugin metadata (name, version)
- `set_manager(manager)`: Called by PluginManager; stores reference

**Example:**
```python
class MyPlugin(PluginBase):
    def __init__(self):
        super().__init__(name="MyPlugin", options={"timeout": 5})
        self.counter = 0
    
    def onStart(self):
        print("Plugin started in process")
    
    def onUpdate(self):
        self.counter += 1
        time.sleep(0.1)
    
    def onStop(self):
        print(f"Plugin ran {self.counter} iterations")
```

### PluginManagerBase

Base manager for registering and controlling plugins.

```python
class PluginManagerBase:
    def __init__(self)
```

**Public Methods:**
- `register_plugin(plugin, name=None)`: Register a plugin with the manager
- `unregister_plugin(plugin, name=None)`: Remove a plugin from the manager
- `start_plugin(name)`: Start a specific plugin by name
- `stop_plugin(name)`: Stop a specific plugin by name
- `start_all_plugins()`: Start all registered plugins
- `stop_all_plugins()`: Stop all registered plugins
- `get_plugin(name)`: Retrieve a plugin by name (returns PluginBase or None)
- `list_plugins()`: Get list of all registered plugin names

**Example:**
```python
manager = PluginManagerBase()
manager.register_plugin(my_plugin, name="custom_name")
manager.start_all_plugins()
time.sleep(5)
manager.stop_all_plugins()
```

### PluginPubSub

Extends PluginBase with pub/sub capabilities and automatic transport detection.

```python
class PluginPubSub(PluginBase):
    def __init__(
        self,
        options: dict = {},
        name: str = None,
        fatal_errors: bool = False
    )
```

**Public Methods:**

**Subscriptions:**
- `subscribe(topic_name, callback)`: Subscribe to any data type
  - Callback signature: `callback(data)` — data can be any type
  - Transport auto-detected on publisher side

- `subscribe_tensor(topic_name, callback, shape=None, dtype=torch.float32, mode=TensorDeliveryMode.LATEST)`: 
  - Subscribe to a tensor topic with optional pre-allocated buffer
  - Callback signature: `callback(tensor)` — zero-copy view into shared memory
  - `shape=None` → lazy allocation on first tensor received
  - `mode=SYNC` → callback must call `consumed()` when done reading
  - `mode=LATEST` → overwrites immediately, no blocking

**Publishing:**
- `publish(topic_name, data)`: Publish any data
  - Transport auto-detected
  - Works from main process or plugin process
  - Returns immediately (non-blocking)

- `publish_tensor(topic_name, tensor)`: Publish tensor on guaranteed zero-copy path
  - More explicit than `publish()` for tensors
  - Bypasses auto-detection (slightly faster)

**Other:**
- `get_tensor_channel(topic_name)`: Get TensorChannel for manual control
  - Returns None if not subscribed
  - Allows manual `read()`, `write()`, `consumed()` calls

**Example:**
```python
class Producer(PluginPubSub):
    def onUpdate(self):
        self.publish("results", {"loss": 0.5, "acc": 0.95})

class Consumer(PluginPubSub):
    def onStart(self):
        self.subscribe("results", self.on_results)
    
    def on_results(self, data):
        print(f"Received: {data}")
```

### PluginManagerPubSub

Manager with unified pub/sub support.

```python
class PluginManagerPubSub(PluginManagerBase):
    def __init__(self)
```

**Public Methods:**
- Inherits all `PluginManagerBase` methods
- `publish_to_topic(topic_name, data)`: Publish from main process
- `publish_tensor_to_topic(topic_name, tensor, lazy=False, ...)`: Publish tensor from main
- `get_topic(topic_name)`: Get or create a PluginTopic
- `get_topic_queue(topic_name, plugin_name)`: Get reception queue for a plugin

**Example:**
```python
manager = PluginManagerPubSub()
manager.register_plugin(producer)
manager.register_plugin(consumer)
manager.start_all_plugins()

# Can publish from main process too
import time; time.sleep(1)
manager.publish_to_topic("control", {"shutdown": True})
```

### PluginEnv

Mixin that adds environment variable get/set/watch capabilities.

```python
class PluginEnv(PluginBase):
    def __init__(self, *args, **kwargs)
```

**Declaration Methods (call in `__init__`):**
- `env_declare(key, default=None, watch=False, signal_only=False, on_change=None)`:
  - Declare interest in an env variable
  - `watch=True` → register `on_change` callback
  - `signal_only=True` → only receive notification, use `env_get()` to fetch
  - `on_change` callback signature: `on_change(old_value, new_value)`

- `env_watch(key, callback)`: Add callback to already-declared variable
  - Callback signature: `callback(old_value, new_value)`

**Runtime Methods:**
- `env_set(key, value)`: Write a value to env registry (works from any process)
  - Transport auto-detected for sharing
  - Non-blocking

- `env_get(key)`: Read current value of env variable
  - Works from any process
  - Returns most recent value (may be stale if `signal_only=True`)

**Example:**
```python
class TrainerPlugin(PluginEnv):
    def __init__(self):
        super().__init__(name="Trainer")
        self.env_declare("model", watch=True)
        self.env_declare("learning_rate", default=1e-3, watch=True, 
                        on_change=self.on_lr_changed)
    
    def on_lr_changed(self, old_lr, new_lr):
        print(f"Learning rate changed: {old_lr} → {new_lr}")
        # Update optimizer
    
    def onUpdate(self):
        model = self.env_get("model")
        # Use model...
```

### PluginManagerEnv

Manager mixin with environment variable registry.

```python
class PluginManagerEnv(PluginManagerBase):
    def __init__(self)
```

**Setup Methods (call before registering plugins):**
- `env_register(key, default=None, signal_only=False)`:
  - Declare an env variable in the registry
  - All plugins can read/write
  - `signal_only=True` → value not shared, only notification sent
  - Returns self for chaining

**Runtime Methods:**
- `env_set(key, value)`: Set value from main process
  - Broadcasts to all watching plugins
  - Transport auto-detected for sharing

- `env_get(key)`: Get current value
  - Works from main process or any plugin

**Example:**
```python
manager = PluginManager()  # has both PubSub and Env

manager.env_register("model", signal_only=False)
manager.env_register("learning_rate", default=1e-3)

manager.register_plugin(trainer)
manager.register_plugin(monitor)

manager.env_set("learning_rate", 5e-4)  # all watchers notified
```

### TensorDeliveryMode

Constants for tensor subscription modes:

```python
class TensorDeliveryMode:
    SYNC = "sync"      # writer blocks until reader consumed every frame
    LATEST = "latest"  # writer never blocks, reader gets newest
```

**SYNC Mode:**
- Writer blocks until reader calls `consumed()`
- Every frame delivered and processed
- Use for: logging, introspection, exact frame capture
- Callback must call `channel.consumed()` when done

**LATEST Mode:**
- Writer never blocks, always overwrites
- Reader may skip frames if slow
- Use for: high-throughput data, visualization, real-time monitoring
- Callback does not need to call `consumed()`

### TensorChannel

Low-level tensor communication channel (advanced usage).

```python
class TensorChannel:
    def __init__(
        self,
        shape: Optional[Tuple] = None,
        dtype: torch.dtype = torch.float32,
        mode: str = TensorDeliveryMode.LATEST
    )
```

**Parameters:**
- `shape=None` → lazy: allocate buffer on first `write()`
- `shape=(D1, D2, ...)` → eager: allocate immediately
- `dtype` → tensor data type (default: float32)
- `mode` → SYNC or LATEST

**Public Methods:**
- `write(tensor)`: Copy tensor into shared buffer
  - Lazy init on first call (if shape was None)
  - SYNC: blocks until `consumed()` called
  - LATEST: always overwrites immediately
  - Raises ValueError if shape/dtype don't match

- `read(timeout=None)`: Read from buffer
  - SYNC: blocks until new data (timeout in seconds)
  - LATEST: returns immediately
  - Returns zero-copy view into shared memory
  - Returns None if not initialized

- `consumed()`: Signal that reader is done (SYNC mode only)
  - Unblocks the writer

- `get_version()`: Monotonic counter (LATEST mode only)
  - Incremented on every write
  - Use to detect if new data arrived

**Example (Advanced):**
```python
# Manual channel creation (usually PluginPubSub handles this)
channel = TensorChannel(shape=(256, 512), dtype=torch.float32, 
                       mode=TensorDeliveryMode.SYNC)

# Writer side
tensor = torch.randn(256, 512)
channel.write(tensor)

# Reader side (in another process)
received = channel.read(timeout=1.0)  # blocks up to 1 second
if received is not None:
    # Process received tensor
    print(received.mean())
    channel.consumed()  # unblock writer
```

### ActivationTap

Forward hook for efficient neural network layer monitoring.

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

**Parameters:**
- `topic_name`: Name of the tensor topic to publish to
- `manager`: PluginManagerPubSub instance
- `dtype`: Tensor data type (default: match layer output)
- `mode`: SYNC or LATEST delivery

**Usage:**
```python
manager = PluginManager()
# ... register plugins ...

# Attach hook to layer
tap = ActivationTap("layer3_activations", manager, 
                   mode=TensorDeliveryMode.LATEST)
handle = model.layer3.register_forward_hook(tap.hook)

# Any plugin can now subscribe
class Analyzer(PluginPubSub):
    def onStart(self):
        self.subscribe_tensor("layer3_activations", self.analyze)
    
    def analyze(self, activation):
        print(f"Layer output shape: {activation.shape}, mean: {activation.mean()}")

# Later, detach hook
handle.remove()
```

## Usage Examples

### Example 1: Basic Pub/Sub

```python
from modularnet.plugin.plugin import Plugin, PluginManager
import time

class Publisher(Plugin):
    def __init__(self):
        super().__init__(name="Publisher")
        self.counter = 0
    
    def onUpdate(self):
        self.publish("counter", self.counter)
        self.counter += 1
        time.sleep(0.5)

class Subscriber(Plugin):
    def __init__(self):
        super().__init__(name="Subscriber")
        self.subscribe("counter", self.on_counter)
    
    def on_counter(self, value):
        print(f"Received: {value}")

# Run
manager = PluginManager()
manager.register_plugin(Publisher())
manager.register_plugin(Subscriber())
manager.start_all_plugins()
time.sleep(3)
manager.stop_all_plugins()
```

### Example 2: Tensor Communication

```python
import torch

class TensorProducer(Plugin):
    def __init__(self):
        super().__init__(name="Producer")
        self.step = 0
    
    def onUpdate(self):
        # Publish CPU tensor → auto uses SHARED_TENSOR
        tensor = torch.randn(256, 512)
        self.publish("features", tensor)
        self.step += 1
        if self.step >= 100:
            self.stop_event.set()

class TensorAnalyzer(Plugin):
    def __init__(self):
        super().__init__(name="Analyzer")
        # Subscribe with lazy buffer allocation
        self.subscribe_tensor("features", self.on_features,
                            mode=TensorDeliveryMode.LATEST)
    
    def on_features(self, tensor):
        print(f"Shape: {tensor.shape}, dtype: {tensor.dtype}, "
              f"mean: {tensor.mean():.4f}")

# Run
manager = PluginManager()
manager.register_plugin(TensorProducer())
manager.register_plugin(TensorAnalyzer())
manager.start_all_plugins()
manager.stop_all_plugins()  # waits for producer to finish
```

### Example 3: Environment Variables

```python
class ConfigurableTrainer(PluginEnv):
    def __init__(self):
        super().__init__(name="Trainer")
        self.env_declare("learning_rate", default=1e-3, watch=True,
                        on_change=self.on_lr_changed)
        self.env_declare("model", watch=True, signal_only=False)
    
    def on_lr_changed(self, old_lr, new_lr):
        print(f"Learning rate: {old_lr} → {new_lr}")
    
    def onUpdate(self):
        lr = self.env_get("learning_rate")
        model = self.env_get("model")
        if model is not None:
            # Train with current LR and model
            pass
        time.sleep(0.1)

class Monitor(PluginEnv):
    def __init__(self):
        super().__init__(name="Monitor")
        self.env_declare("learning_rate", watch=True)
    
    def onStart(self):
        print("Monitor started")
    
    def onUpdate(self):
        lr = self.env_get("learning_rate")
        # Report metrics...
        time.sleep(1)

# Run with env control
manager = PluginManager()
manager.env_register("learning_rate", default=1e-3)
manager.env_register("model", signal_only=True)

trainer = ConfigurableTrainer()
monitor = Monitor()

manager.register_plugin(trainer)
manager.register_plugin(monitor)
manager.start_all_plugins()

time.sleep(2)
manager.env_set("learning_rate", 5e-4)  # notify both plugins

time.sleep(2)
manager.stop_all_plugins()
```

### Example 4: Forward Hooks

```python
import torch
import torch.nn as nn

class Visualizer(PluginPubSub):
    def __init__(self):
        super().__init__(name="Visualizer")
        self.subscribe_tensor("conv1_act", self.visualize,
                            mode=TensorDeliveryMode.LATEST)
    
    def visualize(self, activation):
        # activation is a zero-copy view into shared memory
        print(f"Conv1 output: shape={activation.shape}, "
              f"min={activation.min():.4f}, max={activation.max():.4f}")

# Setup model and hooks
model = nn.Sequential(
    nn.Conv2d(3, 32, 3),
    nn.ReLU(),
    nn.Conv2d(32, 64, 3),
)

manager = PluginManager()
manager.register_plugin(Visualizer())
manager.start_all_plugins()

# Attach hook
tap = ActivationTap("conv1_act", manager, mode=TensorDeliveryMode.LATEST)
model[0].register_forward_hook(tap.hook)

# Run some data through model
with torch.no_grad():
    x = torch.randn(1, 3, 32, 32)
    for i in range(10):
        output = model(x)
        time.sleep(0.1)

manager.stop_all_plugins()
```

### Example 5: Multiple Inheritance

```python
class FullFeaturedPlugin(PluginPubSub, PluginEnv):
    """Plugin with both pub/sub and env variable support."""
    
    def __init__(self):
        super().__init__(name="FullFeatured")
        
        # Pub/Sub setup
        self.subscribe("commands", self.on_command)
        
        # Env variables
        self.env_declare("config", default={}, watch=True,
                        on_change=self.on_config_changed)
    
    def on_command(self, cmd):
        if cmd["type"] == "update_config":
            self.env_set("config", cmd["data"])
    
    def on_config_changed(self, old_cfg, new_cfg):
        print(f"Config updated: {new_cfg}")
    
    def onUpdate(self):
        # Can use both pub/sub and env
        config = self.env_get("config")
        self.publish("metrics", {"active": True})
        time.sleep(0.1)

# Usage
manager = PluginManager()
manager.env_register("config", default={})
manager.register_plugin(FullFeaturedPlugin())
manager.start_all_plugins()

time.sleep(1)
# Send command to update config
manager.publish_to_topic("commands", {"type": "update_config", 
                                      "data": {"new_param": 42}})
time.sleep(1)
manager.stop_all_plugins()
```

## Best Practices

### 1. Plugin Naming and Organization

```python
# ✓ Good: descriptive names
class ModelTrainerPlugin(PluginPubSub, PluginEnv):
    pass

class MetricsCollectorPlugin(PluginPubSub):
    pass

# ✗ Avoid: vague names
class Plugin1(PluginPubSub):
    pass
```

### 2. Error Handling

```python
# ✓ Good: handle exceptions in callbacks
class SafeSubscriber(PluginPubSub):
    def onStart(self):
        self.subscribe("data", self.safe_callback)
    
    def safe_callback(self, data):
        try:
            process_data(data)
        except ValueError as e:
            print(f"Invalid data: {e}")
            # Don't crash the whole plugin

# ✗ Avoid: letting exceptions propagate unchecked
def unsafe_callback(data):
    risky_operation(data)  # may crash the plugin
```

### 3. Tensor Delivery Modes

```python
# ✓ Use SYNC for: logging, recording, exact capture
class Logger(PluginPubSub):
    def onStart(self):
        self.subscribe_tensor("frames", self.log_frame,
                            mode=TensorDeliveryMode.SYNC)
    
    def log_frame(self, frame):
        save_to_disk(frame)
        channel = self.get_tensor_channel("frames")
        if channel:
            channel.consumed()  # Required for SYNC!

# ✓ Use LATEST for: visualization, real-time monitoring
class Visualizer(PluginPubSub):
    def onStart(self):
        self.subscribe_tensor("live_output", self.show,
                            mode=TensorDeliveryMode.LATEST)
    
    def show(self, tensor):
        visualize(tensor)  # skip old frames if slow
        # Don't block writer; may drop frames

# ✗ Avoid: SYNC mode with slow callbacks (blocks producer)
class SlowLogger(PluginPubSub):
    def onStart(self):
        self.subscribe_tensor("data", self.slow_operation,
                            mode=TensorDeliveryMode.SYNC)
    
    def slow_operation(self, tensor):
        time.sleep(5)  # BLOCKS the producer!
        save_tensor(tensor)
```

### 4. Environment Variable Lifecycle

```python
# ✓ Good: declare before registration
manager = PluginManager()
manager.env_register("model", signal_only=True)

plugin = MyPlugin()
manager.register_plugin(plugin)  # plugin._env_queues wired here
manager.start_all_plugins()

# ✓ Good: late binding of env values
manager.env_set("model", model)  # set after plugins started

# ✗ Avoid: declaring after registration
manager.register_plugin(plugin)
manager.env_register("model")  # too late! plugin won't see updates
```

### 5. Resource Cleanup

```python
# ✓ Good: clean up in onStop()
class FileReaderPlugin(PluginPubSub):
    def onStart(self):
        self.file = open("data.txt", "r")
    
    def onStop(self):
        if self.file:
            self.file.close()
```

### 6. Multi-Process Synchronization

```python
# ✓ Good: use env variables for synchronization
class Initializer(PluginEnv):
    def onStart(self):
        result = expensive_initialization()
        self.env_set("initialized", True)

class Dependent(PluginEnv):
    def onStart(self):
        self.env_declare("initialized", watch=True,
                        on_change=self.on_init_done)
        self.initialized = False
    
    def on_init_done(self, old, new):
        if new:
            self.initialized = True
    
    def onUpdate(self):
        if not self.initialized:
            return  # wait for initializer
        do_dependent_work()
```

## Troubleshooting

### Problem: "Process hangs or deadlocks"

**Possible Causes:**
1. SYNC mode callback forgot to call `consumed()`
2. Circular dependencies in env watchers
3. Deadlock in `env_set()` during callback

**Solutions:**
```python
# ✓ Always call consumed() in SYNC mode
def on_frame(self, frame):
    process(frame)
    channel = self.get_tensor_channel("frames")
    if channel:
        channel.consumed()  # Required!

# ✓ Avoid circular updates in callbacks
def on_lr_changed(self, old_lr, new_lr):
    # DON'T call env_set("learning_rate", ...) here!
    # Use another variable for synchronization
    pass

# ✓ Use timeouts when blocking
channel = self.get_tensor_channel("data")
if channel:
    tensor = channel.read(timeout=1.0)
    if tensor is None:
        print("Timeout waiting for data")
```

### Problem: "Transport error: cannot pickle object"

**Possible Causes:**
1. Trying to share non-pickleable data without `signal_only=True`
2. Publishing a complex object without proper `__getstate__`
3. Trying to share file handles or GPU resources

**Solutions:**
```python
# ✓ Use signal_only for non-pickleable data
manager.env_register("dataloader", signal_only=True)

# ✓ Make data pickleable or use a placeholder
class Model(nn.Module):
    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove unpickleable fields
        return state

# ✓ Share references, not objects
def on_data_available(self, _):
    # signal_only=True, so _ is None
    data = self.env_get("dataloader")  # fetch from parent process memory
```

### Problem: "Subscriber not receiving published messages"

**Possible Causes:**
1. Subscriber subscribed after publisher started
2. Topic name mismatch (typo)
3. Plugin crashed silently
4. Manager not started

**Debugging:**
```python
# ✓ Check plugin names
print(manager.list_plugins())  # ['Publisher', 'Subscriber']

# ✓ Check if plugins are running
import psutil
ps = psutil.Process(manager.get_plugin("MyPlugin").task_main.pid)
print(ps.is_running())

# ✓ Enable logging
from modularnet.logger import Logger
Logger.set_level("DEBUG")

# ✓ Verify topic name matching
class Publisher(PluginPubSub):
    def onUpdate(self):
        self.publish("my_topic", data)  # Note: topic name

class Subscriber(PluginPubSub):
    def onStart(self):
        self.subscribe("my_topic", callback)  # Must match exactly
```

### Problem: "High latency or performance degradation"

**Possible Causes:**
1. Using SYNC mode when LATEST is acceptable
2. Pickling large objects repeatedly
3. Subscriber callbacks too slow
4. Too many env_set calls

**Optimization:**
```python
# ✓ Use LATEST mode for high throughput
self.subscribe_tensor("activations", callback,
                     mode=TensorDeliveryMode.LATEST)

# ✓ Use publish_tensor() for tensors (faster than publish)
self.publish_tensor("gradients", grad)  # not self.publish(grad)

# ✓ Pre-allocate tensor channels
self.subscribe_tensor("data", callback, shape=(256, 512),
                     dtype=torch.float32)

# ✓ Batch env updates
# ✗ Bad: multiple env_set calls
for lr in learning_rates:
    manager.env_set("lr", lr)  # N pickle+serialize ops

# ✓ Good: update once per iteration
manager.env_set("lr", current_lr)  # 1 operation per step
```

### Problem: "Memory grows unbounded"

**Possible Causes:**
1. Messages pile up in queues (subscriber too slow)
2. Shared memory not properly freed
3. Env slots holding references to large objects

**Solutions:**
```python
# ✓ Consumer must keep up with producer
# Use LATEST mode if reader might fall behind
self.subscribe_tensor("data", callback, mode=TensorDeliveryMode.LATEST)

# ✓ Ensure env variables are cleared on plugin stop
def onStop(self):
    self.env_set("cache", None)  # release reference

# ✓ Monitor queue sizes (development)
import logging
logging.basicConfig(level=logging.DEBUG)

# ✓ Call stop_all_plugins() to release shared memory
manager.stop_all_plugins()
# Shared memory released when manager destroyed
```

## Performance Characteristics

### Latency

- **Pickle transport**: ~1-10ms per message (depends on size)
- **Shared tensor/module**: ~0.1ms per write (after buffer allocated)
- **Env variable update**: ~1ms per watcher notified (broadcast)

### Memory

- **Per plugin**: ~50MB (Python interpreter + process overhead)
- **Per topic**: ~100 bytes + message queue size
- **Per tensor channel**: tensor buffer size + ~1KB metadata

### Throughput

- **Messages/sec**: 1000-10000 (depends on message size and callbacks)
- **Tensors/sec**: 10000+ for LATEST mode, 1000+ for SYNC (with fast callbacks)
