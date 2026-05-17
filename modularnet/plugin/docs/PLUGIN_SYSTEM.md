# ModularNet Plugin System (Ray-Based)

A lean, high-performance plugin architecture built on Ray for distributed, multi-process plugin execution.

## Overview

The ModularNet plugin system provides:

- **Process Isolation**: Each plugin runs in its own Ray actor (isolated process)
- **Control Channel**: Manager ↔ Plugin communication for lifecycle management
- **High-Bandwidth Pub/Sub**: Topic-based message routing with zero-copy data transfer via ObjectRef
- **Shared Environment**: Global variables with change notification subscriptions
- **Services**: Synchronous request/response RPC calls
- **Lifecycle Events**: Plugins emit onStart, onUpdate, onStop, onError events
- **Dynamic Loading**: Plugins loaded via serializable descriptors (package, class, version, etc.)
- **Lean & Lightweight**: Minimal overhead, production-ready code

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PluginManager (Main Process)             │
│  - Loads/unloads plugins                                    │
│  - Pub/sub routing                                          │
│  - Environment variable management                          │
│  - Status monitoring                                        │
└────────────┬────────────────────────────────────────────────┘
             │
             │ Ray IPC
             │
┌────────────┴─────────────┬──────────────────┬──────────────┐
│                          │                  │              │
▼                          ▼                  ▼              ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌───────────┐
│   Plugin 1   │  │   Plugin 2   │  │   Plugin 3   │  │  Plugin N │
│  (Isolated   │  │  (Isolated   │  │  (Isolated   │  │ (Isolated │
│   Process)   │  │   Process)   │  │   Process)   │  │  Process) │
└──────────────┘  └──────────────┘  └──────────────┘  └───────────┘

Communication Channels:
─────────────────────

1. Control Plane: Commands (load, start, stop, config update)
2. Data Plane: Pub/sub topics (high-bandwidth, zero-copy via ObjectRef)
3. State Plane: Environment variables with onChange subscriptions
4. Service Plane: Request/response RPC calls
```

## Quick Start

### 1. Define a Plugin

```python
from modularnet.plugin import (
    BasePlugin,
    Subscription,
    EnvUpdate
)
from typing import Any, Dict, List

class MyPlugin(BasePlugin):
    """Example plugin."""
    
    def __init__(self, plugin_id: str, config: Dict[str, Any] = None):
        super().__init__(plugin_id, config)
        self.data = []
    
    def on_start(self) -> None:
        """Initialize when plugin starts."""
        print(f"Starting {self.plugin_id}")
    
    def on_stop(self) -> None:
        """Clean up when plugin stops."""
        print(f"Stopping {self.plugin_id}")
    
    def on_update(self, config: Dict[str, Any]) -> None:
        """Handle configuration updates."""
        self.config = config
        print(f"Updated config: {config}")
    
    # --- Pub/Sub Topics ---
    def get_subscriptions(self) -> List[Subscription]:
        """Subscribe to input_topic, process, republish to output_topic."""
        return [
            Subscription(
                name="input_topic",
                callback="process_message",
                publish_to="output_topic"  # Optional republishing
            )
        ]
    
    def process_message(self, data: Any, env: Dict[str, Any]) -> Any:
        """Called when message arrives on subscribed topic."""
        print(f"Processing: {data}")
        self.data.append(data)
        return {"processed": data}  # Republished to output_topic
    
    # --- Environment Variable Subscriptions ---
    def get_env_subscriptions(self) -> List[Subscription]:
        """Subscribe to environment variable changes."""
        return [
            Subscription(
                name="learning_rate",
                callback="on_learning_rate_change"
            )
        ]
    
    def on_learning_rate_change(self, update: EnvUpdate) -> None:
        """Called when subscribed env var changes."""
        print(f"Learning rate changed: {update.old_value} → {update.value}")
    
    # --- Services (Sync RPC) ---
    def get_services(self) -> List[str]:
        """Declare available service methods."""
        return ["get_statistics", "set_parameter"]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Service: return current statistics."""
        return {"messages_processed": len(self.data)}
    
    def set_parameter(self, name: str, value: Any) -> bool:
        """Service: set a parameter."""
        self.config[name] = value
        return True
```

### 2. Load and Manage Plugin

```python
from modularnet.plugin import PluginManager, PluginDescriptor

# Create manager (initializes Ray if needed)
manager = PluginManager()

# Create plugin descriptor (serializable metadata)
descriptor = PluginDescriptor(
    name="my_plugin",
    version="1.0.0",
    package="my_app.plugins",           # Full module path
    class_name="MyPlugin",               # Class name in module
    description="My awesome plugin",
    author="John Doe",
    options={"param1": "default_value"}  # Default config
)

# Load, start, stop plugins
manager.load_plugin(descriptor, config={"param1": "custom_value"})
manager.start_plugin("my_plugin")

# Monitor status
status = manager.get_plugin_status("my_plugin")
print(status)

# Send messages via pub/sub
manager.publish("input_topic", {"command": "process", "data": [1, 2, 3]})

# Set environment variables (broadcast to subscribers)
manager.set_env_variable("learning_rate", 0.001)

# Invoke services
result = manager.invoke_service("my_plugin", "get_statistics")
print(result)

# Update plugin config
manager.update_plugin_config("my_plugin", {"param1": "new_value"})

# Stop and unload
manager.stop_plugin("my_plugin")
manager.unload_plugin("my_plugin")
```

### 3. Processing Loop

For message-driven systems, start a background processing loop:

```python
# Start continuous message processing
manager.start_processing_loop(interval=0.1)  # Check every 100ms

# ... your code publishes messages and sets env vars ...

# Stop when done
manager.stop_processing_loop()
manager.shutdown()
```

## Core Components

### PluginDescriptor

Serializable metadata for dynamic plugin loading:

```python
from modularnet.plugin import PluginDescriptor

descriptor = PluginDescriptor(
    name="plugin_id",
    version="1.0.0",
    package="my.module.path",
    class_name="PluginClassName",
    description="What it does",
    author="Author name",
    options={"key": "value"},           # Default config
    dependencies=["other_plugin_id"]    # Required plugins
)

# Serialize for storage/transmission
json_str = descriptor.to_json()
restored = PluginDescriptor.from_json(json_str)
```

### BasePlugin

Abstract base class for all plugins. Subclass and implement:

**Required:**
- `on_start()`: Initialize resources
- `on_stop()`: Clean up resources

**Optional Overrides:**
- `on_update(config)`: Handle config changes
- `on_error(error, context)`: Handle errors
- `get_subscriptions()`: Define pub/sub subscriptions
- `get_env_subscriptions()`: Define env var subscriptions
- `get_services()`: List service methods

### PluginManager

High-level interface for plugin management:

```python
manager = PluginManager(auto_start_ray=True)

# Lifecycle
manager.load_plugin(descriptor, config={})
manager.start_plugin(plugin_id)
manager.stop_plugin(plugin_id)
manager.update_plugin_config(plugin_id, config)
manager.unload_plugin(plugin_id)

# Pub/Sub
manager.publish(topic, data)
manager.publish_object_ref(topic, numpy_array)  # Zero-copy

# Environment
manager.set_env_variable(key, value)
manager.get_env_variable(key, default)
manager.get_all_env_variables()

# Services
result = manager.invoke_service(plugin_id, method, *args, **kwargs)

# Monitoring
manager.get_plugin_status(plugin_id)
manager.get_all_plugin_status()
manager.get_plugin_errors(plugin_id)
manager.list_plugins()

# Processing
manager.start_processing_loop(interval=0.1)
manager.stop_processing_loop()
manager.shutdown()
```

### PluginOrchestrator (Ray Actor)

Low-level Ray actor that coordinates all plugins:
- Manages plugin registry
- Routes pub/sub messages
- Broadcasts environment variables
- Dispatches service calls

Typically accessed indirectly through PluginManager.

## Communication Patterns

### 1. Control Channel

Manager sends commands to plugins:

```
Manager → Load → Plugin Wrapper → Plugin.on_start()
Manager → Configure → Plugin Wrapper → Plugin.on_update()
Manager → Start/Stop → Plugin Wrapper → Plugin lifecycle
```

### 2. Pub/Sub (Data Plane)

High-bandwidth, topic-based message routing:

```python
# Publish small messages
manager.publish("my_topic", {"cmd": "run"})

# Publish large data via zero-copy ObjectRef
import numpy as np
large_array = np.random.randn(1000000)
manager.publish_object_ref("tensor_topic", large_array)

# Plugin receives and processes
class MyPlugin(BasePlugin):
    def get_subscriptions(self):
        return [Subscription("my_topic", "on_message")]
    
    def on_message(self, data, env):
        # data is the published object (automatically dereferenced)
        print(f"Received: {data}")
```

### 3. Environment Variables (State Plane)

Shared variables accessible to all plugins with change notifications:

```python
# Publish an environment variable
manager.set_env_variable("training_enabled", True)
manager.set_env_variable("learning_rate", 0.001)

# Plugin subscribes
class MyPlugin(BasePlugin):
    def get_env_subscriptions(self):
        return [
            Subscription("training_enabled", "on_training_toggle"),
            Subscription("learning_rate", "on_lr_change"),
        ]
    
    def on_training_toggle(self, update: EnvUpdate):
        print(f"Training: {update.old_value} → {update.value}")
    
    def on_lr_change(self, update: EnvUpdate):
        self.learning_rate = update.value
```

### 4. Services (Request/Response)

Synchronous RPC calls for immediate results:

```python
# Plugin defines service
class MyPlugin(BasePlugin):
    def get_services(self):
        return ["get_loss", "set_weights"]
    
    def get_loss(self) -> float:
        return self.current_loss
    
    def set_weights(self, weights: np.ndarray) -> bool:
        self.model.set_weights(weights)
        return True

# Manager invokes service
loss = manager.invoke_service("my_plugin", "get_loss")
success = manager.invoke_service("my_plugin", "set_weights", some_weights)
```

## Error Handling

Plugins can implement custom error handlers:

```python
class MyPlugin(BasePlugin):
    def on_error(self, error: Exception, context: str = "") -> None:
        """
        Called when an error occurs.
        
        Args:
            error: The exception
            context: Where it occurred ("on_start", "callback_name", etc.)
        """
        logger.error(f"Plugin error in {context}: {error}")
        # Custom recovery logic
```

Monitor errors from manager:

```python
errors = manager.get_plugin_errors("my_plugin")
for error in errors:
    print(error)
```

## Best Practices

1. **Keep Callbacks Fast**: Pub/sub callbacks should be quick. Offload heavy computation.

2. **Use ObjectRef for Large Data**: For numpy arrays, tensors, or large objects:
   ```python
   manager.publish_object_ref("topic", large_array)  # Zero-copy
   ```

3. **Read Env Vars Efficiently**: Cache frequently-accessed env vars locally:
   ```python
   def on_message(self, data, env):
       # env is a local cache - O(1) lookups
       lr = env.get("learning_rate")
   ```

4. **Handle Backpressure**: If queue is full, messages are dropped. Design for this.

5. **Clean Up Resources**: Implement proper `on_stop()` to release resources.

6. **Versioning**: Use semantic versioning in descriptors for compatibility.

7. **Dependencies**: Declare plugin dependencies in descriptor for startup ordering.

## Performance Characteristics

- **Plugin Startup**: ~100-500ms per plugin (Ray actor creation)
- **Pub/Sub Latency**: <1ms (same-machine, direct queue)
- **ObjectRef Transfer**: Near-zero copy (shared memory via Ray Plasma)
- **Env Variable Access**: O(1) local dictionary lookups
- **Service Calls**: ~1-5ms (Ray RPC overhead)

## File Structure

```
modularnet/plugin/
├── __init__.py                    # Public API exports
├── plugin_descriptor.py           # PluginDescriptor class
├── plugin_base.py                 # BasePlugin abstract class
├── environment_manager.py         # EnvironmentManager for env vars
├── plugin_wrapper.py              # PluginHandler (Ray actor)
├── plugin_manager.py              # PluginManager + PluginOrchestrator
├── docs/
│   ├── ray_as_core_for_plugin_system.md  # Architecture details
│   └── PLUGIN_SYSTEM.md          # This file
└── tests/
    ├── __init__.py
    └── test_plugin_system.py      # Comprehensive tests
```

## Migration from Old Plugin System

If migrating from `plugin_old/`:

1. Create `PluginDescriptor` with metadata
2. Create new class inheriting from `BasePlugin`
3. Implement `on_start()` and `on_stop()` (required)
4. Add pub/sub via `get_subscriptions()` if needed
5. Add services via `get_services()` if needed
6. Use `PluginManager` for lifecycle management

Example migration:

```python
# OLD: plugin_old/my_plugin.py
class OldPlugin:
    def __init__(self, config):
        self.config = config
    
    def initialize(self):
        pass
    
    def shutdown(self):
        pass

# NEW: my_app/plugins/my_plugin.py
from modularnet.plugin import BasePlugin

class MyNewPlugin(BasePlugin):
    def on_start(self):
        # ... old initialize() code ...
        pass
    
    def on_stop(self):
        # ... old shutdown() code ...
        pass
    
    def get_services(self):
        return ["old_method_1", "old_method_2"]
    
    def old_method_1(self):
        pass
    
    def old_method_2(self):
        pass
```

## Requirements

- Python 3.8+
- Ray 2.0+

Install Ray:
```bash
pip install ray
```

## License

Same as ModularNet project.
