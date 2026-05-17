# Inter-Plugin Communication Guide

## Overview

Plugins can communicate with each other directly through methods inherited from `BasePlugin`. This enables:

- **Plugin-to-Plugin Messaging**: Plugins publish messages that other plugins subscribe to
- **Environment Variables**: Plugins set and monitor shared environment variables
- **Service Invocation**: Plugins invoke methods on other plugins
- **Zero-Copy Data Transfer**: Use ObjectRef for large data

## Communication API

Each plugin inherits these communication methods from `BasePlugin`:

```python
from modularnet.plugin import BasePlugin

class MyPlugin(BasePlugin):
    def on_start(self):
        # Direct method calls - no wrapper object needed
        self.publish("my_topic", {"data": "hello"})
        self.set_env_variable("key", "value")
        result = self.invoke_service("other_plugin", "method", arg1)
```

### Publishing Messages

Send messages to topics that other plugins subscribe to:

```python
# Simple message
self.publish("my_topic", {"data": "hello"})

# Large data (zero-copy via ObjectRef)
import numpy as np
large_array = np.random.randn(1000, 1000)
self.publish_object_ref("tensor_topic", large_array)

# Returns number of subscribers that received the message
num_subscribers = self.publish("topic", data)
```

### Setting Environment Variables

Set shared state that other plugins can monitor:

```python
# Set a variable (notifies all subscribers)
self.set_env_variable("model_path", "/path/to/model.pt")

# Get a variable
model_path = self.get_env_variable("model_path")

# Get with default
timeout = self.get_env_variable("timeout", default=30)
```

### Invoking Services

Call methods on other plugins synchronously:

```python
# Invoke a service on another plugin
result = self.invoke_service(
    plugin_id="data_processor",
    method_name="process_batch",
    batch_data,  # positional args
    format="numpy"  # keyword args
)
```

## Example: Multi-Plugin Pipeline

### Data Source Plugin (Publishes)

```python
from modularnet.plugin import BasePlugin

class DataSourcePlugin(BasePlugin):
    def on_start(self):
        # Set initial configuration
        self.set_env_variable("batch_size", 32)
        self.set_env_variable("num_workers", 4)
    
    def on_stop(self):
        pass
    
    def get_services(self):
        return ["load_batch"]
    
    def load_batch(self, batch_id: int):
        # Load batch and publish it
        data = self._load_data(batch_id)
        self.publish("data_batch", {
            "batch_id": batch_id,
            "data_ref": data  # Will be converted to ObjectRef
        })
        return {"status": "published"}
```

### Processing Plugin (Subscribes & Publishes)

```python
from modularnet.plugin import BasePlugin, Subscription

class ProcessorPlugin(BasePlugin):
    def __init__(self, plugin_id, config, orchestrator_handle=None):
        super().__init__(plugin_id, config, orchestrator_handle)
        self.processed_count = 0
    
    def on_start(self):
        # Announce we're ready
        self.set_env_variable("processor_ready", True)
    
    def on_stop(self):
        self.set_env_variable("processor_ready", False)
    
    def get_subscriptions(self):
        # Subscribe to data batches
        return [
            Subscription(
                name="data_batch",
                callback="on_batch_received",
                publish_to="processed_batch"  # Forward results
            )
        ]
    
    def on_batch_received(self, data: dict, env: dict):
        # Process data
        batch_id = data["batch_id"]
        processed = self._process(data["data_ref"])
        
        # Publish results
        self.publish("processed_batch", {
            "batch_id": batch_id,
            "result": processed
        })
        
        self.processed_count += 1
    
    def get_services(self):
        return ["get_stats"]
    
    def get_stats(self):
        return {"processed": self.processed_count}
```

### Monitor Plugin (Watches Everything)

```python
from modularnet.plugin import BasePlugin, Subscription

class MonitorPlugin(BasePlugin):
    def __init__(self, plugin_id, config, orchestrator_handle=None):
        super().__init__(plugin_id, config, orchestrator_handle)
        self.batch_size = None
        self.processor_ready = False
    
    def on_start(self):
        # Query service on processor
        stats = self.invoke_service("processor", "get_stats")
        print(f"Processor stats: {stats}")
    
    def on_stop(self):
        pass
    
    def get_env_subscriptions(self):
        return [
            Subscription(
                name="batch_size",
                callback="on_batch_size_change"
            ),
            Subscription(
                name="processor_ready",
                callback="on_processor_status_change"
            ),
        ]
    
    def on_batch_size_change(self, update):
        self.batch_size = update.value
        print(f"Batch size changed to: {update.value}")
    
    def on_processor_status_change(self, update):
        self.processor_ready = update.value
        print(f"Processor ready: {update.value}")
```

## Best Practices

1. **Check Orchestrator Exists**: Methods will raise RuntimeError if orchestrator not available
   - This happens in standalone testing or if plugin wasn't passed orchestrator handle
   - Wrap calls in try-except for robust code

2. **Error Handling**: Wrap communication calls in try-except
   ```python
   try:
       self.publish("topic", data)
   except RuntimeError as e:
       self.logger.error(f"Failed to publish: {e}")
   ```

3. **Service Discovery**: Implement a registry pattern if you have many plugins
   ```python
   def on_start(self):
       # Register yourself with a registry plugin
       self.invoke_service("registry", "register", self.plugin_id, ["method1", "method2"])
   ```

4. **Environment Variable Naming**: Use hierarchical names to avoid conflicts
   ```python
   # Good - scoped to plugin name
   self.set_env_variable("processor.model_path", path)
   self.set_env_variable("processor.batch_size", 32)
   
   # Bad - too generic
   self.set_env_variable("model_path", path)
   ```

5. **Pub/Sub Patterns**: Keep topics well-defined
   - `data_*` for data messages
   - `control_*` for control flow
   - `status_*` for status updates

## Performance Considerations

- **Local Calls**: Environment variables are cached locally for O(1) access
- **Remote Calls**: Services are RPC calls - keep payloads reasonable
- **Large Data**: Use `publish_object_ref()` for arrays/tensors (zero-copy via Ray)
- **Pub/Sub Volume**: Messages are queued - monitor queue depth in high-throughput scenarios

## Migration from Separate Context Object

### Before (with PluginContext):
```python
class MyPlugin(BasePlugin):
    def on_start(self):
        self.context.publish("topic", data)
        self.context.set_env_variable("key", value)
```

### After (Integrated):
```python
class MyPlugin(BasePlugin):
    def on_start(self):
        self.publish("topic", data)
        self.set_env_variable("key", value)
```

Much cleaner!
