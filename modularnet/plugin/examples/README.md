"""
Plugin System Examples

This directory contains toy examples demonstrating the features of the ModularNet plugin system.
Each example is a self-contained, runnable demonstration of specific plugin capabilities.

## Example Index

### 1. **example_count.py** - Basic Lifecycle
Demonstrates the fundamental plugin lifecycle.

**Features:**
- Plugin initialization (`on_start`)
- Main loop execution
- Plugin shutdown (`on_stop`)
- State tracking and cleanup

**Use Case:** Verify that plugins can be loaded, started, stopped, and unloaded.

**Run:**
```bash
python -m modularnet.plugin.examples.example_count
```

**Output:** Shows a counter incrementing, total count at shutdown.

---

### 2. **example_pubsub.py** - Pub/Sub Communication
Demonstrates publish/subscribe messaging between plugins.

**Features:**
- Publisher plugin that publishes to a topic
- Subscriber plugin that receives published messages
- Topic-based async communication
- State synchronization

**Use Case:** Enable data flow between independent plugins.

**Run:**
```bash
python -m modularnet.plugin.examples.example_pubsub
```

**Output:** Shows publisher incrementing, subscriber receiving updates.

---

### 3. **example_env_variables.py** - Environment Variables
Demonstrates shared state management through environment variables.

**Features:**
- Setting environment variables (`env_set`)
- Getting environment variables (`env_get`)
- Subscribing to environment variable changes (`env_subscribe`)
- Cross-plugin state sharing

**Plugins:**
- `ConfigWriter`: Periodically updates shared configuration
- `ConfigReader`: Subscribes to configuration changes and reacts

**Use Case:** Share configuration, parameters, or state between plugins.

**Run:**
```bash
python -m modularnet.plugin.examples.example_env_variables
```

**Output:** Shows configuration updates and subscriber reactions.

---

### 4. **example_rpc_calls.py** - Plugin-to-Plugin RPC
Demonstrates synchronous method invocation between plugins.

**Features:**
- Exposing methods as RPC endpoints
- Calling methods on other plugins (`plugin_call`)
- Passing arguments and receiving return values
- Synchronous request/response pattern

**Plugins:**
- `DataProvider`: Exposes data access methods (get_data, get_item, add_data)
- `DataConsumer`: Calls methods on DataProvider

**Use Case:** Allow plugins to request services from other plugins.

**Run:**
```bash
python -m modularnet.plugin.examples.example_rpc_calls
```

**Output:** Shows consumer requesting data, provider responding.

---

### 5. **example_data_pipeline.py** - Multi-Stage Data Pipeline
Demonstrates a multi-stage data processing pipeline.

**Features:**
- Chained pub/sub topics for data flow
- Stage-by-stage processing
- Coordination through topics and environment variables
- Progress tracking

**Stages:**
1. `DataSource`: Generates raw data
2. `DataProcessor`: Transforms raw data
3. `DataAggregator`: Aggregates processed data and tracks statistics

**Use Case:** Build complex data processing workflows.

**Run:**
```bash
python -m modularnet.plugin.examples.example_data_pipeline
```

**Output:** Shows data flowing through 3 stages, final statistics.

---

### 6. **example_worker_pool.py** - Worker Pool Pattern
Demonstrates load distribution across multiple plugin instances.

**Features:**
- Multiple instances of the same plugin class
- Load distribution via pub/sub queue
- Per-worker metrics tracking
- Dynamic worker pool

**Plugins:**
- `TaskQueue`: Distributes tasks to workers
- `Worker` (3 instances): Process tasks in parallel

**Use Case:** Scale work horizontally by spawning multiple workers.

**Run:**
```bash
python -m modularnet.plugin.examples.example_worker_pool
```

**Output:** Shows 3 workers processing tasks in parallel, per-worker metrics.

---

### 7. **example_configuration.py** - Configuration & Options
Demonstrates plugin customization through configuration options.

**Features:**
- Descriptor with default options
- Custom options at load time
- Option-driven behavior
- Configuration validation

**Plugins:**
- `Sensor`: Configurable sensor with sample_rate, noise_level, sensor_type
- `SensorLogger`: Configurable logging with verbose mode

**Use Case:** Create reusable plugins with customizable behavior.

**Run:**
```bash
python -m modularnet.plugin.examples.example_configuration
```

**Output:** Shows sensor readings logged with configured verbosity.

---

### 8. **example_error_handling.py** - Error Handling & Recovery
Demonstrates resilient plugins with error handling and retry logic.

**Features:**
- Graceful error handling (try/except)
- Retry logic with exponential backoff
- Health monitoring
- State tracking for diagnosis

**Plugins:**
- `UnreliableService`: Simulates a service with random failures
- `ResilientClient`: Calls service with retry logic
- `HealthMonitor`: Monitors system health and reports metrics

**Use Case:** Build fault-tolerant systems that recover from transient failures.

**Run:**
```bash
python -m modularnet.plugin.examples.example_error_handling
```

**Output:** Shows failures, retries, successes, and health metrics.

---

### 9. **example_ml_training.py** - ML Training Pipeline
Demonstrates a comprehensive ML training scenario combining all concepts.

**Features:**
- Multi-stage pipeline (data → train → evaluate)
- Checkpoint management (save best models)
- Metrics tracking and aggregation
- Real-time loss monitoring
- Training progression

**Stages:**
1. `DataLoader`: Generates training batches
2. `Trainer`: Computes loss on batches
3. `Evaluator`: Tracks best loss and saves checkpoints

**Use Case:** Real-world example of modular ML training system.

**Run:**
```bash
python -m modularnet.plugin.examples.example_ml_training
```

**Output:** Shows training progress, loss improvements, saved checkpoints.

---

## Common Patterns

### Pattern 1: Setting Up Plugins
```python
manager = PluginManager()
descriptor = MyPlugin.Descriptor(options={...})
manager.load_plugin(descriptor)
manager.start_plugin(descriptor.name)
```

### Pattern 2: Publishing Data
```python
def main(self):
    data = {"id": 1, "value": 100}
    self.topic_publish("my.topic", data)
```

### Pattern 3: Subscribing to Data
```python
def on_start(self):
    self.topic_subscribe("my.topic", self.on_data)

def on_data(self, data):
    print(f"Received: {data}")
```

### Pattern 4: Sharing State
```python
# Set a value
self.env_set("my.state", value)

# Get a value
value = self.env_get("my.state")

# Subscribe to changes
self.env_subscribe("my.state", self.on_state_change)
```

### Pattern 5: Calling Other Plugins
```python
# Public methods in plugin
def my_method(self, arg1, arg2):
    return result

# Call from another plugin
result = self.plugin_call("plugin_id", "my_method", arg1, arg2)
```

---

## Running All Examples

```bash
# Run each example individually
python -m modularnet.plugin.examples.example_count
python -m modularnet.plugin.examples.example_pubsub
python -m modularnet.plugin.examples.example_env_variables
python -m modularnet.plugin.examples.example_rpc_calls
python -m modularnet.plugin.examples.example_data_pipeline
python -m modularnet.plugin.examples.example_worker_pool
python -m modularnet.plugin.examples.example_configuration
python -m modularnet.plugin.examples.example_error_handling
python -m modularnet.plugin.examples.example_ml_training
```

---

## Learning Path

**Beginner:** Start with basic examples
1. `example_count.py` - Understand lifecycle
2. `example_pubsub.py` - Learn pub/sub
3. `example_env_variables.py` - Understand shared state

**Intermediate:** Learn advanced patterns
4. `example_rpc_calls.py` - Plugin communication
5. `example_data_pipeline.py` - Multi-stage processing
6. `example_configuration.py` - Configuration management

**Advanced:** Real-world patterns
7. `example_worker_pool.py` - Scaling
8. `example_error_handling.py` - Resilience
9. `example_ml_training.py` - Complex systems

---

## Key Concepts

### Plugin Lifecycle
1. **Load**: Instantiate plugin class in subprocess
2. **Start**: Call `on_start()` to initialize
3. **Run**: Repeatedly call `main()` loop
4. **Stop**: Call `on_stop()` to cleanup
5. **Unload**: Terminate subprocess

### Communication Patterns
- **Pub/Sub**: Asynchronous, one-to-many (topics)
- **Environment Variables**: Shared state, async notifications
- **RPC**: Synchronous, one-to-one (method calls)

### Naming Conventions
- Plugin IDs: kebab-case (e.g., "data-provider")
- Topics: dot-separated (e.g., "training.batch")
- Environment variables: dot-separated (e.g., "config.learning_rate")

---

## Troubleshooting

**"Plugin failed to load":** Check that plugin class exists and `Descriptor()` is implemented.

**"Timeout waiting for response":** Plugin may be hanging or not responding. Check logs.

**"Topic not found":** Ensure subscriber subscribes after publisher announces topic.

**"Environment variable not found":** Use `env_get(name, default)` to provide defaults.

---

## Next Steps

After running these examples:
1. Combine concepts (e.g., add error handling to data pipeline)
2. Modify examples (e.g., add more stages to pipeline)
3. Create your own plugin for your use case
4. Integrate with your application
