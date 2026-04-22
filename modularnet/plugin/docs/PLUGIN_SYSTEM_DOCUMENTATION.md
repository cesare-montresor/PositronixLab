# ModularNet Plugin System - Documentation Index

## Quick Navigation

### 📚 Documentation Files

#### **1. [PLUGIN_SYSTEM_COMPREHENSIVE.md](modularnet/plugin/docs/PLUGIN_SYSTEM_COMPREHENSIVE.md)** (1900+ lines)
**The complete guide** - Start here for comprehensive understanding
- Full feature overview and architecture diagrams
- Complete API reference for all classes
- 5 detailed usage examples  
- Best practices section
- Troubleshooting guide
- Performance characteristics

#### **2. [API_REFERENCE.md](modularnet/plugin/docs/API_REFERENCE.md)** (600+ lines)
**Quick lookup reference** - Use this for quick API searches
- Class hierarchy overview
- All method signatures with parameter tables
- Callback signatures
- Common code patterns
- Error handling patterns
- Performance optimization tips

### 🧪 Test Files

#### **Test Suite Overview** (200+ tests, 1500+ lines)

| Test File | Purpose | Tests |
|-----------|---------|-------|
| [test_plugin_base.py](modularnet/plugin/tests/test_plugin_base.py) | PluginBase, PluginManagerBase | 30+ |
| [test_plugin_transport.py](modularnet/plugin/tests/test_plugin_transport.py) | TensorChannel, transport detection | 50+ |
| [test_plugin_pubsub.py](modularnet/plugin/tests/test_plugin_pubsub.py) | Pub/sub communication | 60+ |
| [test_plugin_env.py](modularnet/plugin/tests/test_plugin_env.py) | Environment variables | 40+ |
| [test_plugin_integration.py](modularnet/plugin/tests/test_plugin_integration.py) | Full system integration | 25+ |

### 📖 Getting Started

#### For New Users
1. Read [PLUGIN_SYSTEM_COMPREHENSIVE.md](modularnet/plugin/docs/PLUGIN_SYSTEM_COMPREHENSIVE.md) - Features & Architecture
2. Read **Usage Examples** section
3. Look at [test_plugin_pubsub.py](modularnet/plugin/tests/test_plugin_pubsub.py) for working code

#### For API Reference
1. Check [API_REFERENCE.md](modularnet/plugin/docs/API_REFERENCE.md) for quick lookup
2. Or refer to PLUGIN_SYSTEM_COMPREHENSIVE.md for detailed explanations

#### For Troubleshooting
1. See Troubleshooting section in [PLUGIN_SYSTEM_COMPREHENSIVE.md](modularnet/plugin/docs/PLUGIN_SYSTEM_COMPREHENSIVE.md)
2. Run relevant tests from [modularnet/plugin/tests/](modularnet/plugin/tests/)

### 🚀 Running Tests

```bash
# Run all plugin tests
pytest modularnet/plugin/tests/ -v

# Run specific component tests
pytest modularnet/plugin/tests/test_plugin_base.py -v
pytest modularnet/plugin/tests/test_plugin_pubsub.py -v
pytest modularnet/plugin/tests/test_plugin_env.py -v
pytest modularnet/plugin/tests/test_plugin_transport.py -v
pytest modularnet/plugin/tests/test_plugin_integration.py -v

# Run with output display
pytest modularnet/plugin/tests/ -v -s

# Run with coverage report
pytest modularnet/plugin/tests/ --cov=modularnet.plugin
```

### 📋 Documentation Sections

#### In PLUGIN_SYSTEM_COMPREHENSIVE.md

1. **Features** - What the system can do
2. **Architecture** - System overview with diagrams
3. **Unified Transport System** - Automatic data type detection
4. **API Reference** - All classes and methods
5. **Usage Examples** - 5 complete working examples
6. **Best Practices** - 6 guidelines for good code
7. **Troubleshooting** - 5 common problems + solutions
8. **Performance Characteristics** - Latency, memory, throughput

#### In API_REFERENCE.md

1. **Quick Reference** - Class hierarchies
2. **Core Classes** - Detailed documentation
3. **Transport & Channels** - Tensor communication
4. **Constants & Functions** - Transport detection
5. **Common Patterns** - Reusable code templates
6. **Error Handling** - Exception handling patterns
7. **Performance Tips** - Optimization guidelines
8. **Debugging** - Diagnostic techniques

### 🎯 Test Coverage

- ✅ **Core Components**: Plugin lifecycle, registration, control
- ✅ **Pub/Sub**: Topics, subscriptions, multiple topics
- ✅ **Tensors**: Lazy allocation, SYNC/LATEST modes
- ✅ **Transport**: All 5 transport strategies
- ✅ **Environment Variables**: Registration, watching, callbacks
- ✅ **Integration**: Multi-component scenarios
- ✅ **Stress Tests**: High throughput, many topics
- ✅ **Error Handling**: Exception recovery, timeouts

### 💡 Key Concepts

#### Transport System
The system automatically detects data types and selects optimal transport:
- **PICKLE** - Small objects (default)
- **SHARED_MODULE** - Neural network modules
- **SHARED_TENSOR** - CPU tensors (zero-copy)
- **CUDA_IPC** - CUDA tensors
- **SIGNAL_ONLY** - Non-pickleable data

#### Delivery Modes
- **SYNC** - Writer blocks until reader consumes (exact capture)
- **LATEST** - Writer never blocks, reader gets newest (high throughput)

#### Plugin Types
- **PluginBase** - Basic plugin with lifecycle
- **PluginPubSub** - Add pub/sub messaging
- **PluginEnv** - Add environment variables
- **Plugin** - Both pub/sub and env variables

### 📝 Example: Complete Training Monitor

```python
from modularnet.plugin.plugin import Plugin, PluginManager

class Trainer(Plugin):
    def __init__(self):
        super().__init__(name="Trainer")
        self.env_declare("learning_rate", default=1e-3, watch=True)
        self.loss = 1.0
    
    def onUpdate(self):
        lr = self.env_get("learning_rate")
        self.loss *= 0.95  # Simulate training
        self.publish("metrics", {"loss": self.loss, "lr": lr})
        time.sleep(0.1)

class Monitor(Plugin):
    def __init__(self):
        super().__init__(name="Monitor")
        self.subscribe("metrics", self.on_metrics)
    
    def on_metrics(self, metrics):
        print(f"Loss: {metrics['loss']:.4f}, LR: {metrics['lr']:.2e}")

# Run
manager = PluginManager()
manager.env_register("learning_rate", default=1e-3)
manager.register_plugin(Trainer())
manager.register_plugin(Monitor())
manager.start_all_plugins()

time.sleep(2)
manager.env_set("learning_rate", 5e-4)  # Update learning rate
time.sleep(2)
manager.stop_all_plugins()
```

See [PLUGIN_SYSTEM_COMPREHENSIVE.md](modularnet/plugin/docs/PLUGIN_SYSTEM_COMPREHENSIVE.md) for more examples!

---

**Last Updated**: April 22, 2026  
**Total Documentation**: 2500+ lines  
**Total Tests**: 200+ test cases covering all components  
**Status**: ✅ Complete and comprehensive
