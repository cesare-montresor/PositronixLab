# TrialDeepLearningCore Documentation

## Overview

`TrialDeepLearningCore` is a specialized trial implementation that orchestrates the complete deep learning pipeline. It extends `TrialBase` and automatically initializes and manages two core plugins:

1. **DeeplearningSetupPlugin** - Initializes model and data provider
2. **DeeplearningTrainPlugin** - Handles model training

This class provides a streamlined interface for running deep learning experiments with automatic plugin dependency management.

## Class Hierarchy

```
TrialBase
    └── TrialDeepLearningCore
```

## Usage

### Basic Usage

```python
from positronixlab.config.trial_config import TrialConfig
from positronixlab.trial.trial_manager import TrialManager
from positronixlab.trial.trail import TrialDeepLearningCore

# Create configuration and trial manager
config = TrialConfig(options={...})
trial_manager = TrialManager(TrialDeepLearningCore, config, trial_manager)

# Initialize trial
trial = TrialDeepLearningCore(config, trial_manager)

# Run the trial (setup + training)
trial.run()
```

### Plugin Execution Order

The plugins are loaded in dependency order:

1. **deeplearning_setup** (DeeplearningSetupPlugin)
   - Creates the neural network model
   - Initializes the data provider
   - Configures the device (CPU/GPU)

2. **deeplearning_train** (DeeplearningTrainPlugin)
   - Trains the model using provided data
   - Monitors metrics and logs progress
   - Saves checkpoints

## Constructor

### `__init__(config: TrialConfig, trial_manager: TrialManager)`

**Parameters:**
- `config` (TrialConfig): Configuration object containing all trial parameters including:
  - Model architecture settings
  - Training hyperparameters
  - Dataset specifications
  - Criterion and optimizer settings
  
- `trial_manager` (TrialManager): Manager instance for trial lifecycle management

**Behavior:**
- Calls parent `TrialBase.__init__()` to initialize base infrastructure
- Registers `DeeplearningSetupPlugin` with the plugin manager
- Registers `DeeplearningTrainPlugin` with the plugin manager
- Plugins are initialized in dependency order

## Inherited Methods

### `load_plugin(plugin_class, *args, **kwargs)`

Loads and registers a plugin with the plugin manager.

**Returns:** The instantiated plugin instance

### `run()`

Executes all registered plugins in order:

```python
trial.run()  # Calls plugin_manager.start_all_plugins()
```

## Plugin Details

### DeeplearningSetupPlugin

Initializes the deep learning environment:
- Creates MetaModel instance
- Creates DataProvider instance
- Sets up device configuration
- Registers components in environment

### DeeplearningTrainPlugin

Executes the training loop:
- Iterates through training batches
- Computes loss and updates model
- Logs metrics
- Handles early stopping if configured

## Configuration Requirements

The `config` object should include:

- **Model Settings**: Architecture, hidden dimensions, activation functions
- **Training Settings**: Learning rate, batch size, number of epochs
- **Dataset Settings**: Dataset name, split ratios, preprocessing
- **Loss Settings**: Loss function type (e.g., 'mse', 'crossentropy')
- **Optimizer Settings**: Optimizer type and parameters

See [CONFIG_USAGE.md](../config/docs/CONFIG_USAGE.md) for detailed configuration examples.

## Example Workflow

```python
import torch
from positronixlab.config.trial_config import TrialConfig
from positronixlab.trial.trial_manager import TrialManager
from positronixlab.trial.trail import TrialDeepLearningCore

# 1. Create configuration
config = TrialConfig(options={
    'exp_dataset': 'mnist',
    'exp_metric_modularity': True,
    'hidden_dims': [128, 64],
    'learning_rate': 0.001,
    'num_epochs': 10,
    'batch_size': 32,
})

# 2. Create trial manager
trial_manager = TrialManager(TrialDeepLearningCore, config, trial_manager)

# 3. Initialize trial
trial = TrialDeepLearningCore(config, trial_manager)

# 4. Run the complete pipeline
trial.run()  # Setup + Training
```

## Extension

To create a custom trial with different plugins, extend `TrialBase`:

```python
from positronixlab.trial.trail import TrialBase
from positronixlab.plugin.plugin import PluginManager

class CustomTrial(TrialBase):
    def __init__(self, config, trial_manager):
        super().__init__(config, trial_manager)
        # Load custom plugins
        self.load_plugin(CustomPlugin1, name="custom1", config=config)
        self.load_plugin(CustomPlugin2, name="custom2", config=config)
```

## Error Handling

If any plugin fails during execution:
- The error is caught by the plugin manager
- Execution may halt depending on plugin `fatal_errors` setting
- Check plugin logs for detailed error information

## Performance Considerations

- **Memory**: Model size and batch size affect memory usage
- **Computation**: GPU acceleration recommended for large models
- **Data Loading**: Number of workers affects data pipeline efficiency

## Related Documentation

- [TRIAL_SYSTEM.md](./TRIAL_SYSTEM.md) - Trial system architecture
- [CONFIG_USAGE.md](../config/docs/CONFIG_USAGE.md) - Configuration guide
- [DEEPLEARNING_CORE.md](../deeplearning_core/docs/DEEPLEARNING_CORE.md) - Deep learning core documentation
- [PLUGIN_SYSTEM.md](../plugin/docs/PLUGIN_SYSTEM.md) - Plugin system documentation
