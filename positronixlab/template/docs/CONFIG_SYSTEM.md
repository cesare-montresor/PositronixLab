# Config System Documentation

## Overview

The config system in PositronixLab provides a flexible framework for defining, managing, and composing machine learning experiment configurations. It uses a template-based approach where configurations are built through parameter definitions that can be of different types: **fixed**, **range**, or **choice**.

## Architecture

### Core Components

#### 1. **TemplateBase** (`template_base.py`)
The foundational class for all configuration templates. It manages:
- **Parameter definitions**: Fixed values, ranges, and choices
- **Variants**: Experiment variants (a, b, c, d) for ablation studies
- **Serialization**: Save/load templates to/from YAML files
- **Cloning**: Create independent copies of templates

#### 2. **TemplateFC & TemplateCNN** (`template_base.py`)
Specialized templates for different model architectures:
- **TemplateFC**: Fully connected network configurations
- **TemplateCNN**: Convolutional network configurations

#### 3. **Dataset-Specific Templates**
Preconfigured templates for specific datasets:
- `TemplateMnistFC`, `TemplateMnistCNN`: MNIST dataset
- `TemplateFMnistFC`, `TemplateFMnistCNN`: Fashion-MNIST dataset
- `TemplateCovertype`: Covertype (tabular) dataset
- `TemplateHiggs`: Higgs (large tabular) dataset

#### 4. **Template Registry** (`template.py`)
Central registry for managing and retrieving templates by name:
```python
Template.FromName('mnist_fc')  # Returns TemplateMnistFC instance
```

#### 5. **TrialConfig** (`trial_config.py`)
Runtime configuration object that materializes template parameters into concrete values for trial execution.

## Parameter Types

### Fixed Parameters
Static values that don't vary:
```python
template.add_fixed('task', 'classification', 'str')
template.add_fixed('input_data_shape', 784, 'int')
```

### Range Parameters
Continuous or discrete ranges for hyperparameter optimization:
```python
template.add_range('learning_rate', [0.0001, 0.01], 'float', log_scale=True)
template.add_range('batch_size', [32, 128], 'int')
```
- `log_scale`: When True, uses logarithmic scaling for the range

### Choice Parameters
Discrete selection from predefined values:
```python
template.add_choice('activation', ['relu', 'leaky_relu'], 'str')
template.add_choice('input_batch_norm', [True, False], 'bool')
```
- `is_ordered`: (Optional) Indicates if values have meaningful ordering
- `dependent_parameters`: (Optional) Parameters that depend on this choice

## Parameter Categories

### Experiment Configuration
- `exp_metric_modularity`: Enable/disable modularity metrics
- `exp_modularizer`: Enable/disable modularization

### Model Architecture
- `model_type`: 'fc' or 'cnn'
- `activation`: Activation function ('relu', 'leaky_relu', 'elu')
- `input_num_layers`, `output_num_layers`: Network depth
- `input_hidden_dim`, `output_hidden_dim`: Layer widths
- `input_batch_norm`, `output_batch_norm`: Batch normalization
- `input_residual`, `output_residual`: Residual connections
- `input_dropout_every`, `output_dropout_every`: Dropout scheduling

### Modular Space Configuration
- `modular_space_size`: Number of modules
- `modular_space_lr`: Learning rate for space optimization
- `modular_space_active_q`: Activation quantile threshold
- `modular_space_distance`: Distance metric ('euclidean', 'cosine')
- `modular_space_loss_*`: Individual loss component weights

### Training Configuration
- `train_epochs`: Number of training epochs
- `batch_size`: Batch size
- `learning_rate`: Optimization learning rate
- `dropout`: Global dropout rate

### Dataset Configuration
- `dataset_name`: Name of the dataset
- `input_data_shape`: Input feature/pixel count
- `output_data_shape`: Number of output classes

### NAS Configuration
- `nas_min_acc`: Minimum acceptable accuracy
- `nas_min_mod_score`: Minimum modularity score threshold

## Variants System

Templates support experimental variants for ablation studies:

```python
template = Template.FromName('mnist_fc')

# Get variant 'a': No modularity, no modularizer
variant_a = template.get_variant('a')

# Get variant 'd': Modularity enabled, modularizer enabled
variant_d = template.get_variant('d')
```

### Available Variants
- **'a'**: `exp_metric_modularity=False`, `exp_modularizer=False`
- **'b'**: `exp_metric_modularity=True`, `exp_modularizer=False`
- **'c'**: `exp_metric_modularity=False`, `exp_modularizer=True`
- **'d'**: `exp_metric_modularity=True`, `exp_modularizer=True`

## Usage Examples

### Creating a Template

```python
from positronixlab.config.template import Template

# Load a predefined template
template = Template.FromName('mnist_fc')

# Customize parameters
template.add_fixed('dropout', 0.3, 'float')
template.add_range('batch_size', [16, 64], 'int')
```

### Getting Parameter Values

```python
# Get a specific parameter value
learning_rate = template.get_value('learning_rate')

# Get all parameters
all_params = template.get_params()

# Get filename for saving
filename = template.filename()  # "template_mnist_fc.json"
```

### Working with Variants

```python
# Create variant in-place
template_a = template.get_variant('a', inplace=False)

# Create variant without modification (returns template with both options)
template_default = template.get_variant(None)
```

### Serialization

```python
# Save template to YAML
template.save('/path/to/template.yaml')

# Load template from YAML
loaded_template = TemplateBase.load('/path/to/template.yaml')

# Convert to/from dictionary
template_dict = template.to_dict()
new_template = TemplateBase.from_dict(template_dict)
```

### Using TrialConfig

```python
# Create trial configuration from parameter options
options = {
    'dataset_name': 'mnist',
    'train_epochs': 20,
    'batch_size': 64,
    'learning_rate': 0.001,
    'input_hidden_dim': 256
}

trial_config = TrialConfig(options)

# Access configuration values
print(trial_config.learning_rate)
print(trial_config.batch_size)
```

## Dataset-Specific Configuration

### MNIST/Fashion-MNIST
- Input shape: 784 (for FC) or 28 (for CNN)
- Output shape: 10 classes
- Default NAS accuracy: 0.85 (Fashion-MNIST), 0.95 (MNIST)

### Covertype
- Input shape: 54 features
- Output shape: 7 forest cover types
- Default NAS accuracy: 0.70
- Optimized for larger batch sizes (64-256)

### Higgs
- Input shape: 28 features (physics data)
- Output shape: 2 (binary classification)
- Default NAS accuracy: 0.70
- Supports deeper networks and larger batches

## Parameter Modification Patterns

### Adding Parameters
```python
# Adding to existing template
template = Template.FromName('mnist_fc')
template.add_choice('new_param', [1, 2, 3], 'int')
```

### Removing Parameters
```python
# Remove a parameter
template.remove('dropout')
```

### Cloning Templates
```python
# Deep copy for independent modification
custom_template = template.clone()
custom_template.add_fixed('custom_value', 42, 'int')
```

## Configuration Signatures

Templates can generate a unique signature representing their configuration:
```python
signature = template.signature()
# Used for identifying unique configurations
```

## File Structure

```
positronixlab/config/
├── __init__.py
├── template_base.py          # Core template classes
├── template_fmnist.py        # Fashion-MNIST templates
├── template_mnist.py         # MNIST templates
├── template_covertype.py     # Covertype template
├── template_higgs.py         # Higgs template
├── template.py               # Template registry
├── trial_config.py           # Trial configuration class
├── docs/
│   ├── CONFIG_SYSTEM.md      # This file
│   └── CONFIG_USAGE.md       # Usage guide
└── tests/
    ├── test_template_base.py
    ├── test_trial_config.py
    └── test_template_variants.py
```

## Best Practices

1. **Use Template Registry**: Always retrieve templates via `Template.FromName()` rather than instantiating directly
2. **Immutable Variants**: When creating variants, use `inplace=False` to preserve the original template
3. **Type Consistency**: Ensure parameter types match declarations (int, float, str, bool)
4. **Range Bounds**: Always provide valid bounds [min, max] with min < max
5. **Cloning for Modification**: Clone templates before making custom modifications
6. **Serialization**: Save templates regularly for reproducibility and sharing

## Dependencies

The config system depends on:
- `yaml`: For template serialization
- `copy`: For deep copying templates
- Standard library modules: `os`

## Notes

- Parameter bounds and ranges should follow dataset-specific requirements
- The variant system enables systematic ablation studies
- Templates are designed to be composable and extensible
