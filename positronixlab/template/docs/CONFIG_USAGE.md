# Config Module Usage Guide

## Quick Start

### Basic Template Usage

```python
from positronixlab.config.template import Template

# Load a predefined template
template = Template.FromName('mnist_fc')

# Access parameter values
print(template.get_value('learning_rate'))
print(template.get_value('batch_size'))

# Get all parameters
params = template.get_params()
for param in params:
    print(f"{param['name']}: {param}")
```

### Creating a Trial Configuration

```python
from positronixlab.pipeline import PipelineConfig

# Prepare options dictionary
options = {
    'dataset_name': 'mnist',
    'model_type': 'fc',
    'train_epochs': 20,
    'batch_size': 64,
    'learning_rate': 0.001,
    'input_hidden_dim': 256,
    'output_hidden_dim': 128,
    'dropout': 0.3,
    'activation': 'relu'
}

# Create trial configuration
trial_config = PipelineConfig(options)

# Use in training loop
model = create_model(
    input_dim=trial_config.input_data_shape,
    hidden_dim=trial_config.input_hidden_dim,
    output_dim=trial_config.output_data_shape,
    activation=trial_config.activation
)
```

## Working with Templates

### Selecting Dataset-Specific Templates

```python
from positronixlab.config.template import Template

# MNIST with fully connected network
mnist_fc = Template.FromName('mnist_fc')

# MNIST with CNN
mnist_cnn = Template.FromName('mnist_cnn')

# Fashion-MNIST
fmnist_fc = Template.FromName('fmnist_fc')
fmnist_cnn = Template.FromName('fmnist_cnn')

# Tabular datasets
covertype = Template.FromName('covertype')
higgs = Template.FromName('higgs')
```

### Template Customization

```python
from positronixlab.config.template import Template

template = Template.FromName('mnist_fc')

# Modify existing parameters
template.add_fixed('dropout', 0.25, 'float')

# Add new parameters
template.add_choice('optimizer', ['adam', 'sgd', 'rmsprop'], 'str')

# Remove unnecessary parameters
template.remove('cnn_kernel_size')

# Get filename for saving
print(template.filename())
```

## Variants and Ablation Studies

### Using Variants

```python
from positronixlab.config.template import Template

base_template = Template.FromName('mnist_fc')

# Create variant a: No modularity features
variant_a = base_template.get_variant('a')
print(variant_a.get_value('exp_metric_modularity'))  # False
print(variant_a.get_value('exp_modularizer'))        # False

# Create variant d: Full modularity
variant_d = base_template.get_variant('d')
print(variant_d.get_value('exp_metric_modularity'))  # True
print(variant_d.get_value('exp_modularizer'))        # True

# Create variant without locking
variant_flexible = base_template.get_variant(None)
```

### Running Ablation Studies

```python
from positronixlab.config.template import Template

variants = ['a', 'b', 'c', 'd']
template = Template.FromName('mnist_fc')

for variant in variants:
    variant_template = template.get_variant(variant)
    
    # Extract modularity settings
    has_modularity = variant_template.get_value('exp_metric_modularity')
    has_modularizer = variant_template.get_value('exp_modularizer')
    
    print(f"Variant {variant}: modularity={has_modularity}, modularizer={has_modularizer}")
    
    # Run trial with this variant configuration
    # trial_result = run_trial(variant_template)
```

## Parameter Management

### Adding Different Parameter Types

```python
from positronixlab.config.template import Template

template = Template.FromName('mnist_fc')

# Fixed parameter (always the same value)
template.add_fixed('task', 'classification', 'str')

# Range parameter (continuous or discrete range)
template.add_range('learning_rate', [0.0001, 0.1], 'float', log_scale=True)
template.add_range('hidden_dim', [32, 512], 'int')

# Choice parameter (select from predefined values)
template.add_choice('activation', ['relu', 'leaky_relu', 'elu'], 'str')
template.add_choice('batch_norm', [True, False], 'bool')

# Get parameter values
lr = template.get_value('learning_rate')
activation = template.get_value('activation')
```

### Parameter Value Retrieval

```python
from positronixlab.config.template import Template

template = Template.FromName('mnist_fc')

# Get single parameter value with default
learning_rate = template.get_value('learning_rate', default=0.001)

# Get all parameters as list
all_params = template.get_params()

# Iterate over parameters
for param in all_params:
    print(f"Name: {param.get('name')}")
    print(f"Type: {param.get('value_type')}")
    print(f"Parameter Type: {param.get('type')}")  # 'fixed', 'range', or 'choice'
```

## Serialization

### Saving Templates

```python
from positronixlab.config.template import Template

template = Template.FromName('mnist_fc')

# Save with default filename
template.save()  # Saves to default config path

# Save to specific path
template.save('/path/to/my_template.yaml')

# Save after customization
custom_template = template.clone()
custom_template.add_fixed('custom_param', 42, 'int')
custom_template.save('/path/to/custom_template.yaml')
```

### Loading Templates

```python
from positronixlab.config.template_base import TemplateBase

# Load template from YAML file
loaded_template = TemplateBase.load('/path/to/template.yaml')

# Use loaded template
config_signature = loaded_template.signature()
param_value = loaded_template.get_value('dropout')
```

### Dictionary Conversion

```python
from positronixlab.config.template import Template
from positronixlab.config.template_base import TemplateBase

template = Template.FromName('mnist_fc')

# Convert to dictionary
template_dict = template.to_dict()
print(template_dict)

# Create template from dictionary
new_template = TemplateBase.from_dict(template_dict)

# Useful for JSON serialization
import json
json_str = json.dumps(template_dict)
```

## Template Cloning and Modification

### Safe Template Modification

```python
from positronixlab.config.template import Template

# Original template
base_template = Template.FromName('mnist_fc')

# Clone for safe modification
custom_template = base_template.clone()

# Modify without affecting original
custom_template.add_fixed('custom_layer_size', 128, 'int')
custom_template.remove('cnn_kernel_size')

# Original remains unchanged
print(base_template.get_params())
print(custom_template.get_params())  # Different params
```

## Typical Workflow

### For Neural Architecture Search (NAS)

```python
from positronixlab.template import Template
from positronixlab.pipeline import PipelineConfig

# 1. Start with base template
template = Template.FromName('mnist_fc')

# 2. Create variant for ablation
variant = template.get_variant('d')

# 3. Sample hyperparameters (for random search or NAS)
sampled_params = {
    'learning_rate': 0.001,
    'batch_size': 64,
    'dropout': 0.3,
    'input_hidden_dim': 256,
    'train_epochs': 20
}

# 4. Merge with template defaults
options = {**variant.to_dict()['params'], **sampled_params}

# 5. Create trial configuration
trial_config = PipelineConfig(options)

# 6. Run trial
results = run_trial(trial_config)
```

### For Reproducible Experiments

```python
from positronixlab.config.template import Template
import datetime

# 1. Load and customize template
template = Template.FromName('covertype')
template.add_fixed('random_seed', 42, 'int')
template.add_fixed('experiment_date', str(datetime.date.today()), 'str')

# 2. Save template for reproducibility
experiment_id = f"covertype_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
template.save(f'./experiments/{experiment_id}.yaml')

# 3. Later: Reload and run
from positronixlab.config.template_base import TemplateBase
saved_template = TemplateBase.load(f'./experiments/{experiment_id}.yaml')

# 4. Run with exact same configuration
results = run_experiment(saved_template)
```

## Common Parameter Ranges by Dataset

### MNIST/Fashion-MNIST (Image Classification)
```python
from positronixlab.config.template import Template

template = Template.FromName('mnist_cnn')

# Recommended ranges
template.add_range('learning_rate', [0.0001, 0.01], 'float', log_scale=True)
template.add_range('batch_size', [32, 128], 'int')
template.add_range('train_epochs', [10, 50], 'int')
template.add_choice('activation', ['relu', 'leaky_relu'], 'str')
```

### Covertype (Tabular Data)
```python
from positronixlab.config.template import Template

template = Template.FromName('covertype')

# Recommended ranges (larger for tabular)
template.add_range('learning_rate', [0.0001, 0.01], 'float', log_scale=True)
template.add_range('batch_size', [64, 256], 'int')
template.add_range('train_epochs', [10, 50], 'int')
template.add_range('input_hidden_dim', [128, 512], 'int')
```

### Higgs (Large-Scale Tabular)
```python
from positronixlab.config.template import Template

template = Template.FromName('higgs')

# Recommended ranges (even larger for massive datasets)
template.add_range('learning_rate', [0.00001, 0.001], 'float', log_scale=True)
template.add_range('batch_size', [128, 512], 'int')
template.add_range('train_epochs', [20, 100], 'int')
template.add_range('input_hidden_dim', [256, 1024], 'int')
```

## Debugging and Inspection

### Inspecting Template Configuration

```python
from positronixlab.config.template import Template

template = Template.FromName('mnist_fc')

# Print all parameters
print("=" * 50)
print(f"Template: {template.name}")
print(f"Variant: {template.variant}")
print("=" * 50)

for param in template.get_params():
    print(f"\n{param['name']}:")
    print(f"  Type: {param['type']}")
    print(f"  Value Type: {param['value_type']}")
    if param['type'] == 'fixed':
        print(f"  Value: {param['value']}")
    elif param['type'] == 'range':
        print(f"  Bounds: {param['bounds']}")
        print(f"  Log Scale: {param.get('log_scale', False)}")
    elif param['type'] == 'choice':
        print(f"  Values: {param['values']}")
```

### Comparing Templates

```python
from positronixlab.config.template import Template

template1 = Template.FromName('mnist_fc')
template2 = Template.FromName('fmnist_fc')

# Convert to dictionaries for comparison
dict1 = template1.to_dict()
dict2 = template2.to_dict()

# Find differences
params1 = {p['name']: p for p in dict1['params']}
params2 = {p['name']: p for p in dict2['params']}

for name in set(params1.keys()) | set(params2.keys()):
    if params1.get(name) != params2.get(name):
        print(f"Difference in {name}:")
        print(f"  Template1: {params1.get(name)}")
        print(f"  Template2: {params2.get(name)}")
```

## Troubleshooting

### Template Not Found Error
```python
from positronixlab.config.template import Template

# Check available templates
print(Template.EXPERIMENTS.keys())

# Use one of the available templates
template = Template.FromName('mnist_fc')
```

### Parameter Not Found
```python
from positronixlab.config.template import Template

template = Template.FromName('mnist_fc')

# Use get_value with default to avoid KeyError
value = template.get_value('non_existent_param', default=None)
if value is None:
    print("Parameter not found, using default")
```

### Type Mismatch Issues
```python
from positronixlab.config.template import Template

template = Template.FromName('mnist_fc')

# Always check parameter type before using
params = template.get_params()
for param in params:
    if param['name'] == 'batch_size':
        assert param['value_type'] == 'int', "batch_size should be int"
        assert param['type'] == 'range', "batch_size should be range type"
```
