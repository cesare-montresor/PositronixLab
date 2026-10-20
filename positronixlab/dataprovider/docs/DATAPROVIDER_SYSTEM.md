# PositronixLab DataProvider System

A unified, modular dataset management system for various machine learning datasets with automatic train/val/test splitting, normalization, and DataLoader creation.

## Features

- **Multi-Dataset Support**: MNIST, Fashion-MNIST, Iris, Covertype, HIGGS datasets
- **Flexible Splits**: Configurable train/val/test split ratios
- **Automatic Normalization**: Optional feature normalization using train statistics
- **PyTorch Integration**: Native DataLoader support with configurable workers
- **Transform Pipeline**: Built-in support for data transforms
- **Factory Pattern**: Simple config-based dataset instantiation via `DataProvider.from_config()`
- **Device Agnostic**: Automatic CUDA detection and device management
- **Metadata**: Built-in dataset information (num_features, num_classes, shapes)

## Architecture

```
┌─────────────────────────────────────────────────────┐
│         DataProvider (Factory Pattern)               │
│     Static method: from_config(config)               │
│     Routes to appropriate dataset based on config    │
└─────────────────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┬────────────────┬──────────────┐
        ▼               ▼               ▼                ▼              ▼
   ┌─────────┐   ┌──────────────┐  ┌─────────┐   ┌──────────────┐  ┌──────────┐
   │ Iris    │   │ MNIST        │  │ FashMNIST   │ Covertype    │  │ HIGGS    │
   │Provider │   │ Provider     │  │ Provider│   │ Provider     │  │ Provider │
   └─────────┘   └──────────────┘  └─────────┘   └──────────────┘  └──────────┘
        │               │               │                │              │
        └───────────────┴───────────────┴────────────────┴──────────────┘
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
   ┌──────────────────┐         ┌──────────────────┐
   │ BaseDataProvider │         │    Datasets      │
   │  (Abstract Base) │         │  (PyTorch)       │
   │                  │         │                  │
   │ - Attributes:    │         │ - Custom split   │
   │   • train, val   │         │ - Normalization  │
   │   • test         │         │ - Transforms     │
   │   • criterion    │         │                  │
   │ - Device mgmt    │         │                  │
   └──────────────────┘         └──────────────────┘
        │                               │
        └───────────────┬───────────────┘
                        ▼
                ┌──────────────────┐
                │  PyTorch         │
                │  DataLoaders     │
                │  & Datasets      │
                └──────────────────┘
```

## Quick Start

### 1. Factory Pattern (Recommended)

```python
from positronixlab.dataprovider import DataProvider
from positronixlab.config import MetaModelConfig

# Create config with dataset specification
config = MetaModelConfig(
    dataset_name='mnist',
    batch_size=32,
    model_type='fc'  # 'fc' for fully connected, auto-flattens images
)

# Create provider in one line
provider = DataProvider.from_config(config)

# Get dataloaders
train_loader = provider.train
val_loader = provider.val
test_loader = provider.test

# Access metadata
print(f"Features: {provider.num_features}")
print(f"Classes: {provider.num_classes}")
print(f"Device: {provider.device}")
print(f"Loss function: {provider.criterion}")
```

### 2. Direct Instantiation

```python
from positronixlab.dataprovider import MnistDataProvider

# Create provider directly
provider = MnistDataProvider(
    batch_size=32,
    split=[0.7, 0.3],      # train/val split
    device='cuda',
    num_workers=4
)

# Use dataloaders
for batch_x, batch_y in provider.train:
    print(batch_x.shape, batch_y.shape)
```

### 3. With Transforms

```python
from positronixlab.dataprovider import MnistDataProvider
from positronixlab.config import MetaTransform

# Define transforms
transforms = MetaTransform(['augment_image', 'flatten'])

provider = MnistDataProvider(
    batch_size=32,
    transforms=transforms
)
```

## Supported Datasets

### Iris Dataset

**Classification**: Multi-class (3 classes)

```
Features: 4 (sepal length/width, petal length/width)
Classes: 3 (Iris setosa, versicolor, virginica)
Samples: 150
Task: Classification
```

**Usage:**
```python
from positronixlab.dataprovider import IrisDataProvider

provider = IrisDataProvider(batch_size=32)
# provider.num_features = 4
# provider.num_classes = 3
# provider.train, provider.val
```

### MNIST Dataset

**Classification**: Multi-class (10 classes)

```
Features: 784 (28x28 grayscale images, flattened)
Classes: 10 (digits 0-9)
Samples: 70,000 (60K train, 10K test)
Image Shape: (1, 28, 28)
```

**Usage:**
```python
from positronixlab.dataprovider import MnistDataProvider

provider = MnistDataProvider(batch_size=32)
# Supports transforms like 'flatten', 'augment_image'
```

### Fashion-MNIST Dataset

**Classification**: Multi-class (10 classes)

```
Features: 784 (28x28 grayscale images, flattened)
Classes: 10 (clothing items: T-shirt, trouser, pullover, etc.)
Samples: 70,000 (60K train, 10K test)
Image Shape: (1, 28, 28)
```

**Usage:**
```python
from positronixlab.dataprovider import FashionMnistDataProvider

provider = FashionMnistDataProvider(batch_size=32)
```

### Covertype Dataset

**Classification**: Multi-class (7 classes)

```
Features: 54 (10 continuous + 44 one-hot encoded categorical)
Classes: 7 (Forest cover types)
Samples: 581,012
Feature Breakdown:
  - 3 continuous: elevation, aspect, slope
  - 4 continuous: distances to hydrology, roadways, fire points
  - 3 continuous: hillshade 9am/12pm/3pm
  - 4 binary: wilderness area
  - 40 binary: soil type
Normalization: Recommended (enabled by default)
```

**Usage:**
```python
from positronixlab.dataprovider import CovertypeDataProvider

provider = CovertypeDataProvider(
    batch_size=64,
    split=[0.7, 0.15, 0.15],  # train/val/test
    normalize=True
)
# provider.num_features = 54
# provider.num_classes = 7
```

### HIGGS Dataset

**Classification**: Binary (signal vs background)

```
Features: 28 (21 low-level physics measurements + 7 high-level derived)
Classes: 2 (signal=1, background=0)
Samples: 11,000,000 (very large)
Feature Types:
  - 21 low-level: raw detector measurements
  - 7 high-level: physics-derived quantities
Normalization: Recommended (enabled by default)
Use Cases:
  - Large-scale binary classification
  - Physics-based feature exploration
  - Benchmarking scalable training
```

**Usage:**
```python
from positronixlab.dataprovider import HIGGSDataProvider

# Use subset for testing
provider = HIGGSDataProvider(
    batch_size=128,
    split=[0.7, 0.15, 0.15],
    normalize=True,
    subset_size=100000  # Use only 100K samples for testing
)

# Or use full dataset
provider = HIGGSDataProvider(batch_size=256)
```

## Key Classes

### BaseDataProvider

Abstract base class for all dataset providers.

**Attributes:**
- `batch_size`: Batch size for DataLoaders
- `dataset_split`: Train/val/test split ratios (default: [0.7, 0.3])
- `num_features`: Number of input features
- `num_classes`: Number of output classes
- `num_workers`: DataLoader worker threads (default: 2)
- `x_shape`: Input tensor shape (e.g., (784,) for MNIST)
- `y_shape`: Output tensor shape
- `device`: PyTorch device (auto-detects CUDA)
- `criterion`: Loss function (typically CrossEntropyLoss)
- `train`: PyTorch DataLoader for training
- `val`: PyTorch DataLoader for validation
- `test`: PyTorch DataLoader for testing (if 3-way split)
- `transforms`: Optional data transforms

**Example:**
```python
provider = MnistDataProvider(batch_size=32, split=[0.6, 0.2, 0.2])

# Access attributes
print(provider.num_features)      # 784
print(provider.num_classes)       # 10
print(provider.batch_size)        # 32
print(provider.device)            # 'cuda' or 'cpu'
print(provider.criterion)         # CrossEntropyLoss()

# Use dataloaders
for x, y in provider.train:
    # x shape: (32, 784)
    # y shape: (32,)
    break
```

### DataProvider (Factory)

Static factory class for creating providers from configuration.

**Methods:**
```python
@staticmethod
def from_config(config: MetaModelConfig, 
                device=None, 
                split=None, 
                num_workers=None) -> BaseDataProvider
```

**Parameters:**
- `config.dataset_name`: Name of dataset ('iris', 'mnist', 'fmnist', 'covertype', 'higgs')
- `config.batch_size`: Batch size for DataLoaders
- `config.model_type`: Model architecture type ('fc' for fully connected)
  - 'fc': Images are flattened for fully connected networks
- `device`: Override device (default: auto-detect)
- `split`: Override split ratios (default: from BaseDataProvider)
- `num_workers`: Override number of workers

**Example:**
```python
from positronixlab.dataprovider import DataProvider
from positronixlab.config import MetaModelConfig

config = MetaModelConfig(
    dataset_name='higgs',
    batch_size=128,
    model_type='fc'
)

provider = DataProvider.from_config(config, device='cuda', num_workers=8)
```

## Data Splits

### Two-Way Split (default)

```python
provider = MnistDataProvider(
    batch_size=32,
    split=[0.7, 0.3]  # train, val
)
# provider.train: 42,000 samples
# provider.val:   18,000 samples
# provider.test:  None
```

### Three-Way Split

```python
provider = CovertypeDataProvider(
    batch_size=64,
    split=[0.7, 0.15, 0.15]  # train, val, test
)
# provider.train: 407,708 samples
# provider.val:   87,151 samples
# provider.test:  87,153 samples
```

## Normalization

Covertype and HIGGS datasets support feature normalization:

```python
from positronixlab.dataprovider import CovertypeDataProvider

provider = CovertypeDataProvider(
    batch_size=64,
    normalize=True  # Enable normalization
)

# Normalization details:
# 1. Compute mean/std from training set
# 2. Apply z-score normalization: (x - mean) / std
# 3. Apply same statistics to val/test sets
# Prevents data leakage and improves convergence
```

## Transforms

Use data transforms for preprocessing and augmentation:

```python
from positronixlab.config import MetaTransform
from positronixlab.dataprovider import MnistDataProvider

# Define transforms
transforms = MetaTransform([
    'augment_image',  # Random rotations, shifts
    'flatten'         # Flatten to 1D vector
])

provider = MnistDataProvider(
    batch_size=32,
    transforms=transforms
)
```

Available transforms:
- `flatten`: Flatten image to 1D vector
- `augment_image`: Apply random augmentations (rotation, shift, etc.)
- Custom transforms can be added via MetaTransform

## Configuration Examples

### MNIST Fully Connected Network

```python
from positronixlab.dataprovider import DataProvider
from positronixlab.config import MetaModelConfig

config = MetaModelConfig(
    dataset_name='mnist',
    batch_size=32,
    model_type='fc'  # Auto-flattens images
)

provider = DataProvider.from_config(config)
# Automatically includes 'flatten' transform for FC models
```

### HIGGS Large-Scale Classification

```python
from positronixlab.dataprovider import HIGGSDataProvider

provider = HIGGSDataProvider(
    batch_size=256,
    split=[0.8, 0.1, 0.1],
    num_workers=8,
    normalize=True,
    device='cuda'
)
```

### Multi-Dataset Experiment

```python
from positronixlab.dataprovider import (
    DataProvider, 
    MnistDataProvider,
    CovertypeDataProvider
)
from positronixlab.config import MetaModelConfig

datasets = []

# Image dataset
config_mnist = MetaModelConfig(
    dataset_name='mnist',
    batch_size=32,
    model_type='fc'
)
datasets.append(DataProvider.from_config(config_mnist))

# Tabular dataset
config_covertype = MetaModelConfig(
    dataset_name='covertype',
    batch_size=64,
    model_type='fc'
)
datasets.append(DataProvider.from_config(config_covertype))

for i, provider in enumerate(datasets):
    print(f"Dataset {i}: {provider.num_features} features, {provider.num_classes} classes")
```

## Device Management

Automatic device detection with override capability:

```python
from positronixlab.dataprovider import MnistDataProvider

# Auto-detect CUDA
provider = MnistDataProvider()
print(provider.device)  # 'cuda' if available, else 'cpu'

# Force CPU
provider = MnistDataProvider(device='cpu')

# Force GPU
provider = MnistDataProvider(device='cuda')
```

## DataLoader Interface

All providers expose PyTorch DataLoaders:

```python
provider = MnistDataProvider(batch_size=32)

# Iterate training data
for batch_x, batch_y in provider.train:
    print(batch_x.shape)  # (32, 784)
    print(batch_y.shape)  # (32,)
    # x: normalized floats [0, 1]
    # y: class indices [0, 9]
    
# Iterate validation data
for batch_x, batch_y in provider.val:
    pass

# Iterate test data (if available)
if provider.test:
    for batch_x, batch_y in provider.test:
        pass
```

## Loss Functions

Each provider includes an appropriate loss function:

```python
provider = MnistDataProvider()
criterion = provider.criterion  # CrossEntropyLoss()

# Used for training
loss = criterion(model_output, batch_y)
```

Supported loss functions:
- **Classification (all datasets)**: `nn.CrossEntropyLoss()`
- Expects model output logits, not probabilities

## Best Practices

1. **Use Factory Pattern**: `DataProvider.from_config()` handles configuration automatically
2. **Set Num Workers**: Use multiple workers for faster data loading
   ```python
   provider = MnistDataProvider(batch_size=32, num_workers=4)
   ```
3. **Enable Normalization**: For tabular datasets (Covertype, HIGGS)
   ```python
   provider = CovertypeDataProvider(normalize=True)
   ```
4. **Match Device**: Ensure data and model are on same device
   ```python
   provider = MnistDataProvider(device='cuda')
   model = model.to(provider.device)
   ```
5. **Use Appropriate Splits**: Three-way splits for final evaluation
   ```python
   provider = MnistDataProvider(split=[0.7, 0.15, 0.15])
   ```

## Common Issues

**Issue**: CUDA Out of Memory
- Solution: Reduce batch_size or num_workers

**Issue**: Slow data loading
- Solution: Increase num_workers (default is 2)

**Issue**: Different train/val/test statistics
- Solution: Enable normalization to standardize across splits
- For Covertype/HIGGS: `normalize=True`

**Issue**: Model expects 1D input but gets 2D (or vice versa)
- Solution: Use transforms or ensure model_type matches
- Use `MetaTransform(['flatten'])` for fully connected networks

## Summary Table

| Dataset | Task | Features | Classes | Samples | Split | Normalize |
|---------|------|----------|---------|---------|-------|-----------|
| Iris | Classification | 4 | 3 | 150 | [0.7, 0.3] | No |
| MNIST | Classification | 784 | 10 | 70K | [0.7, 0.3] | No |
| Fashion-MNIST | Classification | 784 | 10 | 70K | [0.7, 0.3] | No |
| Covertype | Classification | 54 | 7 | 581K | [0.7, 0.15, 0.15] | Yes |
| HIGGS | Binary | 28 | 2 | 11M | [0.7, 0.15, 0.15] | Yes |
