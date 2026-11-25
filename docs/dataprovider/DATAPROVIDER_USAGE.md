# PositronixLab DataProvider Usage Guide

Practical examples for configuring, loading, and extending datasets in `positronixlab.dataprovider`.

---

## 1. Using `DataConfig`

`DataConfig` is a lightweight, `BaseConfig`-backed dataclass for dataset hyperparameters:

```python
from positronixlab.dataprovider import DataConfig, DataProviderFactory

cfg = DataConfig(dataset_name="iris", batch_size=16, num_workers=0)
print(cfg.to_dict())
print(cfg.signature())

# Immutable update
cfg_large = cfg.with_updates(batch_size=64)

# Save / reload JSON or YAML
cfg_large.save("/tmp/data_config.yaml")
reloaded = DataConfig.load("/tmp/data_config.yaml")

# Instantiate directly from DataConfig
provider = DataProviderFactory.from_config(cfg)
```

---

## 2. Direct Provider Instantiation & `DataProviderFactory`

```python
from positronixlab.dataprovider import IrisDataProvider, DataProviderFactory

# Direct instantiation
provider = IrisDataProvider(batch_size=16, num_workers=0)
print(provider.num_features, provider.num_classes)  # 4 3
for x, y in provider.train:
    print(x.shape, y.shape)  # torch.Size([16, 4]) torch.Size([16])
    break

# Name-based factory creation (supports aliases and Lepori family presets)
print(DataProviderFactory.list_available())
fmnist = DataProviderFactory.create("fmnist", batch_size=32, flatten=True, num_workers=0)
cvr = DataProviderFactory.create("lepori:cvr_inside_contact", batch_size=16, num_workers=0)
```

> [!TIP]
> Pass `num_workers=0` in unit tests, scripts on small datasets, or when debugging to avoid multiprocessing overhead. Use `POSITRONIXLAB_DATA_DIR` or `root=...` to override the default `./data` directory.

### Custom Loader Overrides (`get_loader`)

```python
# Override drop_last, shuffle, or batch_size on the fly
val_loader = provider.get_loader("val", batch_size=32, drop_last=False, shuffle=False)
```

---

## 3. Tracking Sample Indices with `IndexedDataset`

Set `indexed_dataset=True` on any `DataProvider` (or `DataReader`) to receive `(x, y, idx)` tuples — compatible with `DeeplearningBase.process_batch`:

```python
from positronixlab.dataprovider import IrisDataProvider

provider = IrisDataProvider(batch_size=16, num_workers=0, indexed_dataset=True)
for x, y, idx in provider.train:
    print(x.shape, y.shape, idx.shape)  # (16, 4), (16,), (16,)
    break
```

---

## 4. Compositional Datasets & `ClassRemovalDataProvider`

Compositional providers (`AddMulDataProvider`, `TupleArithmeticDataProvider`, `CVROddOneOutDataProvider`, `SynevalOddOneOutDataProvider`, `ClassRemovalDataProvider`) expose named subtasks:

```python
from positronixlab.dataprovider import (
    AddMulDataProvider,
    ClassRemovalDataProvider,
    IrisDataProvider,
)

# Built-in compositional task
addmul = AddMulDataProvider(batch_size=32, num_workers=0)
print(addmul.list_subtasks())
add_loader = addmul.get_subtask("add").train_loader()

# Wrap any classification provider into a class-removal compositional benchmark
base = IrisDataProvider(batch_size=16, num_workers=0)
cr = ClassRemovalDataProvider.from_provider(base, target_classes=[0, 1])
print(cr.list_subtasks())
```

---

## 5. Working with `HIGGSDataProvider`

`HIGGSDataProvider` memmaps `<root>/HIGGS.npy` once and uses `EpochSubsampleSampler` for deterministic per-epoch subsampling (compatible with `persistent_workers=True`):

```python
from positronixlab.dataprovider import HIGGSDataProvider

provider = HIGGSDataProvider(
    root="./data",
    batch_size=256,
    split=[0.7, 0.15, 0.15],
    epoch_size=0.05,      # 5% of each split per epoch (or int > 1 for train count)
    normalize=True,
    split_seed=42,
    test_seed=1234,
    epoch_seed=42,
    num_workers=2,
)

for epoch in range(3):
    provider.set_epoch(epoch)  # resamples train & val; test stays fixed
    for x, y in provider.train:
        ...
```

---

## 6. Writing a Custom `DataProvider` Subclass

Subclasses of `DataProvider` automatically register in `DataProvider._registry` (choose a unique class name or pass `plugin_name=...` / `aliases=...`):

```python
import torch
from torch.utils.data import TensorDataset
from positronixlab.dataprovider import DataProvider


class SyntheticDataProvider(DataProvider, plugin_name="synthetic_demo"):
    def __init__(self, n_samples: int = 200, n_features: int = 8, n_classes: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.name = "synthetic_demo"
        self.num_features = n_features
        self.num_classes = n_classes
        self.x_shape = (n_features,)
        self.y_shape = ()

        g = torch.Generator().manual_seed(self.seed)
        x = torch.randn(n_samples, n_features, generator=g)
        y = torch.randint(0, n_classes, (n_samples,), generator=g)
        self.dataset = TensorDataset(x, y)
        self.dataset_train, self.dataset_val, self.dataset_test = self._split_dataset(self.dataset)


provider = SyntheticDataProvider(batch_size=16, split=[0.7, 0.15, 0.15], num_workers=0)
print(len(provider.train), len(provider.val), len(provider.test))
```

---

## 7. Running Tests & Examples

```bash
# Run offline core unit tests
.venv/bin/pytest tests/dataprovider/test_dataprovider_core.py -v

# Run offline CPU example
.venv/bin/python examples/dataprovider/example_dataprovider.py
```
