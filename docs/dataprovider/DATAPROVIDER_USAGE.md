# PositronixLab DataProvider Usage Guide

Practical examples for configuring, loading, and extending datasets in `positronixlab.dataprovider`.

---

## 1. Using `DataConfig`

`DataConfig` is a lightweight, `BaseConfig`-backed dataclass for dataset hyperparameters:

```python
from positronixlab.dataprovider.config import DataConfig

cfg = DataConfig(dataset_name="iris", batch_size=16)
print(cfg.to_dict())
print(cfg.signature())

# Immutable update
cfg_large = cfg.with_updates(batch_size=64)

# Save / reload JSON or YAML
cfg_large.save("/tmp/data_config.yaml")
reloaded = DataConfig.load("/tmp/data_config.yaml")
```

---

## 2. Direct Provider Instantiation

For CPU/offline work or when you need provider-specific arguments (`root`, `normalize`, `epoch_size`, etc.), instantiate the provider directly:

```python
from positronixlab.dataprovider import IrisDataProvider

provider = IrisDataProvider(batch_size=16, num_workers=0)

print(provider.num_features, provider.num_classes)  # 4 3
for x, y in provider.train:
    print(x.shape, y.shape)  # torch.Size([16, 4]) torch.Size([16])
    break
```

For vision datasets (downloads to `./data` on first run):

```python
from positronixlab.dataprovider import FashionMnistDataProvider
from positronixlab.metamodel.metatransform import MetaTransform

provider = FashionMnistDataProvider(
    batch_size=32,
    split=[0.8, 0.2],
    num_workers=2,
    transforms=MetaTransform(["flatten"]),
)

for x, y in provider.train:
    print(x.shape, y.shape)  # torch.Size([32, 784]) torch.Size([32])
    break
```

> [!TIP]
> Pass `num_workers=0` in unit tests, scripts on small datasets, or when debugging to avoid multiprocessing overhead.

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

You can also wrap any PyTorch `Dataset` directly:

```python
from torch.utils.data import TensorDataset
import torch
from positronixlab.dataprovider import IndexedDataset, DataReader

raw = TensorDataset(torch.randn(64, 8), torch.randint(0, 3, (64,)))
reader = DataReader(raw, batch_size=16, num_workers=0, indexed_dataset=True)
x, y, idx = next(iter(reader))
```

---

## 4. Factory Instantiation (`DataProviderFactory`)

`DataProviderFactory.from_config()` routes by `config.dataset_name`, `config.batch_size`, and `config.model_type` (`'fc'` auto-attaches `MetaTransform(['flatten'])` for image datasets):

```python
from types import SimpleNamespace
from positronixlab.dataprovider import DataProviderFactory

# Works with PipelineConfig (or any object exposing dataset_name, batch_size, model_type)
cfg = SimpleNamespace(dataset_name="iris", batch_size=16, model_type="fc")
provider = DataProviderFactory.from_config(cfg, num_workers=0)
```

> [!IMPORTANT]
> When using `DataProviderFactory.from_config()` with `'covertype'` or `'higgs'`, pass `split=[0.7, 0.15, 0.15]` explicitly if you want a test split; leaving `split=None` overrides their 3-way default with `[0.7, 0.3]`.

---

## 5. Working with `HIGGSDataProvider`

`HIGGSDataProvider` memmaps `<root>/HIGGS.npy` once and supports per-epoch subsampling via `set_epoch(epoch)`:

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
    num_workers=0,        # recommended when calling set_epoch() (see caveat below)
)

for epoch in range(3):
    provider.set_epoch(epoch)  # resamples train & val; test stays fixed
    for x, y in provider.train:
        ...
```

> [!WARNING]
> With `num_workers > 0`, `DataReader` defaults to `persistent_workers=True`, so worker processes do not see subsequent `provider.set_epoch(epoch)` updates. Use `num_workers=0` when per-epoch resampling is required.

---

## 6. Writing a Custom `DataProvider` Subclass

Subclasses of `DataProvider` automatically register in `DataProvider._registry` (choose a unique class name or pass `plugin_name=...`):

```python
import torch
from torch.utils.data import TensorDataset, random_split
from positronixlab.dataprovider import DataProvider


class SyntheticDataProvider(DataProvider, plugin_name="synthetic_demo"):
    def __init__(self, n_samples: int = 200, n_features: int = 8, n_classes: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.name = "synthetic_demo"
        self.num_features = n_features
        self.num_classes = n_classes
        self.x_shape = (n_features,)
        self.y_shape = ()

        g = torch.Generator().manual_seed(0)
        x = torch.randn(n_samples, n_features, generator=g)
        y = torch.randint(0, n_classes, (n_samples,), generator=g)
        self.dataset = TensorDataset(x, y)

        splits = random_split(self.dataset, self.dataset_split, generator=g)
        self.dataset_train = splits[0]
        self.dataset_val = splits[1]
        self.dataset_test = splits[2] if len(splits) > 2 else None


provider = SyntheticDataProvider(batch_size=16, split=[0.7, 0.15, 0.15], num_workers=0)
print(len(provider.train), len(provider.val), len(provider.test))
```

---

## 7. Running Tests & Examples

```bash
# Run offline core unit tests
.venv/bin/pytest positronixlab/dataprovider/tests/test_dataprovider_core.py -v

# Run offline CPU example
.venv/bin/python positronixlab/dataprovider/examples/example_dataprovider.py
```
