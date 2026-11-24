# PositronixLab DataProvider System

A modular dataset loading and batching system built on PyTorch `Dataset` and `DataLoader`, providing composable configuration (`DataConfig`), lazy `DataReader` creation, sample-index tracking (`IndexedDataset`), and config-based routing (`DataProviderFactory`).

---

## Architecture

```
                ┌──────────────────────────────────────┐
                │   DataConfig / PipelineConfig        │
                │   (dataset_name, batch_size, ...)    │
                └──────────────────┬───────────────────┘
                                   │
                                   ▼
                ┌──────────────────────────────────────┐
                │         DataProviderFactory          │
                │   from_config(config, split, ...)    │
                │   Routes dataset_name -> provider    │
                └──────────────────┬───────────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
┌─────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│ Vision Providers│      │ Tabular Providers│      │ Sequence / Rel.  │
│ MNIST, FMNIST,  │      │ Iris, Covertype, │      │ SCAN, Math,      │
│ CIFAR-10/100    │      │ HIGGS (memmap)   │      │ OddOneOut (x3)   │
└────────┬────────┘      └────────┬─────────┘      └────────┬─────────┘
         └────────────────────────┼─────────────────────────┘
                                  ▼
                ┌──────────────────────────────────────┐
                │            DataProvider              │
                │  - Auto-registry (__init_subclass__) │
                │  - dataset_{train,val,test}          │
                │  - Lazy .train / .val / .test        │
                └──────────────────┬───────────────────┘
                                   │
                                   ▼
                ┌──────────────────────────────────────┐
                │       DataReader (DataLoader)        │
                │  - CUDA-aware fork/spawn selection   │
                │  - Optional IndexedDataset wrapper   │
                └──────────────────────────────────────┘
```

---

## Core Components

### `DataConfig` (`positronixlab/dataprovider/config.py`)

Composable dataclass inheriting from `positronixlab.pipeline.config.BaseConfig` (zero `parallelazy` dependency):

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | `"default_config"` | Inherited from `BaseConfig`. |
| `dataset_name` | `str` | `"fashion_mnist"` | Target dataset identifier. |
| `batch_size` | `int` | `32` | Batch size for `DataReader` loaders. |

Supports `to_dict()`, `from_dict()`, `with_updates()`, `signature()`, and JSON/YAML `save()`/`load()`.

> [!NOTE]
> `DataProviderFactory.from_config()` currently still reads `config.model_type` (from `PipelineConfig`) to decide whether to attach `MetaTransform(['flatten'])` for `'fc'` models. Passing a bare `DataConfig` directly raises `AttributeError` until the factory is updated.

### `IndexedDataset` (`positronixlab/dataprovider/dataprovider.py`)

Wraps any PyTorch `Dataset` yielding `(x, y)` so `__getitem__(idx)` returns `(x, y, idx)`. Used by `DeeplearningBase`/`DeeplearningTrain` when per-sample indices are needed (e.g., for tracking activations or clustering).

### `DataReader` (`positronixlab/dataprovider/dataprovider.py`)

Subclass of `torch.utils.data.DataLoader` with:
- **Subclass registry**: `_registry: dict[str, type[Self]]` populated via `__init_subclass__(plugin_name=...)`. Duplicate names raise `ValueError`.
- **CUDA-safe multiprocessing**: When `multiprocessing_context is None` and `num_workers > 0`, selects `"fork"` if `not torch.cuda.is_initialized()`, otherwise `"spawn"`.
- **Worker cleanup**: When `num_workers == 0`, forces `persistent_workers=False` and `multiprocessing_context=None`.
- **Defaults**: `batch_size=32`, `num_workers=2`, `persistent_workers=True`, `drop_last=True`, `indexed_dataset=False`.

### `DataProvider` (`positronixlab/dataprovider/dataprovider.py`)

Base class for all concrete dataset providers:
- **Subclass registry**: `_registry` populated via `__init_subclass__(plugin_name=...)`.
- **Init parameters**: `batch_size=32`, `split=None` (defaults to `[0.7, 0.3]`), `num_workers=None` (defaults to `2`), `transforms=None`, `indexed_dataset=False`.
- **Metadata attributes**: `num_features`, `num_classes`, `x_shape`, `y_shape`, `dataset`, `dataset_train`, `dataset_val`, `dataset_test`.
- **Lazy loader properties**:
  - `.train` -> `DataReader(self.dataset_train, ...)` (or `None` if `dataset_train is None`)
  - `.val` -> `DataReader(self.dataset_val, ...)` (or `None` if `dataset_val is None`)
  - `.test` -> `DataReader(self.dataset_test, ...)` (or `None` if `dataset_test is None`)

### `DataProviderFactory` (`positronixlab/dataprovider/dataprovider_factory.py`)

Static router `DataProviderFactory.from_config(config, split=None, num_workers=None, indexed_dataset=False) -> DataProvider`.

For image datasets (`mnist`, `fashion_mnist`, `cifar10`, `cifar100`, `cvr_odd_one_out`), constructs a `MetaTransform(['flatten'])` when `config.model_type == 'fc'` (or `MetaTransform([])` otherwise).

---

## Supported Datasets

| Factory Key | Provider Class | Exported in `__init__` | Features / Input Shape | Classes | Default Split | Data Source |
|---|---|---|---|---|---|---|
| `'iris'` | `IrisDataProvider` | Yes | 4 (`(4,)`) | 3 | Unsplit (all 150) | `sklearn.datasets.load_iris()` (offline) |
| `'mnist'` | `MnistDataProvider` | Yes | 784 (`(28, 28)`) | 10 | `[0.7, 0.3]` + test | `torchvision.datasets.MNIST('./data')` |
| `'fashion_mnist'` | `FashionMnistDataProvider` | Yes | 784 (`(28, 28)`) | 10 | `[0.7, 0.3]` + test | `torchvision.datasets.FashionMNIST('./data')` |
| `'cifar10'` | `CIFAR10DataProvider` | No | 3072 (`(3, 32, 32)`) | 10 | `[0.7, 0.3]` + test | `torchvision.datasets.CIFAR10('./data')` |
| `'cifar100'` | `CIFAR100DataProvider` | No | 3072 (`(3, 32, 32)`) | 100 | `[0.7, 0.3]` + test | `torchvision.datasets.CIFAR100('./data')` |
| `'covertype'` | `CovertypeDataProvider` | Yes | 54 (`(54,)`) | 7 | `[0.7, 0.15, 0.15]` | `sklearn.datasets.fetch_covtype(root)` |
| `'higgs'` | `HIGGSDataProvider` | Yes | 28 (`(28,)`) | 2 | `[0.7, 0.15, 0.15]` | Memmapped `<root>/HIGGS.npy` (UCI download) |
| `'scan'` | `SCANDataProvider` | No | `None` (`(20,)` dummy) | `None` | `[0.7, 0.3]` + test | `./data/scan/{train,test}.pt` (dummy fallback) |
| `'deepmind_math'` | `MathDataProvider` | No | `None` (`(50,)` dummy) | `None` | `[0.7, 0.3]` + test | `./data/math/{train,test}.pt` (dummy fallback) |
| `'cvr_odd_one_out'` | `CVROddOneOutDataProvider` | No | `None` (`(4, 3, 64, 64)`) | 4 | `[0.7, 0.3]` + test | `./data/cvr_odd_one_out/{train,test}.pt` (dummy fallback) |
| `'syneval_odd_one_out'` | `SynevalOddOneOutDataProvider` | No | `None` (`(4, 32)` dummy) | 4 | `[0.7, 0.3]` + test | `./data/syneval_odd_one_out/{train,test}.pt` (dummy fallback) |
| *(none)* | `OddOneOutDataProvider` | No | `None` (`(4, 3, 64, 64)`) | 4 | `[0.7, 0.3]` + test | `./data/odd_one_out/{train,test}.pt` (dummy fallback) |

---

## HIGGS Large-Dataset Design (`higgs_dataprovider.py`)

`HIGGSDataProvider` is designed for the 11M-sample UCI HIGGS dataset:
1. **Single shared memmap**: `_load_or_build_npy(root)` opens `HIGGS.npy` with `np.load(..., mmap_mode='r')` once; train/val/test `HIGGSDataset` views reference index arrays over the shared tensor without duplicating the full array.
2. **Three independent seeds**:
   - `test_seed` (default `1234`): carves the test split out of the full dataset first so test membership stays fixed across train/val ablations.
   - `split_seed` (default `42`): partitions the remainder into train and val.
   - `epoch_seed` (default `42`): drives deterministic per-epoch subsampling (`epoch_size` as float fraction `(0.0, 1.0]` or int count) via `HIGGSDataProvider.set_epoch(epoch)`.
3. **Train-only normalization**: `compute_normalization()` samples up to 100,000 rows from train and applies `(x - mean) / (std + 1e-8)` lazily in `__getitem__` for train, val, and test.

---

## Known Limitations & Caveats

See [`analysis.md`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/analysis.md) for the full defect list. Key operational caveats:
- **Transitive `parallelazy` import**: Importing `positronixlab.dataprovider` eagerly imports `DataProviderFactory` -> `PipelineConfig` -> `parallelazy`.
- **`DataReader` defaults**: `drop_last=True` and `shuffle=False` apply to `train`, `val`, and `test`. Small validation splits (`len(val) < batch_size`) yield zero batches.
- **`HIGGSDataProvider.set_epoch()` + `persistent_workers=True`**: With `num_workers > 0`, persistent DataLoader workers do not observe main-process `set_epoch()` updates; use `num_workers=0` if per-epoch resampling is required.
- **`IrisDataProvider` leakage**: `dataset_train`, `dataset_val`, and `dataset_test` all reference the same unsplit 150-sample `IrisDataset`.
- **2-way split unpacking**: Vision and `.pt` providers unpack `train, val = dataset.random_split(split)` and fail on 3-element `split` lists.
- **`transforms` interface**: Most datasets call `transforms.apply(x)` (`MetaTransform`) rather than `transforms(x)` (`torchvision.transforms.Compose`).
