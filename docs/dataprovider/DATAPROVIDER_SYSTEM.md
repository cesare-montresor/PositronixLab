# PositronixLab DataProvider System

A modular dataset loading and batching system built on PyTorch `Dataset` and `DataLoader`, providing composable configuration (`DataConfig`), zero-copy tensor datasets (`BaseTensorDataset`), compositional multi-subtask providers (`CompositionalDataProvider`), lazy `DataReader` creation, sample-index tracking (`IndexedDataset`), and config/name-based routing (`DataProviderFactory`).

`positronixlab/dataprovider/__init__.py` uses PEP 562 lazy `__getattr__` exports so standalone ML and baseline imports never pull in `parallelazy` or heavyweight dependencies eagerly.

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
                │   create() / from_config()           │
                │   Aliases + Lepori family presets    │
                └──────────────────┬───────────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
┌─────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│ Vision Providers│      │ Tabular Providers│      │ Seq / Composit.  │
│ MNIST, FMNIST,  │      │ Iris, Covertype, │      │ SCAN, Math,      │
│ CIFAR-10/100,   │      │ HIGGS (memmap)   │      │ AddMul, TupleAr.,│
│ PermutedMNIST,  │      │                  │      │ OddOneOut (x3)   │
│ ClassRemoval    │      │                  │      │                  │
└────────┬────────┘      └────────┬─────────┘      └────────┬─────────┘
         └────────────────────────┼─────────────────────────┘
                                  ▼
                ┌──────────────────────────────────────┐
                │  DataProvider / CompositionalDP      │
                │  - Auto-registry (__init_subclass__) │
                │  - _split_dataset() (2- or 3-way)    │
                │  - Lazy .train / .val / .test        │
                │  - get_loader() / set_epoch()        │
                └──────────────────┬───────────────────┘
                                   │
                                   ▼
                ┌──────────────────────────────────────┐
                │       DataReader (DataLoader)        │
                │  - CUDA-aware fork/spawn selection   │
                │  - Optional IndexedDataset wrapper   │
                │  - EpochSubsampleSampler support     │
                └──────────────────────────────────────┘
```

---

## Core Components

### `DataConfig` (`positronixlab/dataprovider/config.py`)

Composable dataclass inheriting from `positronixlab.pipeline.config.BaseConfig` (zero `parallelazy` dependency). Also defines `DatasetName` (`Literal` of 28 canonical dataset keys).

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | `"default_config"` | Inherited from `BaseConfig`. |
| `dataset_name` | `Union[DatasetName, str]` | `"fashion_mnist"` | Target dataset key, alias, or family preset. |
| `batch_size` | `int` | `32` | Batch size for `DataReader` loaders. |
| `split` | `Optional[List[float]]` | `None` | Optional 2- or 3-way split ratios; `None` keeps provider's `default_split`. |
| `num_workers` | `Optional[int]` | `None` | DataLoader workers (`None` defaults to `2`). |
| `root` | `Optional[str]` | `None` | Dataset root (`None` defaults to `./data`). |
| `flatten` | `bool` | `False` | Auto-attach `MetaTransform(['flatten'])` for image providers (`supports_flatten`). |
| `indexed_dataset` | `bool` | `False` | Wrap datasets in `IndexedDataset` to yield `(x, y, idx)`. |
| `shuffle_train` | `bool` | `True` | Shuffle the training loader by default. |
| `drop_last` | `bool` | `True` | Drop incomplete trailing batches. |
| `persistent_workers` | `bool` | `True` | Keep worker processes alive when `num_workers > 0`. |
| `pin_memory` | `bool` | `False` | Pin host memory for faster GPU transfers. |
| `seed` | `int` | `42` | Deterministic split and sampling seed. |
| `dataset_kwargs` | `Dict[str, Any]` | `{}` | Extra provider-specific keyword arguments. |

Supports `to_dict()`, `from_dict()`, `with_updates()`, `signature()`, and JSON/YAML `save()`/`load()`. Passing a bare `DataConfig` directly to `DataProviderFactory.from_config()` works out of the box.

### Core Classes & Helpers (`positronixlab/dataprovider/dataprovider.py`)

- **`resolve_data_root(root=None) -> Path`**: Returns `Path(root)` if provided, otherwise `./data`.
- **`apply_transform(transforms, x)`**: Applies either callable transforms (`transforms(x)`) or legacy `.apply(x)` (`MetaTransform`).
- **`BaseTensorDataset`**: Zero-copy tensor-backed `Dataset` with `compute_normalization()`, `apply_normalization()`, `copy()`, and `random_split()`.
- **`IndexedDataset`**: Wraps any `Dataset` yielding `(x, y)` so `__getitem__(idx)` returns `(x, y, idx)`.
- **`EpochSubsampleSampler(pool_size, epoch_size, *, base_seed=42, shuffle=True, sort_for_locality=False)`**: Deterministic per-epoch subsampling via `set_epoch(epoch)`, compatible with `persistent_workers=True`.
- **`DataReader(DataLoader)`**:
  - Subclass registry via `__init_subclass__(plugin_name=None, override=False)`, `get(name)`, `list_registered()`.
  - CUDA-safe multiprocessing: selects `"fork"` when `not torch.cuda.is_initialized()`, else `"spawn"`.
  - Automatically sets `persistent_workers=False` and `multiprocessing_context=None` when `num_workers == 0`.
- **`DataProvider`** (with `DataProviderOptions` `TypedDict`):
  - Subclass registry via `__init_subclass__(plugin_name=None, aliases=(), override=False, register=True)`, `get(name)`, `list_registered()`.
  - Class attributes: `default_split = (0.7, 0.3)`, `supports_flatten = False`.
  - `__init__(batch_size=32, split=None, num_workers=None, transforms=None, indexed_dataset=False, root=None, seed=42, shuffle_train=True, drop_last=True, persistent_workers=True, pin_memory=False)`.
  - `_split_dataset(dataset, *, test_dataset=None, split=None, seed=None)`: Handles both 2-way and 3-way splits cleanly.
  - Lazy `.train` (shuffled when `shuffle_train=True`), `.val`, `.test` properties, plus `get_loader(split, ...)` and `set_epoch(epoch)`.
  - Optional `self.criterion: Optional[nn.Module] = None`.
- **`SubtaskSplit`** & **`CompositionalDataProvider(DataProvider, register=False)`**:
  - Multi-subtask contract exposing `subtasks`, `list_subtasks()`, `get_subtask(name)`, `eval_matrix_spec()`, `base_split`, and `{train,val,test}_loader()`.

### `DataProviderFactory` (`positronixlab/dataprovider/dataprovider_factory.py`)

- `DataProviderFactory.list_available() -> list[str]`: Lists all registered keys, aliases, and family presets.
- `DataProviderFactory.create(dataset_name, *, flatten=False, **kwargs) -> DataProvider`: Direct name-based instantiation.
- `DataProviderFactory.from_config(config, split=None, num_workers=None, indexed_dataset=None, **extra_kwargs) -> DataProvider`: Works with both `DataConfig` and `PipelineConfig` (sets `flatten=True` when `config.flatten` or `config.model_type == 'fc'`; preserves provider `default_split` when `split=None`).

---

## Supported Datasets

All 15 concrete providers and their dataset classes are lazily exported from `positronixlab.dataprovider`:

| Factory Key / Aliases | Provider Class | Input Shape | Classes | Default Split | Data Source |
|---|---|---|---|---|---|
| `'iris'` | `IrisDataProvider` | `(4,)` | 3 | `(0.7, 0.3)` | `sklearn.datasets.load_iris()` (offline) |
| `'mnist'` | `MnistDataProvider` | `(1, 28, 28)` | 10 | `(0.7, 0.3)` + test | `torchvision.datasets.MNIST` |
| `'fashion_mnist'`, `'fmnist'` | `FashionMnistDataProvider` | `(1, 28, 28)` | 10 | `(0.7, 0.3)` + test | `torchvision.datasets.FashionMNIST` |
| `'cifar10'` | `CIFAR10DataProvider` | `(3, 32, 32)` | 10 | `(0.7, 0.3)` + test | `torchvision.datasets.CIFAR10` |
| `'cifar100'` | `CIFAR100DataProvider` | `(3, 32, 32)` | 100 | `(0.7, 0.3)` + test | `torchvision.datasets.CIFAR100` |
| `'permuted_mnist'`, `'csordas:permuted_mnist'` | `PermutedMnistDataProvider` | `(1, 28, 28)` | 10 | `(0.7, 0.3)` + test | Permuted MNIST (pixel permutations) |
| `'class_removal'`, `'csordas:class_removal'` | `ClassRemovalDataProvider` | Wraps base | Base | Wraps base | Compositional wrapper (`from_provider()`) removing/isolating classes |
| `'covertype'` | `CovertypeDataProvider` | `(54,)` | 7 | `(0.7, 0.15, 0.15)` | `sklearn.datasets.fetch_covtype` |
| `'higgs'` | `HIGGSDataProvider` | `(28,)` | 2 | `(0.7, 0.15, 0.15)` | Memmapped `<root>/HIGGS.npy` (UCI) |
| `'scan'`, `'csordas:scan'` | `SCANDataProvider` | `(16,)` | 10 (action vocab) | `(0.7, 0.3)` + test | GitHub `brendenlake/SCAN` auto-download → `<root>/scan/{raw,cache}/` (or legacy `<root>/scan/{train,test}.pt`) |
| `'deepmind_math'` | `MathDataProvider` | `(160,)` | 99 (char vocab) | `(0.7, 0.3)` + eval | HF `cesare-montresor/DeepMind-Mathematics-Dataset` auto-download → `<root>/math/{raw,cache}/` (or legacy `<root>/math/{train,test}.pt`) |
| `'addmul'`, `'csordas:add_mul'` | `AddMulDataProvider` | Compositional | Task-dep. | `(0.7, 0.3)` + test | Synthetic Add/Mul arithmetic (`CompositionalDataProvider`) |
| `'tuple_arithmetic'`, `'csordas:tuple_arithmetic'` | `TupleArithmeticDataProvider` | Compositional | Task-dep. | `(0.7, 0.3)` + test | Synthetic tuple arithmetic (`CompositionalDataProvider`) |
| `'cvr_odd_one_out'`, `'lepori:cvr_*'` | `CVROddOneOutDataProvider` | `(4, 3, H, W)` | 4 | Compositional | Procedural CVR Odd-One-Out (`inside_contact`, `inside_count`, `contact_count`) |
| `'syneval_odd_one_out'`, `'lepori:syneval_*'` | `SynevalOddOneOutDataProvider` | `(2, 4, max_len)` | 4 | Compositional | SynEval Odd-One-Out (`sv_{singular,plural}`, `pro_ant_{singular,plural}`) |

---

## HIGGS Large-Dataset Design (`higgs_dataprovider.py`)

`HIGGSDataProvider` is designed for the 11M-sample UCI HIGGS dataset:
1. **Single shared memmap**: `_load_or_build_npy(root)` opens `HIGGS.npy` with `np.load(..., mmap_mode='r')` once; train/val/test `HIGGSDataset` views reference index arrays over the shared tensor without duplicating memory.
2. **Three independent seeds**: `test_seed` (`1234`), `split_seed` (`42`), and `epoch_seed` (`42`) driving `EpochSubsampleSampler` via `provider.set_epoch(epoch)` (works cleanly with `persistent_workers=True`).
3. **Train-only normalization**: Samples up to 100,000 rows from train and applies `(x - mean) / (std + 1e-8)` lazily in `__getitem__`.

---

## Operational Notes

- Small validation/test splits (`len(split) < batch_size`) yield zero batches when `drop_last=True`; pass `drop_last=False` or use `provider.get_loader("val", drop_last=False)`.
- See [`reports/dataprovider_analysis.md`](file:///home/cesare/Projects/PositronixLab/reports/dataprovider_analysis.md) for historical audit notes.
