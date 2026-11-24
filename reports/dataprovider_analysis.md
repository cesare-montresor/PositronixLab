# `positronixlab/dataprovider` — Module Analysis

## 1. Overview & Refactor Status

`positronixlab/dataprovider` provides PyTorch dataset wrappers, `DataLoader` management (`DataReader`), configuration (`DataConfig`), and a factory (`DataProviderFactory`) for feeding classification, sequence, and relational datasets into `positronixlab` models and trainers.

The module is currently **mid-refactor** across three overlapping generations of code:

| File | Role | Status |
|---|---|---|
| [`__init__.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/__init__.py) | Eagerly exports `DataConfig`, `DataProvider`, `DataReader`, `IndexedDataset`, `DataProviderFactory`, and 5 of 12 concrete providers. | Partial / Coupled |
| [`config.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/config.py) | `DataConfig(BaseConfig)`: new composable dataset config (`dataset_name="fashion_mnist"`, `batch_size=32`). | Active (not yet wired into factory) |
| [`dataprovider.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/dataprovider.py) | Core abstractions: `IndexedDataset`, `DataReader(DataLoader)` (CUDA-aware fork/spawn, auto-registering), `DataProvider` (lazy `train`/`val`/`test` properties, auto-registering). | Active |
| [`dataprovider_factory.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/dataprovider_factory.py) | `DataProviderFactory.from_config(config: PipelineConfig, ...)`: hardcoded 11-branch `if/elif` router + `MetaTransform` flattening for `'fc'` models. | Transitional (coupled to `PipelineConfig`) |
| [`higgs_dataprovider.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/higgs_dataprovider.py) | `HIGGSDataProvider` & `HIGGSDataset`: recently rewritten with single-load `np.load(..., mmap_mode='r')`, 3-seed split (`test_seed`/`split_seed`/`epoch_seed`), and `set_epoch()` resampling. | Recently refactored (some bugs + stale test suite) |
| [`covertype_dataprovider.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/covertype_dataprovider.py) | `CovertypeDataProvider` & `CovertypeDataset`: sklearn `fetch_covtype`, 54 features, 7 classes, continuous-feature z-score normalization. | Active (inefficient `copy()`) |
| [`mnist_dataprovider.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/mnist_dataprovider.py), [`fmnist_dataprovider.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/fmnist_dataprovider.py), [`cifar10_dataprovider.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/cifar10_dataprovider.py), [`cifar100_dataprovider.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/cifar100_dataprovider.py) | Torchvision image providers (`download=True` into hardcoded `./data`). | Active |
| [`iris_dataprovider.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/iris_dataprovider.py) | `IrisDataProvider` & `IrisDataset`: sklearn `load_iris()` (offline, 150 samples). | Active (severe data-leakage bug) |
| [`scan_dataprovider.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/scan_dataprovider.py), [`math_dataprovider.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/math_dataprovider.py), [`oddoneout_dataprovider.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/oddoneout_dataprovider.py), [`cvr_oddoneout_dataprovider.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/cvr_oddoneout_dataprovider.py), [`syneval_oddoneout_dataprovider.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/syneval_oddoneout_dataprovider.py) | Preprocessed `.pt` tensor providers with silent dummy-tensor fallbacks when `./data/<name>/{train,test}.pt` is absent. | Stub / WIP |

---

## 2. Bugs & Defects

### B1. Eager transitive `parallelazy` import breaks standalone module import (High)
- **Location:** [`__init__.py:3`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/__init__.py#L3), [`dataprovider_factory.py:1`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/dataprovider_factory.py#L1)
- **Root Cause:** `__init__.py` eagerly imports `DataProviderFactory`, which eagerly executes `from positronixlab.pipeline import PipelineConfig` purely for a type annotation. `PipelineConfig` lives in `pipeline_config.py`, which eagerly imports `from parallelazy.plugin_data import ManagerOptions`.
- **Impact:** Importing *any* symbol or submodule under `positronixlab.dataprovider` (including `DataConfig`, `DataReader`, `IrisDataProvider`, or `tests/test_higgs.py`) fails with `ModuleNotFoundError: No module named 'parallelazy'` in environments without `parallelazy` (and `ImportError` on `parallelazy 0.0.6`).

### B2. `DataProviderFactory.from_config()` crashes when passed `DataConfig` (High)
- **Location:** [`config.py:9-13`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/config.py#L9-L13), [`dataprovider_factory.py:22-25`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/dataprovider_factory.py#L22-L25)
- **Root Cause:** `DataConfig` is documented as `"Composable dataset config for DataProviderFactory"`, yet `DataProviderFactory.from_config()` unconditionally reads `config.model_type` (`line 25`). `DataConfig` only defines `dataset_name` and `batch_size`.
- **Impact:** Passing `DataConfig` directly to `DataProviderFactory.from_config()` raises `AttributeError: 'DataConfig' object has no attribute 'model_type'`.

### B3. `DataProviderFactory.from_config(..., split=None)` silently drops the test split on `HIGGS` and `Covertype` (High)
- **Location:** [`dataprovider_factory.py:22,29`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/dataprovider_factory.py#L22-L29), [`covertype_dataprovider.py:34,39`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/covertype_dataprovider.py#L34-L39), [`higgs_dataprovider.py:108,113`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/higgs_dataprovider.py#L108-L113), [`dataprovider.py:84`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/dataprovider.py#L84)
- **Root Cause:** `DataProviderFactory.from_config()` always forwards `"split": None` in `kwargs` when `split` is omitted. This overrides the `split=[0.7, 0.15, 0.15]` default on `CovertypeDataProvider` and `HIGGSDataProvider`, causing `DataProvider.__init__` to replace `None` with `[0.7, 0.3]` (a 2-way split).
- **Impact:** Creating `covertype` or `higgs` via `DataProviderFactory` silently produces `provider.dataset_test is None` and `provider.test is None`. (Both classes also use mutable list defaults `split=[0.7, 0.15, 0.15]`.)

### B4. `HIGGSDataProvider.set_epoch()` is silently defeated by `DataReader(persistent_workers=True)` (High)
- **Location:** [`higgs_dataprovider.py:215-235,370-382`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/higgs_dataprovider.py#L215-L235), [`dataprovider.py:40-41,109-121`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/dataprovider.py#L40-L41)
- **Root Cause:** `DataReader` defaults to `num_workers=2, persistent_workers=True`. With persistent workers, each worker process holds its own snapshot of `HIGGSDataset` after worker startup; calling `provider.set_epoch(e)` in the main process mutates `self.x`/`self.y` only in the main process and is never propagated to already-running workers.
- **Impact:** Under default `DataProvider` settings (`num_workers=2`), `HIGGSDataProvider.set_epoch()` has no effect after epoch 0 — reintroducing the exact fixed-subsample bug the rewrite aimed to fix.

### B5. `_load_or_build_npy()` crashes with `FileNotFoundError` when converting `HIGGS.csv` -> `HIGGS.npy` (High)
- **Location:** [`higgs_dataprovider.py:270-272`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/higgs_dataprovider.py#L270-L272)
- **Root Cause:** `tmp_npy = npy_file.with_suffix('.npy.tmp')` followed by `np.save(tmp_npy, data)` and `tmp_npy.rename(npy_file)`. Because `tmp_npy` does not end in `.npy`, `numpy.save` automatically appends `.npy` and writes `HIGGS.npy.tmp.npy`. The subsequent `tmp_npy.rename(npy_file)` fails with `FileNotFoundError`.
- **Impact:** First-time conversion from `HIGGS.csv` to `HIGGS.npy` always crashes after spending minutes parsing the 7.5 GB CSV.

### B6. `IrisDataProvider` has 100% train/val/test data leakage and ignores `split` and `transforms` (High)
- **Location:** [`iris_dataprovider.py:17-18,22-35`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/iris_dataprovider.py#L17-L35)
- **Root Cause:** `self.dataset_train, self.dataset_val, self.dataset_test = self.dataset, self.dataset, self.dataset` assigns the identical unsplit 150-row dataset to all three splits. `IrisDataset` also ignores `self.transforms`.
- **Impact:** Any validation/test metric on Iris evaluates directly on training data.

### B7. 3-way `split` crashes 9 of 12 providers with `ValueError: too many values to unpack` (Medium)
- **Location:** [`mnist_dataprovider.py:22`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/mnist_dataprovider.py#L22), [`fmnist_dataprovider.py:18`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/fmnist_dataprovider.py#L18), [`cifar10_dataprovider.py:15`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/cifar10_dataprovider.py#L15), [`cifar100_dataprovider.py:15`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/cifar100_dataprovider.py#L15), [`scan_dataprovider.py:15`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/scan_dataprovider.py#L15), [`math_dataprovider.py:15`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/math_dataprovider.py#L15), [`oddoneout_dataprovider.py:16`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/oddoneout_dataprovider.py#L16), [`cvr_oddoneout_dataprovider.py:15`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/cvr_oddoneout_dataprovider.py#L15), [`syneval_oddoneout_dataprovider.py:15`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/syneval_oddoneout_dataprovider.py#L15)
- **Root Cause:** All 9 providers unconditionally unpack two values: `self.dataset_train, self.dataset_val = self.dataset.random_split(self.dataset_split)`.
- **Impact:** Passing a 3-way split such as `split=[0.7, 0.15, 0.15]` raises `ValueError: too many values to unpack (expected 2)`.

### B8. `DataReader` defaults (`drop_last=True`, `shuffle=False`) applied identically to `train`, `val`, and `test` (Medium)
- **Location:** [`dataprovider.py:43,107-122`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/dataprovider.py#L43-L122)
- **Root Cause:** `DataProvider.train`, `.val`, and `.test` all construct `DataReader` with `drop_last=True` and `shuffle=False` (DataLoader default) and offer no parameters to override them.
- **Impact:**
  1. `provider.train` never shuffles between epochs.
  2. `provider.val` and `provider.test` silently discard up to `batch_size - 1` evaluation samples — or yield **0 batches** when a split is smaller than `batch_size` (e.g., 30-sample val split with `batch_size=32`).

### B9. `HIGGSDataProvider` 2-way split ignores `split_seed` (unshuffled) and `epoch_size=1` edge case (Medium)
- **Location:** [`higgs_dataprovider.py:163-180`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/higgs_dataprovider.py#L163-L180)
- **Root Cause:**
  1. In the 2-way split (`else` branch, lines 163–169), indices are assigned via contiguous `np.arange(train_size)` and `np.arange(train_size, true_dataset_size)` without shuffling, ignoring `split_seed` completely.
  2. Line 172 checks `isinstance(epoch_size, int) and epoch_size > 1`; `epoch_size=1` falls into the `else` branch (so it does not propagate percentage to val/test), and `bool` values pass `isinstance(..., int)`.

### B10. `transforms` contract mismatch: type-hinted as `transforms.Compose`, invoked as `.apply()` (Medium)
- **Location:** [`dataprovider.py:81`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/dataprovider.py#L81), [`mnist_dataprovider.py:67`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/mnist_dataprovider.py#L67), [`fmnist_dataprovider.py:63`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/fmnist_dataprovider.py#L63), [`cifar10_dataprovider.py:56`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/cifar10_dataprovider.py#L56), [`cifar100_dataprovider.py:51`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/cifar100_dataprovider.py#L51), [`covertype_dataprovider.py:182`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/covertype_dataprovider.py#L182), [`higgs_dataprovider.py:412`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/higgs_dataprovider.py#L412), [`scan_dataprovider.py:56`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/scan_dataprovider.py#L56), [`math_dataprovider.py:54`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/math_dataprovider.py#L54)
- **Root Cause:** `DataProvider.__init__` annotates `transforms: transforms.Compose = None`, but 8 of 12 datasets call `self.transforms.apply(x)` (specific to `MetaTransform`, which does not implement `__call__`). Only the 3 `*OddOneOutDataset` classes fall back to `self.transforms(x)`.
- **Impact:** Passing a standard `torchvision.transforms.Compose` (or any callable) raises `AttributeError: 'Compose' object has no attribute 'apply'`.

### B11. `Dataset.copy()` reloads the entire dataset from disk/sklearn per split (and regenerates random dummy labels) (Medium)
- **Location:** [`covertype_dataprovider.py:159-166`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/covertype_dataprovider.py#L159-L166), [`mnist_dataprovider.py:49-50`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/mnist_dataprovider.py#L49-L50), [`fmnist_dataprovider.py:45-46`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/fmnist_dataprovider.py#L45-L46), [`cifar10_dataprovider.py:40-41`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/cifar10_dataprovider.py#L40-L41), [`cifar100_dataprovider.py:39-40`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/cifar100_dataprovider.py#L39-L40), and the 5 `.pt` stub datasets.
- **Root Cause:** `copy(indices)` re-instantiates the dataset class from scratch. `CovertypeDataProvider` calls `fetch_covtype()` **4 times** for a 3-way split. `MnistDataset.copy()` additionally omits `transforms=self.transforms`. In `OddOneOutDataset`, `CVROddOneOutDataset`, and `SynevalOddOneOutDataset`, when the `.pt` file is absent, `copy()` draws fresh random `torch.randint` targets on every copy, desynchronizing split labels from `self.dataset`.

### B12. `MNIST` / `FashionMNIST` return 2D `(28, 28)` tensors (missing channel dimension) (Medium)
- **Location:** [`mnist_dataprovider.py:38`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/mnist_dataprovider.py#L38), [`fmnist_dataprovider.py:34`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/fmnist_dataprovider.py#L34)
- **Root Cause:** `self.ds.data.to(dtype=torch.float) / 255` omits `.unsqueeze(1)`, yielding `(N, 28, 28)` instead of `(N, 1, 28, 28)` (unlike `CIFAR10`/`CIFAR100`, which produce `(N, 3, 32, 32)`).
- **Impact:** Incompatible with `nn.Conv2d` (expects `(B, C, H, W)`) and contradicts the previous documentation.

### B13. `tests/test_higgs.py` is completely out of sync with the refactored `HIGGSDataset` (Medium)
- **Location:** [`tests/test_higgs.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/tests/test_higgs.py)
- **Root Cause:** Beyond the `parallelazy` collection failure (B1), `test_higgs.py` targets the pre-rewrite `HIGGSDataset` API: it passes `root=temp_dir` to `HIGGSDataset(...)` (now requires positional `full_x, full_y`), calls removed `_sample_epoch(seed=...)`, accesses removed `current_epoch_indices`, and expects in-place normalized `dataset.x` (now normalized lazily in `__getitem__`).

### B14. Minor defects
- **`CovertypeDataset` in-place normalization** ([`covertype_dataprovider.py:145-158`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/covertype_dataprovider.py#L145-L158)): calling `compute_normalization()` or `apply_normalization()` twice double-normalizes `self.x[:, :10]` in-place.
- **`torch.load()` without `weights_only=True`** in all 5 `.pt` providers (`scan`, `math`, `oddoneout`, `cvr_oddoneout`, `syneval_oddoneout`) and silent dummy-tensor fallbacks with no warning.
- **`IndexedDataset.__getitem__`** ([`dataprovider.py:18-20`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/dataprovider.py#L18-L20)): hardcodes 2-tuple unpacking `x, y = self.dataset[idx]`.
- **`HIGGS` `_load_or_build_npy` read-only memmap & eager label copy** ([`higgs_dataprovider.py:278-279`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/higgs_dataprovider.py#L278-L279)): `torch.from_numpy(data[:, 1:])` on a read-only memmap triggers PyTorch's non-writable tensor warning; `data[:, 0].astype(np.int64)` eagerly copies labels into RAM (~88 MB for 11M rows).

---

## 3. Inconsistencies & Design Issues

1. **Unused `_registry` vs. hardcoded `DataProviderFactory`**:
   - Both `DataProvider` and `DataReader` auto-register subclasses via `__init_subclass__` into `_registry` and raise `ValueError` on name collision, yet neither exposes a lookup API and `DataProviderFactory` ignores `_registry` completely in favor of an 11-branch `if/elif`.
2. **Incomplete exports & missing factory entry**:
   - `__init__.py` exports only 5 of 12 providers (`Covertype`, `FashionMnist`, `HIGGS`, `Iris`, `Mnist`), omitting `CIFAR10`, `CIFAR100`, `SCAN`, `Math`, `OddOneOut`, `CVROddOneOut`, and `SynevalOddOneOut`.
   - `OddOneOutDataProvider` is absent from both `__init__.py` and `DataProviderFactory`.
3. **Dataset name & metadata drift**:
   - `FashionMnistDataProvider.name` is `'fmnist'`, whereas `DataProviderFactory` keys it as `'fashion_mnist'`.
   - Only `IrisDataProvider` and `MnistDataProvider` set `self.criterion = nn.CrossEntropyLoss()`; the other 10 leave it unset.
   - Only `CovertypeDataProvider` and `HIGGSDataProvider` populate `self.x_shape` and `self.y_shape`; the other 10 leave them `None`.
   - Only `CovertypeDataProvider` and `HIGGSDataProvider` accept `root=...`; all other disk-backed providers hardcode `'./data'` (breaking whenever CWD is not the project root).
4. **CUDA fork/spawn heuristic in `DataReader`**:
   - `dataprovider.py:47-48` picks `"fork"` if `not torch.cuda.is_initialized()` else `"spawn"`. If a `DataReader` is created *before* CUDA initialization (with `persistent_workers=True`) and CUDA is initialized later in the parent, fork-safety is preserved only for workers already forked; conversely, in multithreaded contexts (e.g., Dash UI or `parallelazy` workers), `"fork"` can deadlock even without CUDA.
5. **Config drift**:
   - `DataConfig` (`dataset_name`, `batch_size`) is decoupled from `parallelazy`, while `DataProviderFactory.from_config()` still expects `PipelineConfig` (reading `model_type`). Split ratios, `root`, `num_workers`, `indexed_dataset`, and `shuffle` are not yet represented on `DataConfig`.

---

## 4. Opportunities & Recommendations

1. **Decouple `dataprovider` from `parallelazy` (Quick Win / High Impact)**:
   - Use `from __future__ import annotations` + `TYPE_CHECKING` guard (or accept `DataConfig | Protocol`) in `dataprovider_factory.py` instead of eagerly importing `PipelineConfig`.
   - Support `DataConfig` directly in `DataProviderFactory.from_config()` via `getattr(config, "model_type", None) == "fc"` (or a `flatten: bool = False` argument/field on `DataConfig`).
2. **Drive `DataProviderFactory` from `DataProvider._registry`**:
   - Register each provider with `class MnistDataProvider(DataProvider, plugin_name="mnist"):` and let `DataProviderFactory` look up `DataProvider.get(name)` instead of maintaining a 12-way `if/elif`. Add `DataProvider.list_registered()` and optional `override=True` for tests.
3. **Fix `DataReader` defaults per split**:
   - In `DataProvider`: default `shuffle=True, drop_last=True` for `train`, and `shuffle=False, drop_last=False` for `val` and `test`. Expose `drop_last`, `shuffle`, and `persistent_workers` as configurable options.
4. **Fix `HIGGSDataProvider` worker resampling and CSV->NPY conversion**:
   - Use `np.save` with a `.npy` temp filename (e.g., `npy_file.with_name(npy_file.stem + ".tmp.npy")`) so `rename()` succeeds.
   - Replace `HIGGSDataset.set_epoch()` in-place mutation with a custom PyTorch `Sampler` (so `DataLoader` sends fresh indices to persistent workers each epoch) or set `persistent_workers=False` for `HIGGSDataProvider`.
   - Shuffle the 2-way split with `split_seed` (matching the 3-way split behavior).
5. **Fix `IrisDataProvider` leakage & unify `root` + `copy()`**:
   - Actually split `IrisDataset` using `self.dataset_split` and support `transforms`.
   - Share backing tensors via `torch.utils.data.Subset` (or thin view classes like `HIGGSDataset`) instead of reloading datasets from disk on every `copy()`.
   - Make `root: str | Path = "./data"` configurable on all disk-backed providers (with an env var fallback like `POSITRONIXLAB_DATA_DIR`), and emit a `UserWarning` when `.pt` stub datasets fall back to dummy tensors.
6. **Normalize the `transforms` protocol**:
   - Implement `__call__` on `MetaTransform` (aliasing `apply`) and invoke `self.transforms(x)` uniformly across all datasets so both `MetaTransform` and standard `torchvision.transforms.Compose` work seamlessly.

---

## 5. Docs / Tests / Examples Summary

- **Docs**:
  - Updated [`docs/DATAPROVIDER_SYSTEM.md`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/docs/DATAPROVIDER_SYSTEM.md) to reflect the actual codebase (`DataConfig`, `DataProvider`, `DataReader`, `IndexedDataset`, `DataProviderFactory`, all 12 providers, HIGGS memmap/resampling architecture, and current caveats).
  - Created [`docs/DATAPROVIDER_USAGE.md`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/docs/DATAPROVIDER_USAGE.md) with accurate, runnable usage patterns.
- **Tests**:
  - Audited [`tests/test_higgs.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/tests/test_higgs.py) (fails collection without `parallelazy` and targets the pre-rewrite `HIGGSDataset` API).
  - Created [`tests/test_dataprovider_core.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/tests/test_dataprovider_core.py): offline CPU unit test suite covering `DataConfig`, `IndexedDataset`, `DataReader`, `DataProvider`, `DataProviderFactory`, offline concrete providers (`Iris`, `Math`, `SCAN`, `OddOneOut`, `CVR`, `Syneval`, and `HIGGS` with a temporary mock `.npy`), plus `@pytest.mark.xfail(strict=True)` regression pins for confirmed bugs (B2, B3, B5, B6, B7, B9, B10).
- **Examples**:
  - Created [`examples/example_dataprovider.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/examples/example_dataprovider.py): runnable offline CPU walkthrough of `DataConfig`, `IrisDataProvider`, `IndexedDataset`, custom `DataProvider` subclassing, and `DataProviderFactory`.
