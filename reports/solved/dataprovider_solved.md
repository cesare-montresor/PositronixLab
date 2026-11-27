# `positronixlab/dataprovider` — Solved Issues Archive

> **Archived:** 2026-11-27 (`v0.1.3` / `5740a0c` + working tree).
> **Active open issues:** See [dataprovider_analysis.md](file:///home/cesare/Projects/PositronixLab/reports/dataprovider_analysis.md).

---

## Resolved Bugs (13 of 14 Original Audit Bugs + Partial B8)

| ID | Summary | Resolution Evidence |
| :--- | :--- | :--- |
| **B1** | Eager transitive `parallelazy` import broke standalone module import | `__init__.py` upgraded to PEP 562 lazy exports + `TYPE_CHECKING` guard in `dataprovider_factory.py:11-12`. |
| **B2** | `DataProviderFactory.from_config()` crashed on `DataConfig` (`model_type` missing) | `dataprovider_factory.py:119-122` accepts `DataConfig`; `test_b2_factory_accepts_data_config` passes. |
| **B3** | `from_config(..., split=None)` silently dropped test split on `HIGGS` & `Covertype` | `dataprovider_factory.py:139-140`, `higgs_dataprovider.py:60`, `covertype_dataprovider.py:52`; `test_b3_*` passes. |
| **B4** | `HIGGSDataProvider.set_epoch()` defeated by `persistent_workers=True` | Replaced with `EpochSubsampleSampler` (`dataprovider.py:223-264`, `higgs_dataprovider.py:30-42, 182-194, 205-276`). |
| **B5** | `_load_or_build_npy()` `FileNotFoundError` on `HIGGS.csv` → `HIGGS.npy` rename | Fixed temp filename to `f"{npy_file.stem}.tmp.npy"` (`higgs_dataprovider.py:308-310`); `test_b5_*` passes. |
| **B6** | `IrisDataProvider` 100% train/val/test leakage & ignored `split`/`transforms` | Properly splits `IrisDataset` and applies transforms (`iris_dataprovider.py:19-54`); `test_b6_*` passes. |
| **B7** | 3-way `split` crashed 9 of 12 providers (`too many values to unpack`) | Unified `_split_dataset` (`dataprovider.py:410-436`) and `higgs_dataprovider.py:93-131`; `test_b7_*` passes. |
| **B8 (partial)** | `provider.train` defaulted to `shuffle=False` | `shuffle_train=True` added (`dataprovider.py:375, 468`). *(Note: `drop_last=True` on `val`/`test` remains open.)* |
| **B9** | `HIGGS` 2-way split ignored `split_seed`; `epoch_size=1` / `bool` edge case | Fixed in `higgs_dataprovider.py:117-129, 134, 391`; `test_b9_*` passes. |
| **B10** | `transforms` contract mismatch (`.apply()` vs callable) | Unified via `apply_transform()` (`dataprovider.py:65-71`) + `MetaTransform.__call__`; `test_b10_*` passes. |
| **B11** | `Dataset.copy()` reloaded entire dataset from disk/sklearn per split | Replaced with zero-copy index views via `BaseTensorDataset` (`dataprovider.py:90-220`). |
| **B12** | `MNIST`/`FashionMNIST` returned 2D `(28, 28)` without channel dim | Added `.unsqueeze(1)` → `(1, 28, 28)` (`mnist_dataprovider.py:34`, `fmnist_dataprovider.py:34`). |
| **B13** | Stale `tests/test_higgs.py` out of sync with refactored `HIGGSDataset` | Removed; replaced by `tests/dataprovider/test_dataprovider_core.py`. |
| **B14** | Covertype double-norm, `weights_only`, `IndexedDataset` 2-tuple, HIGGS memmap warning | Fixed across `dataprovider.py:83-87, 138-158, 207-212`, `higgs_dataprovider.py:315-322, 437-438`, and `weights_only=True`. |

## Resolved Design & Refactor Items

- **Registry-driven `DataProviderFactory`:** Hardcoded 11-branch `if/elif` replaced with `_registry` + lazy `_BUILTIN_MODULES` + `_FAMILY_PRESETS`.
- **Full `__init__.py` exports:** Expanded from 5 providers to all 16 concrete providers + `CompositionalDataProvider` + `DataProviderLoader` (PEP 562 lazy).
- **Removed stub `oddoneout_dataprovider.py`:** Upgraded `SCAN`, `Math`, `CVR`, `SynEval` to real providers and added `ClassRemoval`, `AddMul`, `TupleArithmetic`, `PermutedMnist`, and `CVRSimCLR`.
- **`DataConfig` expanded:** Now includes `split`, `num_workers`, `root`, `flatten`, `indexed_dataset`, `shuffle_train`, `drop_last`, `persistent_workers`, `pin_memory`, `seed`, `dataset_kwargs`.

## Resolved Minor & Hygiene Cleanups (2026-11-27)

- **Duplicate dictionary keys (`F601`):** Removed 6 duplicate keys in `dataprovider/__init__.py:144-149`.
- **Clone helpers forwarding `seed`:** `AddMulDataProvider.with_restrict()` and `TupleArithmeticDataProvider.with_restrict_tuple()` now forward `seed=self.seed`.
- **Redundant `if/elif` branch:** Simplified `cache_dir` resolution in `syneval_oddoneout_dataprovider.py:192-200`.
- **Runtime `assert`:** Replaced `assert self._synthetic is not None` with explicit `RuntimeError` in `cvr_simclr_dataprovider.py:207`.
- **`warnings.warn` `stacklevel` (v0.1.3 C8 / `B028`):** Added `stacklevel=2` in `higgs_dataprovider.py:403`.
- **`getattr` constant (`B009`):** Replaced `getattr(ds, "targets"/"y")` with direct attribute access in `class_removal_dataprovider.py:25-30`.
