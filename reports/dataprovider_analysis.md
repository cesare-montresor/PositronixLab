# `positronixlab/dataprovider` — Open Issues Analysis

> **Status (2026-11-27 — `v0.1.3` / `5740a0c` + working tree):** **44 passed, 0 xfailed, 0 failed** (`tests/dataprovider/`).
> **Solved issues archive:** 13 of 14 original bugs + minor cleanups moved to [solved/dataprovider_solved.md](file:///home/cesare/Projects/PositronixLab/reports/solved/dataprovider_solved.md).

---

## 1. Open Bugs & Defects

### Medium Severity

1. **`drop_last=True` still applied to `val` and `test` loaders (Original B8 remaining)**
   - **Location:** `positronixlab/dataprovider/dataprovider.py:376, 479, 518, 558, 579` (pinned as current behavior in `tests/dataprovider/test_dataprovider_core.py:635`)
   - **Details:** `shuffle_train=True` was fixed, but `val` and `test` `DataReader`s still inherit `drop_last=True`, silently discarding up to `batch_size - 1` evaluation samples (or yielding 0 batches when `len(split) < batch_size`).
   - **Fix:** Default `drop_last=False` for `val`/`test` loaders (keep `drop_last` configurable separately for train vs eval).

2. **Split-seed edge cases & `dataset_test` aliasing (v0.1.3 B8 remaining)**
   - **Location:** `dataprovider.py:420, 428`; `class_removal_dataprovider.py:102, 112`; `cvr_simclr_dataprovider.py:287`
   - **Details:** `DataProvider.seed` safely defaults to `42` and unsplit `DataProvider.test` returns `None`, **however**:
     - Passing explicit `seed=None` falls back to the global RNG (risking cross-process split divergence in spawned workers).
     - `ClassRemovalDataProvider` and `CVRSimCLRDataProvider` alias `dataset_test = dataset_val`, so evaluating on `.test` evaluates on validation data.

### Low Severity & Hygiene

3. **Heavy top-level imports (v0.1.3 E2):** Top-level `pandas`/`requests` in `higgs_dataprovider.py:12-13`, plus top-level `requests` in `scan_dataprovider.py:21` and `math_dataprovider.py:23`.
4. **Misc hygiene (v0.1.3 F):** `print()` in `higgs_dataprovider.py:303, 331-332, 342, 348`; `zip()` without `strict=True` in `cvr_simclr_dataprovider.py:158`, `scan_dataprovider.py:268`, `math_dataprovider.py:444`; `pickle.load` without untrusted-file docstring warning in `syneval_oddoneout_dataprovider.py:95`.

---

## 2. Open Design Notes

- **CUDA fork/spawn heuristic (`dataprovider.py`):** Picking `"fork"` when `not torch.cuda.is_initialized()` can still deadlock in multithreaded processes (e.g., Dash UI or `parallelazy` workers) even before CUDA is initialized.
- **Metadata consistency:** Not all providers uniformly populate `self.criterion`, `self.x_shape`, and `self.y_shape`.
