# `positronixlab/deeplearning` — Open Issues Analysis

> **Status (2026-11-27):** **32 passed, 1 xfailed, 0 failed** (`tests/deeplearning/`).
> **Solved issues archive:** Resolved B1–B9 and legacy cleanup moved to [solved/deeplearning_solved.md](file:///home/cesare/Projects/PositronixLab/reports/solved/deeplearning_solved.md).

---

## 1. Open Bugs & Defects

### Medium Severity

1. **`DeeplearningBase.__init__` requires `config.task` & eagerly imports `PipelineConfig` (Config gap)**
   - **Location:** `deeplearning_base.py:12, 33`, `__init__.py:1-12` (xfail `test_deeplearning.py:551`)
   - **Details:** Incompatible with standalone `TrainConfig`; `__init__.py` eagerly imports `torch`, `sklearn`, `tqdm.asyncio`, and `PipelineConfig` (`parallelazy`).

### Low Severity

2. **`DeeplearningTrain.on_*()` does not validate `phase` values (B8 remaining):** `deeplearning_train.py:81-110` — typos like `phase="validation"` silently register nothing.
3. **`inference()` `(N, 1)` target broadcast & `eval()` mode leak on exception (B10):** `deeplearning_train.py:193, 227-228` — `y == y_hat_cls` broadcasts `(N, 1) == (N,)` to `(N, N)`; `self.model.train()` not wrapped in `try/finally`.
4. **Unhandled batch arity, `sanity_check()` error order, `evaluate()` return annotation (B11):** `deeplearning_base.py:73-77`, `deeplearning_train.py:117-127`, `deeplearning_eval.py:58, 71` (`-> dict` vs `list`).

---

## 2. Open Design & Hygiene Issues

- **Config drift:** `TrainConfig` duplicates `epochs` and `train_epochs`, neither read by `DeeplearningTrain.train()`.
- **Unused constructor state & unweighted epoch average:** `DeeplearningBase.dataloader`/`draft` stored but unused; `process_epoch()` averages batches unweighted instead of sample-weighted; `_epoch_did_end` passes `Counter` instead of `dict`; `model.device` assumed.
- **Silent early-stopping mismatch:** `add_early_stopping(metric="loss")` silently never fires if metric name differs (e.g. `"ce_loss"`).
- **Hygiene:** 5 `print()` calls in library code; `tqdm.asyncio` imported in sync code.
