# `positronixlab/deeplearning` — Open Issues Analysis

> **Status (2026-11-27):** **28 passed, 5 xfailed, 0 failed** (`tests/deeplearning/`).
> **Solved issues archive:** Resolved B1 (`EarlyStopping`), B2–B3 (`inference()`), B4 (`process_batch()` grad guard), and legacy cleanup moved to [solved/deeplearning_solved.md](file:///home/cesare/Projects/PositronixLab/reports/solved/deeplearning_solved.md).

---

## 1. Open Bugs & Defects

### Medium Severity

1. **Empty `DataLoader` crashes `process_epoch()` and `inference()` (B5)**
   - **Location:** `deeplearning_base.py:137-158`, `deeplearning_train.py:216-218` (xfail `test_deeplearning.py:507`)
   - **Details:** `UnboundLocalError` on `batch_metrics`/`batch_idx` and `ZeroDivisionError` on `batch_cnt == 0` / `data_num == 0`.

3. **`process_epoch()` double-updates the `tqdm` progress bar (B6)**
   - **Location:** `deeplearning_base.py:111, 126`
   - **Details:** Manual `batch_bar.update()` inside `enumerate(batch_bar)` increments progress twice per batch (200%).

4. **`DeeplearningTrain.on_batch_start()` drops `with_data` & ignores invalid `phase` (B8)**
   - **Location:** `deeplearning_train.py:84-112` (xfail `test_deeplearning.py:544`)
   - **Details:** Omits `with_data` parameter supported by `DeeplearningBase`/`DeeplearningEval`; typos like `phase="validation"` silently register nothing.

5. **Multiple `backward=True` metrics crash in `process_batch()` (B9)**
   - **Location:** `deeplearning_base.py:90-97` (xfail `test_deeplearning.py:552`)
   - **Details:** Calls `loss.backward()` per metric without summing or `retain_graph=True`; uses deprecated `loss.data.item()`.

6. **`DeeplearningBase.__init__` requires `config.task` & eagerly imports `PipelineConfig` (Config gap)**
   - **Location:** `deeplearning_base.py:12, 33`, `__init__.py:1-12` (xfail `test_deeplearning.py:569`)
   - **Details:** Incompatible with standalone `TrainConfig`; `__init__.py` eagerly imports `torch`, `sklearn`, `tqdm.asyncio`, and `PipelineConfig` (`parallelazy`).

### Low Severity

7. **Off-by-one in `draft` mode (B7):** `deeplearning_base.py:107, 129-134`, `deeplearning_train.py:172, 204-208` (xfail `test_deeplearning.py:523`) — runs 21 batches instead of 20; dead `draft_batches = 20` before `break`.
8. **`inference()` `(N, 1)` target broadcast & `eval()` mode leak on exception (B10):** `deeplearning_train.py:193, 220-221` — `y == y_hat_cls` broadcasts `(N, 1) == (N,)` to `(N, N)`; `self.model.train()` not wrapped in `try/finally`.
9. **Unhandled batch arity, `sanity_check()` error order, `evaluate()` return annotation (B11):** `deeplearning_base.py:73-77`, `deeplearning_train.py:117-127`, `deeplearning_eval.py:58, 71` (`-> dict` vs `list`).

---

## 2. Open Design & Hygiene Issues

- **Config drift:** `TrainConfig` duplicates `epochs` and `train_epochs`, neither read by `DeeplearningTrain.train()`.
- **Unused constructor state & unweighted epoch average:** `DeeplearningBase.dataloader`/`draft` stored but unused; `process_epoch()` averages batches unweighted instead of sample-weighted; `_epoch_did_end` passes `Counter` instead of `dict`; `model.device` assumed.
- **Silent early-stopping mismatch:** `add_early_stopping(metric="loss")` silently never fires if metric name differs (e.g. `"ce_loss"`).
- **Hygiene (v0.1.3 F):** Implicit `Optional` (`RUF013` at `deeplearning_base.py:24, 185`, `deeplearning_train.py:57, 156-157`); unused loop var `value` (`B007` at `deeplearning_base.py:137`); redundant f-string (`F541` at `deeplearning_train.py:260`); 5 `print()` calls in library code; `tqdm.asyncio` imported in sync code.
