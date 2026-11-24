# `positronixlab/deeplearning` — Module Analysis

## 1. Overview & Refactor Status

`positronixlab/deeplearning` provides the training, validation, and evaluation loop abstractions for `positronixlab`. It is currently **mid-refactor** from a legacy monolithic training script (`deeplearning_legacy.py`) and monolithic `PipelineConfig` into a layered callback-driven engine with composable `BaseConfig` hyperparameters:

| File | Role | Status |
|---|---|---|
| [`__init__.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/deeplearning/__init__.py) | Eagerly re-exports `TrainConfig`, `DeeplearningTrain`, `EarlyStopping`, `DeeplearningEval`, `DeeplearningBase`. | Active |
| [`config.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/deeplearning/config.py) | `TrainConfig(BaseConfig)`: composable training hyperparameter mixin (`optimizer`, `learning_rate`, `epochs`, `train_epochs`). | Active (new foundation, not yet wired into loops) |
| [`deeplearning_base.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/deeplearning/deeplearning_base.py) | `DeeplearningBase`: core batch/epoch loop, metric registry + best-epoch tracking, and 5 lifecycle callbacks (`on_batch_start`, `on_prediction`, `on_batch_end`, `on_epoch_start`, `on_epoch_end`). | Active |
| [`deeplearning_train.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/deeplearning/deeplearning_train.py) | `DeeplearningTrain` (orchestrates `trainer`/`validator` `DeeplearningBase` instances + standalone `inference()`) and `EarlyStopping`. | Active (contains high-severity bugs) |
| [`deeplearning_eval.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/deeplearning/deeplearning_eval.py) | `DeeplearningEval`: thin wrapper around a single `DeeplearningBase` instance (`optimizer=None`) for multi-epoch evaluation. | Active |
| `deeplearning_legacy.py` | Archived pre-refactor implementation (do not touch). | Legacy |

### Composable Config Refactor Status
- `TrainConfig(BaseConfig)` has been introduced in [`config.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/deeplearning/config.py) and is already composed into `TrialConfig` in `positronixlab/nas`.
- However, `DeeplearningBase`, `DeeplearningTrain`, and `DeeplearningEval` still type-hint `config: PipelineConfig` (importing from `positronixlab.pipeline`, which transitively pulls in `parallelazy`), require `config.task` (present on `DataConfig` / `PipelineConfigV1` / `PipelineConfigV2`, but absent on both `TrainConfig` and `PipelineConfig`), and ignore `config.epochs` / `config.train_epochs`.

---

## 2. Bugs & Defects

### B1. `EarlyStopping.should_stop()` only maximizes metrics and initializes `metric_last = 0` (High)
- **Location:** [`deeplearning_train.py:52-54`](file:///home/cesare/Projects/PositronixLab/positronixlab/deeplearning/deeplearning_train.py#L52-L54), [`deeplearning_train.py:218-246`](file:///home/cesare/Projects/PositronixLab/positronixlab/deeplearning/deeplearning_train.py#L218-L246)
- **Root Cause:**
  1. **Direction mismatch:** `DeeplearningTrain.add_early_stopping()` defaults to `metric="loss"`, yet `EarlyStopping` has no `minimize` / `mode` parameter and hardcodes `gain = metric - self.metric_last` (`line 229`). When validation loss decreases every epoch (e.g. `1.0 -> 0.5 -> 0.2 -> 0.1 -> 0.05`), epoch 1 sets `metric_last = 1.0`, and every subsequent improvement yields a negative `gain`, triggering early stopping after `max_attempts + 1` epochs. Conversely, a diverging loss (`1.0 -> 2.0 -> 5.0`) resets the counter every epoch and never stops.
  2. **Broken baseline sentinel:** `self.metric_last = 0` (`line 223`) fails for any negative maximized metric (e.g. negative log-likelihood, negative MSE, R² < 0) and for small positive first-epoch metrics (`0 < metric < min_gain`), treating the initial observation as a failed attempt rather than establishing a baseline (`None`).
  3. **Off-by-one patience:** `if self.num_attempts > self.max_attempts:` (`line 242`) allows `max_attempts + 1` non-improving epochs before returning `True` (e.g. 4 failures when `max_attempts=3`), while logging `(4/3)` on `line 240`.
  4. **Formatting defect:** `+{int(gain*100)}% metric` (`lines 238, 240`) prints `+-50%` on negative gains and labels non-percentage metrics as `%`.
- **Impact:** Early stopping with the default `metric="loss"` is completely inverted — it halts healthy converging runs and lets diverging runs continue indefinitely.

### B2. `DeeplearningTrain.inference()` under-reports `avg_loss` by a factor of `batch_size` (High)
- **Location:** [`deeplearning_train.py:173-176`](file:///home/cesare/Projects/PositronixLab/positronixlab/deeplearning/deeplearning_train.py#L173-L176), [`deeplearning_train.py:204`](file:///home/cesare/Projects/PositronixLab/positronixlab/deeplearning/deeplearning_train.py#L204)
- **Root Cause:** `total_loss += loss.detach().item()` accumulates the per-batch mean loss (standard PyTorch criteria and `MetaCriterion` use `reduction="mean"`), while `data_num += len(x)` accumulates total sample count. Line 204 then computes `total_loss / data_num` instead of weighting each batch loss by `len(x)` (`total_loss += loss.detach().item() * len(x)`).
- **Impact:** Reported inference loss is shrunk by `~batch_size` (e.g. 64× too small for `batch_size=64`) and varies arbitrarily with batch size, making it incomparable with `train()`/`validate()` losses.

### B3. `DeeplearningTrain.inference()` crashes on 3-tuple `(x, y, idx)` batches (High)
- **Location:** [`deeplearning_train.py:168`](file:///home/cesare/Projects/PositronixLab/positronixlab/deeplearning/deeplearning_train.py#L168)
- **Root Cause:** `for batch_idx, (x, y) in enumerate(tqdm(dataloader, desc="Inference")):` hardcodes 2-tuple unpacking, whereas `DeeplearningBase.process_batch()` (`deeplearning_base.py:74-77`) explicitly supports both `(x, y)` and `(x, y, idx)` (produced by `IndexedDataset`).
- **Impact:** Raises `ValueError: too many values to unpack (expected 2)` on any dataset yielding sample indices.

### B4. `DeeplearningBase.process_batch()` runs forward pass with gradients enabled when called directly in eval mode (Medium)
- **Location:** [`deeplearning_base.py:85-98`](file:///home/cesare/Projects/PositronixLab/positronixlab/deeplearning/deeplearning_base.py#L85-L98)
- **Root Cause:** `y_hat = self.model(x)` (`line 85`) executes *before* `with torch.set_grad_enabled(self.is_training):` (`line 90`). When invoked via `process_epoch()`, the outer `with torch.set_grad_enabled(self.is_training):` (`line 115`) masks the issue, but calling `process_batch()` directly with `optimizer=None` builds a full autograd graph during forward inference.
- **Impact:** Unnecessary GPU/CPU memory retention and slower direct batch evaluation.

### B5. Empty `DataLoader` crashes `process_epoch()` and `inference()` (Medium)
- **Location:** [`deeplearning_base.py:138-159`](file:///home/cesare/Projects/PositronixLab/positronixlab/deeplearning/deeplearning_base.py#L138-L159), [`deeplearning_train.py:204-206`](file:///home/cesare/Projects/PositronixLab/positronixlab/deeplearning/deeplearning_train.py#L204-L206)
- **Root Cause:**
  - In `DeeplearningBase.process_epoch()`, if `len(dataloader) == 0`, the loop never executes so `batch_metrics` (`line 138`) and `batch_idx` (`line 159`) are unbound (`UnboundLocalError`), and `epoch_metrics[name] /= batch_cnt` (`line 140`) would divide by zero.
  - In `DeeplearningTrain.inference()`, `data_num == 0` causes `ZeroDivisionError` on `lines 204-206`.
- **Impact:** Unhandled crash on empty validation/test splits.

### B6. `DeeplearningBase.process_epoch()` double-updates the `tqdm` progress bar (Medium)
- **Location:** [`deeplearning_base.py:112-127`](file:///home/cesare/Projects/PositronixLab/positronixlab/deeplearning/deeplearning_base.py#L112-L127)
- **Root Cause:** Iterating over `batch_bar = tqdm(dataloader, ...)` already calls `batch_bar.update(1)` per iteration. Calling `batch_bar.update()` manually on `line 127` increments the bar twice per batch.
- **Impact:** Progress bar reaches 100% halfway through the epoch and ends at 200% (`2N/N`), distorting ETA and throughput readouts.

### B7. Off-by-one in `draft` mode: runs 21 batches instead of 20 (Low)
- **Location:** [`deeplearning_base.py:108,130-135`](file:///home/cesare/Projects/PositronixLab/positronixlab/deeplearning/deeplearning_base.py#L108), [`deeplearning_train.py:164,192-196`](file:///home/cesare/Projects/PositronixLab/positronixlab/deeplearning/deeplearning_train.py#L164)
- **Root Cause:** `draft_batches = 20` is decremented *after* each batch and breaks only when `draft_batches < 0`, so batch indices `0..20` (21 batches total) are executed. Line 134 (`draft_batches = 20` immediately before `break`) is also dead code.
- **Impact:** Minor mismatch between intended 20-batch cap and actual 21-batch execution.

### B8. `DeeplearningTrain.on_batch_start()` drops `with_data` parameter and ignores invalid `phase` values (Medium)
- **Location:** [`deeplearning_train.py:76-105`](file:///home/cesare/Projects/PositronixLab/positronixlab/deeplearning/deeplearning_train.py#L76-L105)
- **Root Cause:**
  - `DeeplearningBase.on_batch_start(callback, with_data=False)` and `DeeplearningEval.on_batch_start(callback, with_data=False)` accept `with_data`, but `DeeplearningTrain.on_batch_start(callback, phase="both")` omits it.
  - All five callback forwarding methods check `if phase in ("train", "both"):` / `if phase in ("val", "both"):` without validating `phase`; typos such as `phase="validation"` silently register no callback.
- **Impact:** Callers cannot opt into batch data via `DeeplearningTrain.on_batch_start()`, and invalid `phase` names fail silently.

### B9. Multiple `backward=True` metrics crash in `DeeplearningBase.process_batch()` (Medium)
- **Location:** [`deeplearning_base.py:91-98`](file:///home/cesare/Projects/PositronixLab/positronixlab/deeplearning/deeplearning_base.py#L91-L98)
- **Root Cause:** Each metric with `backward=True` calls `loss.backward()` independently inside the loop without summing losses first or passing `retain_graph=True`. Registering two `backward=True` metrics sharing the forward graph raises `RuntimeError: Trying to backward through the graph a second time`. Additionally, `loss.data.item()` uses deprecated `.data` and fails if a non-backward metric returns a Python `float`.
- **Impact:** Multi-term training losses cannot be registered as separate `backward=True` metrics.

### B10. `DeeplearningTrain.inference()` accuracy/CM silently corrupts on `(N, 1)` targets and leaks `eval()` mode on exceptions (Low)
- **Location:** [`deeplearning_train.py:158-159,180-183,208-209`](file:///home/cesare/Projects/PositronixLab/positronixlab/deeplearning/deeplearning_train.py#L180-L183)
- **Root Cause:**
  - `y == y_hat_cls` broadcasts `(N, 1) == (N,)` into an `(N, N)` boolean matrix, producing accuracies > 1.0 instead of squeezing or validating shape.
  - `self.model.train()` restoration (`line 209`) is not wrapped in `try...finally:`, so any exception inside the loop leaves the model in `eval()` mode.

### B11. Unhandled batch arity in `DeeplearningBase.process_batch()` and misleading `sanity_check()` error order (Low)
- **Location:** [`deeplearning_base.py:74-80`](file:///home/cesare/Projects/PositronixLab/positronixlab/deeplearning/deeplearning_base.py#L74-L80), [`deeplearning_train.py:109-120`](file:///home/cesare/Projects/PositronixLab/positronixlab/deeplearning/deeplearning_train.py#L109-L120), [`deeplearning_eval.py:61`](file:///home/cesare/Projects/PositronixLab/positronixlab/deeplearning/deeplearning_eval.py#L61)
- **Root Cause:**
  - `process_batch()` handles `len(data_batch) == 2` and `== 3` with no `else`, raising `UnboundLocalError` for other tuple lengths.
  - `DeeplearningTrain.sanity_check(must_backward=True)` checks `backward_found` before `len(self.trainer.metrics) == 0`, so an empty metrics dict reports `"No metric with backward=True found"` instead of `"No metrics found"`; `validate()` checks `self.trainer.metrics` instead of `self.validator.metrics`.
  - `DeeplearningEval.evaluate(self, epoch_total: int) -> dict` annotates return type as `dict`, but returns `self.history` (`list[dict[str, float]]`).

---

## 3. Inconsistencies & Design Issues

1. **Config mismatch (`TrainConfig` vs `PipelineConfig` vs `config.task`)**:
   - [`config.py:14-15`](file:///home/cesare/Projects/PositronixLab/positronixlab/deeplearning/config.py#L14-L15) defines both `epochs: int = 10` and `train_epochs: int = 10` (duplicating the legacy V1/V2 field drift), yet neither is read by `DeeplearningTrain.train(epoch_total)`.
   - [`deeplearning_base.py:34`](file:///home/cesare/Projects/PositronixLab/positronixlab/deeplearning/deeplearning_base.py#L34) accesses `self.task = config.task` and never uses `self.task`. This breaks initialization with standalone `TrainConfig()` or `PipelineConfig()`, forcing callers to compose a config with `.task`.
2. **Eager heavy imports in `__init__.py`**:
   - Importing `from positronixlab.deeplearning import TrainConfig` eagerly imports `torch`, `sklearn.metrics`, `tqdm.asyncio`, `DataProvider`, and `PipelineConfig` (which transitively imports `parallelazy`).
3. **Redundant constructor parameters in `DeeplearningBase`**:
   - `self.dataloader` (`line 32`) and `self.draft` (`line 35`) are stored in `__init__` but never read anywhere; `process_epoch()` requires `dataloader` again and defaults `draft: int = False` (also wrong type annotation).
4. **Unweighted batch averaging vs sample-weighted evaluation**:
   - `DeeplearningBase.process_epoch()` computes an unweighted mean over batches (over-weighting a smaller final batch when `drop_last=False`) and iterates `batch_metrics.items()` instead of `epoch_metrics.items()` at `line 138`.
5. **Silent early-stopping misconfiguration**:
   - [`deeplearning_train.py:136`](file:///home/cesare/Projects/PositronixLab/positronixlab/deeplearning/deeplearning_train.py#L136) checks `if self.early_stop is not None and self.early_stop_metric in val_metrics:`. If the user calls `add_early_stopping()` (default `metric="loss"`) while naming their metric `"ce_loss"`, early stopping silently never runs.
6. **Single-slot callbacks & `Counter` leakage**:
   - Each hook (`on_batch_start`, etc.) stores a single callable, silently overwriting any previously registered listener.
   - `_epoch_did_end` passes a `collections.Counter` instead of a plain `dict`.
7. **Logging & console I/O inconsistency**:
   - All three active files define `logger = logging.getLogger(__name__)` and never call it; instead they use `print(...)`, unconditional `tqdm` progress bars (`tqdm.asyncio` in `deeplearning_train.py:13` despite synchronous code), and root `logging.info(...)` in `EarlyStopping`.
8. **Implicit `model.device` requirement & unused imports**:
   - Accessing `model.device` directly fails on standard `nn.Module` subclasses that lack a `.device` attribute.
   - Unused imports exist across `deeplearning_base.py` (`DataProvider`, `Callable`), `deeplearning_train.py` (`Any`), and `deeplearning_eval.py` (`Any`, `torch`, `DataProvider`, `MetaModel`), along with a stale copy-pasted comment on `deeplearning_eval.py:38`.

---

## 4. Opportunities & Recommendations

1. **Fix `EarlyStopping` immediately (Priority 1)**:
   - Add `minimize: bool = True` (or `mode: Literal["min", "max"] = "min"`), initialize `self.metric_last: float | None = None` so epoch 1 establishes the baseline, fix patience check to `self.num_attempts >= self.max_attempts`, and auto-infer `minimize` from `self.validator.metrics[metric]["minimize"]` in `add_early_stopping()` (raising `KeyError`/`ValueError` if `early_stop_metric` is missing from `val_metrics` during `sanity_check()`).
2. **Fix `DeeplearningTrain.inference()` (Priority 1)**:
   - Weight batch loss by `len(x)` (`total_loss += loss.detach().item() * len(x)`), support both `(x, y)` and `(x, y, idx)` batches (via a shared `_unpack_batch()` helper with `DeeplearningBase`), guard `data_num == 0`, wrap `model.eval()`/`train()` in `try...finally:`, and switch `tqdm.asyncio` to `tqdm`.
3. **Complete the `TrainConfig` decoupling (Priority 2)**:
   - Remove the unused `self.task = config.task` from `DeeplearningBase.__init__`, drop the duplicate `train_epochs` field from `TrainConfig` (or alias it cleanly), default `train(epoch_total: int | None = None)` to `self.config.epochs`, and add lazy PEP 562 exports in `positronixlab/deeplearning/__init__.py` so importing `TrainConfig` does not pull in `torch`/`sklearn`/`parallelazy`.
4. **Harden `DeeplearningBase` (Priority 2)**:
   - Move `with torch.set_grad_enabled(self.is_training):` above `self.model(x)` in `process_batch()`.
   - Remove the duplicate `batch_bar.update()` call (`line 127`).
   - Handle empty dataloaders cleanly (return `{}` or `0.0`s without `UnboundLocalError`/`ZeroDivisionError`).
   - Sum all `backward=True` losses into a single tensor before calling `.backward()` once, and support sample-weighted epoch averages (`len(x)`).
   - Fix `draft` mode off-by-one (`batch_cnt >= draft_batches`) and default `process_epoch(dataloader=None, draft=None)` to `self.dataloader` / `self.draft`.
5. **Standardize callbacks, logging, and device detection (Priority 3)**:
   - Forward `with_data` in `DeeplearningTrain.on_batch_start()`, validate `phase in ("train", "val", "both")`, consider supporting lists of callbacks, pass `dict(epoch_metrics)` to `_epoch_did_end`, and resolve device via `getattr(model, "device", None) or next(model.parameters()).device`.
   - Replace `print()` / root `logging.info()` with module `logger`, and add an optional `verbose: bool = True` flag to silence `tqdm` in headless NAS/CI runs.

---

## 5. Docs / Tests / Examples Summary

- **Documentation ([`docs/`](file:///home/cesare/Projects/PositronixLab/positronixlab/deeplearning/docs/))**:
  - Created [`DEEPLEARNING_SYSTEM.md`](file:///home/cesare/Projects/PositronixLab/positronixlab/deeplearning/docs/DEEPLEARNING_SYSTEM.md) — architecture, lifecycle flow, API reference for `TrainConfig`, `DeeplearningBase`, `DeeplearningTrain`, `EarlyStopping`, `DeeplearningEval`, and runtime contracts.
  - Created [`DEEPLEARNING_USAGE.md`](file:///home/cesare/Projects/PositronixLab/positronixlab/deeplearning/docs/DEEPLEARNING_USAGE.md) — practical usage guide covering config composition, metrics, callbacks, evaluation/inference, draft mode, and workarounds for known defects.
- **Tests ([`tests/`](file:///home/cesare/Projects/PositronixLab/positronixlab/deeplearning/tests/))**:
  - Created [`test_deeplearning.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/deeplearning/tests/test_deeplearning.py) — CPU unit test suite covering `TrainConfig`, `DeeplearningBase`, `DeeplearningTrain`, `EarlyStopping`, and `DeeplearningEval`, including strict `xfail` tests pinning confirmed bugs B1–B8 and config/callback gaps.
- **Examples ([`examples/`](file:///home/cesare/Projects/PositronixLab/positronixlab/deeplearning/examples/))**:
  - Created [`example_deeplearning_train.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/deeplearning/examples/example_deeplearning_train.py) — concise, self-contained runnable CPU example demonstrating `TrainConfig` composition, `DeeplearningTrain` with metrics and callbacks, `inference()`, and `DeeplearningEval`.
