# PositronixLab DeepLearning System

Callback-driven training, validation, and evaluation engine for PyTorch models in `positronixlab`. Designed to decouple model optimization loops from downstream observers (e.g. `ModularObserver`, `ModularSpace`, stats streams) via lightweight lifecycle hooks.

## Architecture

```
                        TrainConfig (BaseConfig mixin)
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
        DeeplearningTrain                        DeeplearningEval
   ┌───────────────────────────┐            ┌────────────────────────┐
   │ trainer   (optimizer=opt) │            │ evaluator (opt=None)   │
   │ validator (optimizer=None)│            └───────────┬────────────┘
   │ early_stop (EarlyStopping)│                        │
   │ inference()  [standalone] │                        │
   └─────────────┬─────────────┘                        │
                 └───────────────────┬──────────────────┘
                                     ▼
                             DeeplearningBase
        ┌────────────────────────────────────────────────────────┐
        │ process_epoch(dataloader, epoch, epoch_total, draft)   │
        │   ├─ _epoch_did_start                                  │
        │   ├─ for batch in dataloader:                          │
        │   │    process_batch(...)                              │
        │   │      ├─ _batch_did_start                           │
        │   │      ├─ y_hat = model(x)                           │
        │   │      ├─ _did_prediction                            │
        │   │      ├─ evaluate metrics & backward (if training)  │
        │   │      ├─ optimizer.step() (if training)             │
        │   │      └─ _batch_did_end                             │
        │   ├─ average metrics & update history / best_idx       │
        │   └─ _epoch_did_end                                    │
        └────────────────────────────────────────────────────────┘
```

## Module Layout

| File | Contents |
|---|---|
| `positronixlab/deeplearning/__init__.py` | Re-exports `TrainConfig`, `DeeplearningTrain`, `EarlyStopping`, `DeeplearningEval`, `DeeplearningBase`. |
| `positronixlab/deeplearning/config.py` | `TrainConfig(BaseConfig)` composable training hyperparameter dataclass. |
| `positronixlab/deeplearning/deeplearning_base.py` | `DeeplearningBase`: core batch/epoch loop, metric registry, and lifecycle callbacks. |
| `positronixlab/deeplearning/deeplearning_train.py` | `DeeplearningTrain` (train/val/inference orchestrator) and `EarlyStopping`. |
| `positronixlab/deeplearning/deeplearning_eval.py` | `DeeplearningEval`: evaluation wrapper around `DeeplearningBase`. |

---

## Runtime Contracts

### Model Contract
- **`model.device`** *(required by all classes)*: `DeeplearningBase`, `DeeplearningTrain`, and `DeeplearningEval` read `model.device` directly in `__init__` to move input tensors `x` and targets `y`. Standard `nn.Module` subclasses must expose a `.device` attribute (provided automatically by `MetaModel` / `BaseModel`).
- **`model.loss(x, y, y_hat)`** *(required by `DeeplearningTrain.inference()` only)*: `inference()` calls `self.model.loss(x, y, y_hat)` directly rather than registered metric criteria.

### Config Contract
- `DeeplearningBase.__init__` reads `config.task`.
- `DeeplearningTrain.__init__` reads `config.optimizer` (`"adamw"`, `"sgd"`, or `"adam"`) and `config.learning_rate`.
- `DeeplearningTrain.inference()` reads `config.task` (computes accuracy and optional confusion matrix when `config.task == "classification"`).
- Note: `TrainConfig` alone does **not** define `task`; compose it with `MetaModelConfig` (which defines `task: str = "classification"`) or a custom `BaseConfig` subclass defining `task: str` before passing it to `Deeplearning*`.

### DataLoader / Batch Contract
- Both `DeeplearningBase.process_batch()` and `DeeplearningTrain.inference()` accept either:
  - 2-tuples `(x, y)` (standard supervised batches), or
  - 3-tuples `(x, y, idx)` (indexed batches from `IndexedDataset`).
- `DeeplearningTrain` expects a `DataProvider`-like object exposing `.train` and `.val` `DataLoader` attributes.

---

## API Reference

### `TrainConfig(BaseConfig)` (`config.py`)

Composable dataclass mixin inheriting serialization (`to_dict`, `from_dict`, `to_yaml`, `save`, `load`, `with_updates`, `signature`) from `positronixlab.pipeline.config.BaseConfig`.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` *(inherited)* | `str` | `"default_config"` | Config identifier. |
| `optimizer` | `str` | `"adamw"` | Optimizer name (`"adamw"`, `"sgd"`, `"adam"`). |
| `learning_rate` | `float` | `0.001` | Optimizer learning rate. |
| `epochs` | `int` | `10` | Training epoch count (not automatically read by `train()`). |
| `train_epochs` | `int` | `10` | Legacy duplicate of `epochs`. |

---

### `DeeplearningBase` (`deeplearning_base.py`)

Core execution engine for a single loop role (training, validation, or evaluation).

#### Constructor
```python
DeeplearningBase(
    config: PipelineConfig,
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer = None,
    draft: bool = False,
)
```
- `is_training` *(property)*: `True` iff `self.optimizer is not None`.

#### Metric Registration
```python
add_metric(name: str, criterion=None, minimize: bool = True, backward: bool = False)
```
- `criterion(x, y, y_hat) -> torch.Tensor`: scalar tensor criterion.
- `minimize`: controls `best_idx` tracking across epochs (`True` tracks minimum; `False` tracks maximum).
- `backward`: when `True` and `is_training` is `True`, calls `loss.backward()` on this metric. Register at most **one** `backward=True` metric per forward graph.
- Stored in `self.metrics[name]` with keys `name`, `best_idx`, `history`, `minimize`, `criterion`, `backward`.

#### Execution Methods
- `process_batch(epoch, epoch_total, batch_idx, batch_total, data_batch) -> dict[str, float]`:
  Runs one forward pass, evaluates all registered metrics, optionally backpropagates and steps the optimizer, and returns `{metric_name: float_value}`.
- `process_epoch(dataloader, epoch, epoch_total, draft=False) -> tuple[dict[str, float], int, int]`:
  Iterates over `dataloader` under `torch.set_grad_enabled(self.is_training)`, computes unweighted per-batch mean for each metric, appends to `self.metrics[name]["history"]`, updates `best_idx`, and returns `(epoch_metrics, last_batch_idx, batch_total)`.

#### Lifecycle Callbacks
Each hook holds a **single** callback (calling it again replaces the previous callback):

| Registration Method | Callback Signature |
|---|---|
| `on_epoch_start(cb)` | `(epoch, epoch_total)` |
| `on_epoch_end(cb)` | `(epoch, epoch_total, epoch_metrics)` |
| `on_batch_start(cb, with_data=False)` | `(epoch, epoch_total, batch_idx, batch_total, data_or_none)` |
| `on_prediction(cb)` | `(epoch, epoch_total, batch_idx, batch_total, data_batch, y_hat)` |
| `on_batch_end(cb, with_metrics=False)` | `(epoch, epoch_total, batch_idx, batch_total, metrics_or_none)` |

---

### `DeeplearningTrain` (`deeplearning_train.py`)

High-level trainer combining a `self.trainer` (`DeeplearningBase` with optimizer) and `self.validator` (`DeeplearningBase` without optimizer).

#### Constructor
```python
DeeplearningTrain(config: PipelineConfig, model: nn.Module, dp: DataProvider, draft: bool = False)
```
- Creates `self.optim` from `config.optimizer` (`"adamw"`, `"sgd"`, `"adam"`) and `config.learning_rate`.
- Initializes `self.history = {"train": [], "val": []}`.

#### Key Methods
- `add_metric(name, criterion=None, minimize=True, backward=False)`: registers the metric on both `trainer` and `validator`.
- `add_early_stopping(max_attempts=3, min_gain=0.01, metric="loss", minimize=None)`: attaches an `EarlyStopping` instance monitoring `val_metrics[metric]`. When `minimize is None`, auto-infers `minimize = "loss" in metric.lower()`.
- `on_prediction / on_batch_start / on_batch_end / on_epoch_start / on_epoch_end(..., phase="both")`: forwards callbacks to `"train"`, `"val"`, or `"both"`. *(Note: `on_batch_start` does not currently forward `with_data`.)*
- `sanity_check(must_backward=True)`: verifies at least one metric (and one `backward=True` metric when `must_backward=True`) is registered.
- `train(epoch_total: int) -> dict`: runs `epoch_total` epochs (forced to `1` when `draft=True`) of train + validate, appends to `self.history`, and checks early stopping.
- `validate(epoch=0, epoch_total=1) -> dict`: runs one validation epoch on `dp.val`.
- `inference(dataloader=None, compute_cm=False, force_eval=True, on_batch_end=None) -> tuple[float, float | None, float, np.ndarray | None]`:
  Standalone `torch.no_grad()` evaluation returning `(avg_loss, accuracy, avg_forward_time_per_sample, confusion_matrix)`.

---

### `EarlyStopping` (`deeplearning_train.py`)

```python
EarlyStopping(max_attempts: int = 3, min_gain: float = 0.01, minimize: bool = False)
```
- `should_stop(metric) -> bool`: records the first observed value as baseline (`metric_last` initializes to `None`, returning `False`). On subsequent calls, computes `gain = self.metric_last - metric` when `minimize=True` or `metric - self.metric_last` when `minimize=False`, resets `num_attempts` when `gain >= min_gain`, and returns `True` when `num_attempts >= max_attempts`.

---

### `DeeplearningEval` (`deeplearning_eval.py`)

```python
DeeplearningEval(config: PipelineConfig, model: nn.Module, dl: DataLoader, draft: bool = False)
```
- Wraps a single `self.evaluator = DeeplearningBase(config, model, dl, draft=draft)` (`optimizer=None`).
- Exposes `add_metric`, all five callback registration methods (including `on_batch_start(..., with_data=False)`), `sanity_check()`, and `evaluate(epoch_total: int)` (returns `self.history: list[dict[str, float]]`).

---

## Train & Eval Loops Outside `DeeplearningTrain`

Reference inventory of all training, pretraining, mask-optimization, and evaluation loops in `positronixlab/` that currently run outside `DeeplearningTrain` / `DeeplearningEval` (tracked for future `metamodel` / `deeplearning` trainer unification):

| Module & Entry Point | Loop Kind | Optimization / Eval Details | Refactor Notes |
|---|---|---|---|
| `positronixlab/baselines/lepori/models.py:pretrain_simclr` | Train + Val (self-supervised contrastive) | `AdamW(lr=5e-4, weight_decay=0.0)` + `CosineAnnealingLR(eta_min=lr/50)` on `SimCLRLoss(temperature=0.07)`; selects best checkpoint by `val_top5`; saves `backbone.pt` & `mlp.pt`. | Prime candidate to unify once `DeeplearningTrain` supports LR schedulers, multi-view inputs, and custom checkpoint-monitor metrics. |
| `positronixlab/baselines/neurosurgeon/discoverer.py:NeuroSurgeonDiscoverer.discover` | Mask Train (frozen base model) | `Adam` over `wrapped.mask_parameters()` minimizing `task_loss + l0_lambda * l0_loss` with exponential temperature annealing (`max_temp ** (1 / epochs)`). | Mask-only optimization loop; shares the same batch/loss pattern as `CsordasGumbelDiscoverer`. |
| `positronixlab/baselines/csordas/discoverer.py:CsordasGumbelDiscoverer.discover` | Mask Train (frozen base model) | `Adam` over Gumbel-Sigmoid mask logits minimizing `task_loss + (mask_loss_weight / B) * sum(logits)`. | Mask-only optimization loop; shares batch/loss pattern with `NeuroSurgeonDiscoverer`. |
| `positronixlab/tasks/benchmarks.py:ModularityBenchmark.evaluate_mask` | Eval (`torch.no_grad()`) | Evaluates a `model` under `MaskApplier` (`retain`, `zero_ablate`, `random_ablate`, `randomly_sampled`, `complement_sampled`) over a dataloader, returning loss, accuracy, and optional confusion matrix. | Could delegate to a shared `DeeplearningEval` / evaluator once `loss_fn` & non-`MetaModel` `SOTAModel` contracts are unified. |
| `positronixlab/tasks/atlas_discoverer.py:AtlasDiscoverer.discover` / `discover_subtasks` | Forward collection + decoupled space optimization | Runs `torch.no_grad()` forward passes through `ModularObserver` to collect activations, then steps `LayerSpace` + K-Means. | Observation-only on the main model; main weights are never updated. |
| `positronixlab/modularspace/layerspace.py` & `modularspace.py` | Decoupled space-position optimization | Per-layer `Adam` optimizing `positions: [num_cus, modular_space_size]` on detached activation scores (`loss.backward()`, `optimizer.step()`). | Intentionally decoupled from `DeeplearningTrain` (separate graph & optimizer per layer). |
| `positronixlab/tasks/compositionality/weightmask_check.py:train_weight_masks` & `compositionality.py:evaluate_masked_runs` | Legacy prototype Train + Eval | Earlier prototype mask-training (`Adam`) and masked-evaluation loops superseded by `CsordasGumbelDiscoverer` and `ModularityBenchmark`. | Candidate for deprecation/removal during refactor. |
