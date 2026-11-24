# DeepLearning Usage Guide

Practical recipes for configuring and running `DeeplearningTrain`, `DeeplearningEval`, `DeeplearningBase`, and `TrainConfig`.

---

## 1. Composing a Config

`TrainConfig` is a composable `BaseConfig` mixin. Because `DeeplearningBase` and `DeeplearningTrain.inference()` read `config.task`, compose `TrainConfig` with `DataConfig` (or a small config defining `task`):

```python
from dataclasses import dataclass
from positronixlab.deeplearning import TrainConfig

@dataclass
class ExperimentConfig(TrainConfig):
    task: str = "classification"

cfg = ExperimentConfig(optimizer="adamw", learning_rate=1e-3, epochs=5)
```

With `positronixlab.dataprovider.config.DataConfig`:

```python
from dataclasses import dataclass
from positronixlab.dataprovider.config import DataConfig
from positronixlab.deeplearning import TrainConfig

@dataclass
class ExperimentConfig(TrainConfig, DataConfig):
    pass
```

---

## 2. Model Requirements

Any `nn.Module` can be used as long as it exposes `.device` (and `.loss(x, y, y_hat)` if you call `DeeplearningTrain.inference()`):

```python
import torch
from torch import nn

class SimpleClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )
        self._criterion = nn.CrossEntropyLoss()
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def loss(self, x: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        return self._criterion(y_hat, y)
```

---

## 3. Training & Validation with `DeeplearningTrain`

Register at least one metric with `backward=True` before calling `train()`. Register additional monitoring metrics (such as accuracy) with `backward=False`:

```python
from types import SimpleNamespace
from torch.utils.data import DataLoader, TensorDataset
from positronixlab.deeplearning import DeeplearningTrain

# DataProvider only needs .train and .val DataLoaders
dp = SimpleNamespace(
    train=DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True),
    val=DataLoader(TensorDataset(x_val, y_val), batch_size=32),
)

trainer = DeeplearningTrain(config=cfg, model=model, dp=dp)

# Primary training loss (backpropagated)
trainer.add_metric("loss", criterion=model.loss, minimize=True, backward=True)

# Additional classification accuracy metric (not backpropagated)
def accuracy(x, y, y_hat):
    return (y_hat.argmax(dim=1) == y).float().mean()

trainer.add_metric("acc", criterion=accuracy, minimize=False, backward=False)

history = trainer.train(epoch_total=cfg.epochs)
print("Final val metrics:", history["val"][-1])
```

### Best-epoch tracking
Each `DeeplearningBase` tracks metric history and best epoch index automatically:

```python
val_loss_info = trainer.validator.metrics["loss"]
best_epoch = val_loss_info["best_idx"]
best_loss = val_loss_info["history"][best_epoch]
```

---

## 4. Lifecycle Callbacks

Callbacks can be attached to `"train"`, `"val"`, or `"both"` (default):

```python
def log_epoch(epoch, epoch_total, metrics):
    print(f"Epoch {epoch + 1}/{epoch_total} finished: {dict(metrics)}")

def inspect_batch(epoch, epoch_total, batch_idx, batch_total, metrics):
    # metrics is populated only when with_metrics=True
    pass

def on_pred(epoch, epoch_total, batch_idx, batch_total, data_batch, y_hat):
    # Fires right after model(x), before loss computation
    pass

trainer.on_epoch_end(log_epoch, phase="val")
trainer.on_batch_end(inspect_batch, with_metrics=True, phase="train")
trainer.on_prediction(on_pred, phase="train")
```

> **Note:** Each hook holds a single callback; registering a second callback on the same hook replaces the first. If you need batch data in `on_batch_start` on `DeeplearningTrain`, attach it directly on `trainer.trainer.on_batch_start(cb, with_data=True)` (since `DeeplearningTrain.on_batch_start` does not currently forward `with_data`).

---

## 5. Standalone Evaluation (`DeeplearningEval` & `inference()`)

### Using `DeeplearningEval` (recommended for custom metrics & indexed datasets)
`DeeplearningEval` supports any custom metrics, all lifecycle hooks, and both `(x, y)` and `(x, y, idx)` batches:

```python
from positronixlab.deeplearning import DeeplearningEval

evaluator = DeeplearningEval(config=cfg, model=model, dl=test_loader)
evaluator.add_metric("loss", criterion=model.loss, minimize=True)
evaluator.add_metric("acc", criterion=accuracy, minimize=False)

eval_history = evaluator.evaluate(epoch_total=1)  # returns list[dict[str, float]]
print("Test metrics:", eval_history[0])
```

### Using `DeeplearningTrain.inference()` (accuracy, confusion matrix, timing)
```python
avg_loss, acc, sec_per_sample, cm = trainer.inference(
    dataloader=test_loader,
    compute_cm=True,
    force_eval=True,
)
print(f"Accuracy: {acc:.2%}, forward time: {sec_per_sample * 1e3:.3f} ms/sample")
print("Confusion matrix:\n", cm)
```

> **Known Caveats in `inference()`:**
> - Only supports 2-tuple `(x, y)` batches (not 3-tuple `(x, y, idx)`).
> - `avg_loss` currently divides the sum of per-batch mean losses by total sample count `N`, making it `~batch_size` times smaller than `DeeplearningEval`'s loss (see `analysis.md`, Bug B2). Use `DeeplearningEval` or `validate()` for comparable loss values.

---

## 6. Early Stopping

```python
# Recommended currently ONLY for positive maximized metrics (e.g. "acc")
trainer.add_early_stopping(max_attempts=3, min_gain=0.01, metric="acc")
```

> **Warning (Known Bug B1):** `EarlyStopping.should_stop()` currently **maximizes** its metric (`gain = metric - self.metric_last`) with `metric_last = 0`. Do **not** use the default `metric="loss"` until Bug B1 is fixed, as decreasing losses will trigger premature stopping.

---

## 7. Draft Mode (Fast Smoke Runs)

Pass `draft=True` to `DeeplearningTrain` or `DeeplearningEval` to cap execution to 1 epoch and ~20 batches per epoch (currently 21 due to an off-by-one — see `analysis.md`, Bug B7):

```python
trainer = DeeplearningTrain(config=cfg, model=model, dp=dp, draft=True)
```

---

## 8. Runnable Example

Run the bundled CPU example:

```bash
.venv/bin/python positronixlab/deeplearning/examples/example_deeplearning_train.py
```
