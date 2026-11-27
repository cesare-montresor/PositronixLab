"""
Minimal runnable CPU example for `positronixlab.deeplearning`.

Demonstrates:
- Composing `TrainConfig` with a `task` field
- Training & validating a small MLP with `DeeplearningTrain`
- Registering multiple metrics (loss + accuracy) and epoch callbacks
- Running standalone evaluation with `DeeplearningEval` and `inference()`

Run:
    .venv/bin/python positronixlab/deeplearning/examples/example_deeplearning_train.py
"""

import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _ensure_parallelazy_if_missing() -> None:
    """Allow running this standalone CPU example even when `parallelazy` is not accessible."""
    try:
        import parallelazy  # noqa: F401
        from parallelazy.plugin_data import ManagerOptions  # noqa: F401
        return
    except ImportError:
        pass

    @dataclass
    class _StubManagerOptions:
        work_dir: str | None = None
        log_dir: str | None = None
        display_logs: bool = True
        write_logs: bool = True

    pkg = types.ModuleType("parallelazy")
    pkg.PluginManager = object
    pkg.is_debugpy_attached = lambda: False
    plugin_data = types.ModuleType("parallelazy.plugin_data")
    plugin_data.ManagerOptions = _StubManagerOptions
    pkg.plugin_data = plugin_data
    sys.modules["parallelazy"] = pkg
    sys.modules["parallelazy.plugin_data"] = plugin_data


_ensure_parallelazy_if_missing()

from positronixlab.deeplearning import DeeplearningEval, DeeplearningTrain, TrainConfig


@dataclass
class ExampleConfig(TrainConfig):
    """Compose TrainConfig with `task` (required by DeeplearningBase & inference)."""

    task: str = "classification"


class SimpleMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.device = torch.device("cpu")
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self._criterion = nn.CrossEntropyLoss()
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def loss(self, x: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        return self._criterion(y_hat, y)


def accuracy(x: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    return (y_hat.argmax(dim=1) == y).float().mean()


def make_synthetic_data(n_samples: int = 300, in_dim: int = 8, num_classes: int = 3):
    torch.manual_seed(42)
    centers = torch.randn(num_classes, in_dim) * 2.0
    labels = torch.randint(0, num_classes, (n_samples,))
    features = centers[labels] + torch.randn(n_samples, in_dim) * 0.6

    train_ds = TensorDataset(features[:200], labels[:200])
    val_ds = TensorDataset(features[200:250], labels[200:250])
    test_ds = TensorDataset(features[250:], labels[250:])

    return (
        DataLoader(train_ds, batch_size=32, shuffle=True),
        DataLoader(val_ds, batch_size=32, shuffle=False),
        DataLoader(test_ds, batch_size=32, shuffle=False),
    )


def main():
    cfg = ExampleConfig(optimizer="adamw", learning_rate=0.02, epochs=5)
    train_dl, val_dl, test_dl = make_synthetic_data()
    dp = SimpleNamespace(train=train_dl, val=val_dl)

    model = SimpleMLP(in_dim=8, hidden_dim=32, num_classes=3)

    # 1. Train & validate
    trainer = DeeplearningTrain(config=cfg, model=model, dp=dp)
    trainer.add_metric("loss", criterion=model.loss, minimize=True, backward=True)
    trainer.add_metric("acc", criterion=accuracy, minimize=False, backward=False)

    trainer.on_epoch_end(
        lambda ep, tot, m: print(f"  [callback] train epoch {ep + 1}/{tot}: loss={m['loss']:.4f}, acc={m['acc']:.3f}"),
        phase="train",
    )

    history = trainer.train(epoch_total=cfg.epochs)
    best_idx = trainer.validator.metrics["acc"]["best_idx"]
    print(f"\nBest validation epoch: {best_idx + 1} (acc={history['val'][best_idx]['acc']:.3f})")

    # 2. Evaluate on test split with DeeplearningEval
    evaluator = DeeplearningEval(config=cfg, model=model, dl=test_dl)
    evaluator.add_metric("loss", criterion=model.loss, minimize=True)
    evaluator.add_metric("acc", criterion=accuracy, minimize=False)
    eval_history = evaluator.evaluate(epoch_total=1)
    print(f"DeeplearningEval test metrics: {eval_history[0]}")

    # 3. Standalone inference with confusion matrix
    _, inf_acc, sec_per_sample, cm = trainer.inference(dataloader=test_dl, compute_cm=True)
    print(f"Inference accuracy: {inf_acc:.3f} ({sec_per_sample * 1e3:.3f} ms/sample)")
    print(f"Confusion matrix:\n{cm}")


if __name__ == "__main__":
    main()
