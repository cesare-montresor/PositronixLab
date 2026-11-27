import inspect
import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _ensure_parallelazy_stub() -> None:
    """Install a minimal `parallelazy` stub when unavailable (e.g. sandboxed CI)."""
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
    pkg.PluginManager = object  # type: ignore[attr-defined]
    pkg.is_debugpy_attached = lambda: False  # type: ignore[attr-defined]
    plugin_data = types.ModuleType("parallelazy.plugin_data")
    plugin_data.ManagerOptions = _StubManagerOptions  # type: ignore[attr-defined]
    pkg.plugin_data = plugin_data  # type: ignore[attr-defined]
    sys.modules["parallelazy"] = pkg
    sys.modules["parallelazy.plugin_data"] = plugin_data


_ensure_parallelazy_stub()

from positronixlab.deeplearning import (
    DeeplearningBase,
    DeeplearningEval,
    DeeplearningTrain,
    EarlyStopping,
    TrainConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class TaskTrainConfig(TrainConfig):
    """TrainConfig composed with `task` (required by DeeplearningBase & inference)."""

    task: str = "classification"


class TinyClassifier(nn.Module):
    def __init__(self, in_dim: int = 4, num_classes: int = 3):
        super().__init__()
        self.device = torch.device("cpu")
        self.fc = nn.Linear(in_dim, num_classes)
        self._ce = nn.CrossEntropyLoss()
        self.last_forward_grad_enabled: bool | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_forward_grad_enabled = torch.is_grad_enabled()
        return self.fc(x)

    def loss(self, x: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        return self._ce(y_hat, y)


class TinyRegressor(nn.Module):
    def __init__(self, in_dim: int = 4):
        super().__init__()
        self.device = torch.device("cpu")
        self.fc = nn.Linear(in_dim, 1)
        self._mse = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    def loss(self, x: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        return self._mse(y_hat, y)


def _make_classification_loaders(n_train: int = 32, n_val: int = 16, batch_size: int = 8, indexed: bool = False):
    torch.manual_seed(42)
    x_tr = torch.randn(n_train, 4)
    y_tr = torch.randint(0, 3, (n_train,))
    x_va = torch.randn(n_val, 4)
    y_va = torch.randint(0, 3, (n_val,))

    if indexed:
        tr_ds = TensorDataset(x_tr, y_tr, torch.arange(n_train))
        va_ds = TensorDataset(x_va, y_va, torch.arange(n_val))
    else:
        tr_ds = TensorDataset(x_tr, y_tr)
        va_ds = TensorDataset(x_va, y_va)

    return (
        DataLoader(tr_ds, batch_size=batch_size, shuffle=False),
        DataLoader(va_ds, batch_size=batch_size, shuffle=False),
    )


def _acc_metric(x: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    return (y_hat.argmax(dim=1) == y).float().mean()


# ---------------------------------------------------------------------------
# TrainConfig tests
# ---------------------------------------------------------------------------


class TestTrainConfig:
    def test_defaults(self):
        cfg = TrainConfig()
        assert cfg.name == "default_config"
        assert cfg.optimizer == "adamw"
        assert cfg.learning_rate == pytest.approx(0.001)
        assert cfg.epochs == 10
        assert cfg.train_epochs == 10

    def test_serialization_and_updates(self, tmp_path):
        cfg = TrainConfig(optimizer="sgd", learning_rate=0.05, epochs=25, train_epochs=25)
        d = cfg.to_dict()
        restored = TrainConfig.from_dict(d)
        assert restored == cfg
        assert restored.signature() == cfg.signature()

        updated = cfg.with_updates(learning_rate=0.01, ignored_key=123)
        assert updated.learning_rate == pytest.approx(0.01)
        assert updated.optimizer == "sgd"
        assert cfg.learning_rate == pytest.approx(0.05)  # immutable update

        path = tmp_path / "train_cfg.json"
        cfg.save(str(path))
        loaded = TrainConfig.load(str(path))
        assert loaded == cfg

    def test_from_dict_coercion(self):
        cfg = TrainConfig.from_dict({"optimizer": "adam", "learning_rate": "0.002", "epochs": "7"})
        assert cfg.optimizer == "adam"
        assert cfg.learning_rate == pytest.approx(0.002)
        assert cfg.epochs == 7


# ---------------------------------------------------------------------------
# DeeplearningBase tests
# ---------------------------------------------------------------------------


class TestDeeplearningBase:
    def test_is_training_property(self):
        cfg = TaskTrainConfig()
        model = TinyClassifier()
        tr_dl, _ = _make_classification_loaders()

        eval_base = DeeplearningBase(cfg, model, tr_dl, optimizer=None)
        assert eval_base.is_training is False

        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        train_base = DeeplearningBase(cfg, model, tr_dl, optimizer=opt)
        assert train_base.is_training is True

    def test_process_batch_updates_weights_only_when_training(self):
        torch.manual_seed(0)
        cfg = TaskTrainConfig()
        tr_dl, _ = _make_classification_loaders(n_train=16, batch_size=8)
        batch = next(iter(tr_dl))

        # Eval mode: weights unchanged
        eval_model = TinyClassifier()
        w_before = eval_model.fc.weight.detach().clone()
        eval_base = DeeplearningBase(cfg, eval_model, tr_dl, optimizer=None)
        eval_base.add_metric("loss", criterion=eval_model.loss, minimize=True, backward=True)
        eval_metrics = eval_base.process_batch(0, 1, 0, 2, batch)
        assert "loss" in eval_metrics
        assert torch.equal(eval_model.fc.weight.detach(), w_before)

        # Train mode: weights change
        train_model = TinyClassifier()
        w_before = train_model.fc.weight.detach().clone()
        opt = torch.optim.SGD(train_model.parameters(), lr=0.5)
        train_base = DeeplearningBase(cfg, train_model, tr_dl, optimizer=opt)
        train_base.add_metric("loss", criterion=train_model.loss, minimize=True, backward=True)
        train_metrics = train_base.process_batch(0, 1, 0, 2, batch)
        assert "loss" in train_metrics
        assert not torch.equal(train_model.fc.weight.detach(), w_before)

    def test_process_batch_supports_three_tuple_indexed_batches(self):
        cfg = TaskTrainConfig()
        model = TinyClassifier()
        tr_dl, _ = _make_classification_loaders(n_train=16, batch_size=8, indexed=True)
        base = DeeplearningBase(cfg, model, tr_dl, optimizer=None)
        base.add_metric("loss", criterion=model.loss)

        epoch_metrics, last_idx, batch_total = base.process_epoch(tr_dl, epoch=0, epoch_total=1)
        assert "loss" in epoch_metrics
        assert last_idx == 1
        assert batch_total == 2

    def test_metric_history_and_best_idx_tracking(self):
        cfg = TaskTrainConfig()
        model = TinyClassifier()
        tr_dl, _ = _make_classification_loaders(n_train=8, batch_size=8)
        base = DeeplearningBase(cfg, model, tr_dl, optimizer=None)

        loss_values = iter([0.9, 0.4, 0.6])
        acc_values = iter([0.5, 0.8, 0.7])
        base.add_metric("loss", criterion=lambda x, y, yh: torch.tensor(next(loss_values)), minimize=True)
        base.add_metric("acc", criterion=lambda x, y, yh: torch.tensor(next(acc_values)), minimize=False)

        for epoch in range(3):
            base.process_epoch(tr_dl, epoch=epoch, epoch_total=3)

        assert base.metrics["loss"]["history"] == pytest.approx([0.9, 0.4, 0.6])
        assert base.metrics["loss"]["best_idx"] == 1
        assert base.metrics["acc"]["history"] == pytest.approx([0.5, 0.8, 0.7])
        assert base.metrics["acc"]["best_idx"] == 1

    def test_callbacks_and_flags(self):
        cfg = TaskTrainConfig()
        model = TinyClassifier()
        tr_dl, _ = _make_classification_loaders(n_train=16, batch_size=8)
        base = DeeplearningBase(cfg, model, tr_dl, optimizer=None)
        base.add_metric("loss", criterion=model.loss)

        events = []
        base.on_epoch_start(lambda ep, tot: events.append(("epoch_start", ep, tot)))
        base.on_batch_start(lambda ep, tot, bi, bt, data: events.append(("batch_start", bi, data is not None)), with_data=True)
        base.on_prediction(lambda ep, tot, bi, bt, data, yh: events.append(("pred", bi, yh.shape)))
        base.on_batch_end(lambda ep, tot, bi, bt, m: events.append(("batch_end", bi, m is not None and "loss" in m)), with_metrics=True)
        base.on_epoch_end(lambda ep, tot, m: events.append(("epoch_end", ep, "loss" in m)))

        base.process_epoch(tr_dl, epoch=0, epoch_total=1)

        assert events[0] == ("epoch_start", 0, 1)
        assert ("batch_start", 0, True) in events
        assert ("pred", 0, torch.Size([8, 3])) in events
        assert ("batch_end", 0, True) in events
        assert events[-1] == ("epoch_end", 0, True)


# ---------------------------------------------------------------------------
# DeeplearningTrain & EarlyStopping tests
# ---------------------------------------------------------------------------


class TestDeeplearningTrain:
    @pytest.mark.parametrize(
        "opt_name,opt_cls",
        [
            ("adamw", torch.optim.AdamW),
            ("sgd", torch.optim.SGD),
            ("adam", torch.optim.Adam),
        ],
    )
    def test_optimizer_selection(self, opt_name, opt_cls):
        cfg = TaskTrainConfig(optimizer=opt_name, learning_rate=0.01)
        tr_dl, va_dl = _make_classification_loaders()
        dp = SimpleNamespace(train=tr_dl, val=va_dl)
        trainer = DeeplearningTrain(cfg, TinyClassifier(), dp)
        assert isinstance(trainer.optim, opt_cls)

    def test_invalid_optimizer_raises(self):
        cfg = TaskTrainConfig(optimizer="rmsprop")
        tr_dl, va_dl = _make_classification_loaders()
        dp = SimpleNamespace(train=tr_dl, val=va_dl)
        with pytest.raises(ValueError, match="Unknown optimizer"):
            DeeplearningTrain(cfg, TinyClassifier(), dp)

    def test_sanity_checks(self):
        cfg = TaskTrainConfig()
        tr_dl, va_dl = _make_classification_loaders()
        dp = SimpleNamespace(train=tr_dl, val=va_dl)
        model = TinyClassifier()
        trainer = DeeplearningTrain(cfg, model, dp)

        with pytest.raises(ValueError, match="No metrics found"):
            trainer.sanity_check(must_backward=False)

        trainer.add_metric("acc", criterion=_acc_metric, minimize=False, backward=False)
        with pytest.raises(ValueError, match="backward=True"):
            trainer.sanity_check(must_backward=True)

        # validate() should succeed without a backward metric
        val_metrics = trainer.validate()
        assert "acc" in val_metrics

    def test_train_and_validate_history(self):
        cfg = TaskTrainConfig(optimizer="adam", learning_rate=0.05)
        tr_dl, va_dl = _make_classification_loaders(n_train=32, n_val=16, batch_size=8)
        dp = SimpleNamespace(train=tr_dl, val=va_dl)
        model = TinyClassifier()
        trainer = DeeplearningTrain(cfg, model, dp)
        trainer.add_metric("loss", criterion=model.loss, minimize=True, backward=True)
        trainer.add_metric("acc", criterion=_acc_metric, minimize=False, backward=False)

        history = trainer.train(epoch_total=2)
        assert len(history["train"]) == 2
        assert len(history["val"]) == 2
        assert set(history["train"][0].keys()) == {"loss", "acc"}
        assert set(history["val"][0].keys()) == {"loss", "acc"}

    def test_draft_mode_forces_single_epoch(self):
        cfg = TaskTrainConfig()
        tr_dl, va_dl = _make_classification_loaders()
        dp = SimpleNamespace(train=tr_dl, val=va_dl)
        model = TinyClassifier()
        trainer = DeeplearningTrain(cfg, model, dp, draft=True)
        trainer.add_metric("loss", criterion=model.loss, backward=True)

        history = trainer.train(epoch_total=5)
        assert len(history["train"]) == 1
        assert len(history["val"]) == 1

    def test_phase_callback_routing(self):
        cfg = TaskTrainConfig()
        tr_dl, va_dl = _make_classification_loaders(n_train=8, n_val=8, batch_size=8)
        dp = SimpleNamespace(train=tr_dl, val=va_dl)
        model = TinyClassifier()
        trainer = DeeplearningTrain(cfg, model, dp)
        trainer.add_metric("loss", criterion=model.loss, backward=True)

        train_epochs = []
        val_epochs = []
        both_starts = []
        trainer.on_epoch_end(lambda ep, tot, m: train_epochs.append(ep), phase="train")
        trainer.on_epoch_end(lambda ep, tot, m: val_epochs.append(ep), phase="val")
        trainer.on_epoch_start(lambda ep, tot: both_starts.append(ep), phase="both")

        trainer.train(epoch_total=2)
        assert train_epochs == [0, 1]
        assert val_epochs == [0, 1]
        assert len(both_starts) == 4  # 2 train + 2 val

    def test_inference_classification_and_cm(self):
        cfg = TaskTrainConfig(task="classification")
        tr_dl, va_dl = _make_classification_loaders(n_train=16, n_val=16, batch_size=8)
        dp = SimpleNamespace(train=tr_dl, val=va_dl)
        model = TinyClassifier()
        model.train()

        trainer = DeeplearningTrain(cfg, model, dp)
        batch_callbacks = []
        avg_loss, acc, fwd_time, cm = trainer.inference(
            dataloader=va_dl,
            compute_cm=True,
            force_eval=True,
            on_batch_end=lambda bi, bt, l, a: batch_callbacks.append((bi, bt, l, a)),
        )

        assert isinstance(avg_loss, float)
        assert 0.0 <= acc <= 1.0
        assert fwd_time >= 0.0
        assert isinstance(cm, np.ndarray)
        assert cm.sum() == 16
        assert len(batch_callbacks) == 2
        assert model.training is True  # restored after force_eval

    def test_inference_regression_returns_none_acc_and_cm(self):
        torch.manual_seed(0)
        cfg = TaskTrainConfig(task="regression")
        x = torch.randn(16, 4)
        y = torch.randn(16, 1)
        dl = DataLoader(TensorDataset(x, y), batch_size=8)
        dp = SimpleNamespace(train=dl, val=dl)
        model = TinyRegressor()
        trainer = DeeplearningTrain(cfg, model, dp)

        avg_loss, acc, fwd_time, cm = trainer.inference(dataloader=dl, compute_cm=True)
        assert isinstance(avg_loss, float)
        assert acc is None
        assert cm is None
        assert fwd_time >= 0.0


class TestEarlyStopping:
    def test_maximizing_positive_metric_normal_flow(self):
        es = EarlyStopping(max_attempts=2, min_gain=0.05)
        assert es.should_stop(0.50) is False  # baseline -> reset
        assert es.num_resets == 1
        assert es.should_stop(0.60) is False  # gain +0.10 -> reset
        assert es.num_resets == 2
        assert es.should_stop(0.61) is False  # attempt 1/2
        assert es.should_stop(0.61) is True   # attempt 2/2 (>= max_attempts=2)
        assert es.total_attempts == 4


# ---------------------------------------------------------------------------
# DeeplearningEval tests
# ---------------------------------------------------------------------------


class TestDeeplearningEval:
    def test_sanity_check_requires_metric(self):
        cfg = TaskTrainConfig()
        _, va_dl = _make_classification_loaders()
        evaluator = DeeplearningEval(cfg, TinyClassifier(), va_dl)
        with pytest.raises(ValueError, match="No metrics found"):
            evaluator.evaluate(epoch_total=1)

    def test_evaluate_returns_history_and_fires_callbacks(self):
        cfg = TaskTrainConfig()
        _, va_dl = _make_classification_loaders(n_val=16, batch_size=8)
        model = TinyClassifier()
        evaluator = DeeplearningEval(cfg, model, va_dl)
        evaluator.add_metric("loss", criterion=model.loss, minimize=True)
        evaluator.add_metric("acc", criterion=_acc_metric, minimize=False)

        seen_data = []
        evaluator.on_batch_start(lambda ep, tot, bi, bt, data: seen_data.append(data is not None), with_data=True)

        history = evaluator.evaluate(epoch_total=2)
        assert isinstance(history, list)
        assert len(history) == 2
        assert set(history[0].keys()) == {"loss", "acc"}
        assert seen_data == [True, True, True, True]

    def test_evaluate_draft_forces_single_epoch(self):
        cfg = TaskTrainConfig()
        _, va_dl = _make_classification_loaders(n_val=16, batch_size=8)
        model = TinyClassifier()
        evaluator = DeeplearningEval(cfg, model, va_dl, draft=True)
        evaluator.add_metric("loss", criterion=model.loss)

        history = evaluator.evaluate(epoch_total=4)
        assert len(history) == 1


# ---------------------------------------------------------------------------
# Confirmed Bug Regression Tests (strict xfail until fixed)
# ---------------------------------------------------------------------------


class TestKnownBugs:
    def test_b1_early_stopping_decreasing_loss_should_not_stop(self):
        cfg = TaskTrainConfig()
        tr_dl, va_dl = _make_classification_loaders()
        dp = SimpleNamespace(train=tr_dl, val=va_dl)
        trainer = DeeplearningTrain(cfg, TinyClassifier(), dp)
        trainer.add_early_stopping(max_attempts=2, min_gain=0.01, metric="loss")

        decreasing_losses = [1.0, 0.8, 0.6, 0.4, 0.2]
        stopped = [trainer.early_stop.should_stop(v) for v in decreasing_losses]
        assert not any(stopped)

    def test_b1_early_stopping_negative_improving_metric(self):
        es = EarlyStopping(max_attempts=2, min_gain=0.1)
        improving_negative = [-10.0, -5.0, -2.0, -1.0, -0.5]
        stopped = [es.should_stop(v) for v in improving_negative]
        assert not any(stopped)

    def test_b1_early_stopping_patience_off_by_one(self):
        es = EarlyStopping(max_attempts=2, min_gain=0.05)
        assert es.should_stop(0.50) is False  # baseline
        assert es.should_stop(0.50) is False  # failure 1/2
        assert es.should_stop(0.50) is True   # failure 2/2 should stop

    def test_b2_inference_avg_loss_scaled_wrong_by_batch_size(self):
        class ConstantLossModel(TinyClassifier):
            def loss(self, x, y, y_hat):
                return torch.tensor(2.0)

        cfg = TaskTrainConfig(task="classification")
        tr_dl, va_dl = _make_classification_loaders(n_train=16, n_val=16, batch_size=8)
        dp = SimpleNamespace(train=tr_dl, val=va_dl)
        trainer = DeeplearningTrain(cfg, ConstantLossModel(), dp)

        avg_loss, _, _, _ = trainer.inference(dataloader=va_dl)
        assert avg_loss == pytest.approx(2.0)

    def test_b3_inference_three_tuple_indexed_batches(self):
        cfg = TaskTrainConfig(task="classification")
        tr_dl, va_dl = _make_classification_loaders(n_train=16, n_val=16, batch_size=8, indexed=True)
        dp = SimpleNamespace(train=tr_dl, val=va_dl)
        trainer = DeeplearningTrain(cfg, TinyClassifier(), dp)

        avg_loss, acc, _, _ = trainer.inference(dataloader=va_dl)
        assert isinstance(avg_loss, float)
        assert 0.0 <= acc <= 1.0

    @pytest.mark.xfail(
        strict=True,
        reason="Bug B4: DeeplearningBase.process_batch() runs self.model(x) outside torch.set_grad_enabled(self.is_training)",
    )
    def test_b4_process_batch_eval_disables_grad_during_forward(self):
        cfg = TaskTrainConfig()
        model = TinyClassifier()
        tr_dl, _ = _make_classification_loaders(n_train=8, batch_size=8)
        base = DeeplearningBase(cfg, model, tr_dl, optimizer=None)
        base.add_metric("loss", criterion=model.loss)

        batch = next(iter(tr_dl))
        base.process_batch(0, 1, 0, 1, batch)
        assert model.last_forward_grad_enabled is False

    @pytest.mark.xfail(
        strict=True,
        reason="Bug B5: DeeplearningBase.process_epoch() raises UnboundLocalError on an empty DataLoader",
    )
    def test_b5_process_epoch_empty_dataloader(self):
        cfg = TaskTrainConfig()
        model = TinyClassifier()
        empty_ds = TensorDataset(torch.empty(0, 4), torch.empty(0, dtype=torch.long))
        empty_dl = DataLoader(empty_ds, batch_size=8)
        base = DeeplearningBase(cfg, model, empty_dl, optimizer=None)
        base.add_metric("loss", criterion=model.loss)

        metrics, _, batch_total = base.process_epoch(empty_dl, epoch=0, epoch_total=1)
        assert batch_total == 0
        assert isinstance(metrics, dict)

    @pytest.mark.xfail(
        strict=True,
        reason="Bug B7: draft mode decrements draft_batches=20 and breaks on < 0, executing 21 batches instead of 20",
    )
    def test_b7_draft_mode_processes_20_batches(self):
        cfg = TaskTrainConfig()
        model = TinyClassifier()
        dl, _ = _make_classification_loaders(n_train=60, n_val=8, batch_size=2)  # 30 batches
        base = DeeplearningBase(cfg, model, dl, optimizer=None, draft=True)
        base.add_metric("loss", criterion=model.loss)

        seen = 0

        def _count(*args):
            nonlocal seen
            seen += 1

        base.on_batch_end(_count)
        base.process_epoch(dl, epoch=0, epoch_total=1, draft=True)
        assert seen == 20

    @pytest.mark.xfail(
        strict=True,
        reason="Bug B8: DeeplearningTrain.on_batch_start() omits the with_data parameter supported by DeeplearningBase",
    )
    def test_b8_train_on_batch_start_accepts_with_data(self):
        sig = inspect.signature(DeeplearningTrain.on_batch_start)
        assert "with_data" in sig.parameters

    @pytest.mark.xfail(
        strict=True,
        reason="Bug B9: Multiple backward=True metrics call loss.backward() without retain_graph or summing losses",
    )
    def test_b9_multiple_backward_metrics(self):
        cfg = TaskTrainConfig()
        model = TinyClassifier()
        tr_dl, _ = _make_classification_loaders(n_train=8, batch_size=8)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        base = DeeplearningBase(cfg, model, tr_dl, optimizer=opt)
        base.add_metric("loss_ce", criterion=model.loss, backward=True)
        base.add_metric("loss_l2", criterion=lambda x, y, yh: (yh ** 2).mean() * 0.01, backward=True)

        batch = next(iter(tr_dl))
        metrics = base.process_batch(0, 1, 0, 1, batch)
        assert "loss_ce" in metrics and "loss_l2" in metrics

    @pytest.mark.xfail(
        strict=True,
        reason="Config gap: DeeplearningBase.__init__ requires config.task, which standalone TrainConfig does not define",
    )
    def test_standalone_train_config_accepted_by_deeplearning_base(self):
        model = TinyClassifier()
        tr_dl, _ = _make_classification_loaders(n_train=8, batch_size=8)
        base = DeeplearningBase(TrainConfig(), model, tr_dl)
        assert base.is_training is False
