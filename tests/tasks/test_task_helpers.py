"""Unit tests for task-aware accuracy and subtask criterion helpers."""

from __future__ import annotations

from positronixlab.baselines import DataProviderLoader, ModelLoader
from positronixlab.dataprovider import TupleArithmeticDataProvider
from positronixlab.tasks import make_accuracy_fn, resolve_subtask_criterion


def test_task_accuracy_fn_and_subtask_criterion_digits_mode() -> None:
    dp = TupleArithmeticDataProvider(
        op="add",
        n_digits=1,
        n_tuples=2,
        target_mode="digits",
        n_train=64,
        n_val=32,
        n_test=32,
        batch_size=16,
        seed=0,
    )
    kwargs = ModelLoader.infer_kwargs("csordas:mlp", dp, {"layer_sizes": [32]})
    model = ModelLoader.create("csordas:mlp", device="cpu", **kwargs)
    subtasks = DataProviderLoader.resolve_subtasks(dp)
    assert "tuple_0" in subtasks and "tuple_1" in subtasks

    crit = resolve_subtask_criterion(dp, "tuple_0")
    assert getattr(crit, "restrict_tuple", None) == 0

    acc_fn = make_accuracy_fn(dp, subtask="tuple_0")
    acc = acc_fn(model, dp.val_loader())
    assert 0.0 <= acc <= 1.0
