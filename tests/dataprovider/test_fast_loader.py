"""Unit tests for FastTensorLoader and vectorized dataset materialization."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from positronixlab.dataprovider import (
    AddMulDataProvider,
    FastTensorLoader,
    TupleArithmeticDataProvider,
    make_fast_loader,
    materialize_dataset_tensors,
)


class _ScalarLabelDataset(Dataset):
    """Mimics standard torchvision datasets returning (Tensor, int)."""

    def __init__(self, n: int = 16) -> None:
        self._x = torch.randn(n, 4)
        self._y = [i % 3 for i in range(n)]

    def __len__(self) -> int:
        return len(self._y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self._x[idx], self._y[idx]


def test_fast_tensor_loader_and_vectorized_materialization() -> None:
    dp_addmul = AddMulDataProvider(
        n_digits=1, target_mode="digits", n_train=64, n_val=16, n_test=16, batch_size=16, seed=0
    )
    xy = materialize_dataset_tensors(dp_addmul.train_loader().dataset)
    assert xy is not None
    x, y = xy
    assert x.shape == (64, dp_addmul.num_features)
    assert y.shape == (64, 1)

    loader = make_fast_loader(
        dp_addmul.train_loader(),
        device="cpu",
        batch_size=16,
        shuffle=True,
        max_steps=2,
        subtask="add",
    )
    assert isinstance(loader, FastTensorLoader)
    assert len(loader) == 2
    assert loader.subtask == "add"
    batches = list(loader)
    assert len(batches) == 2
    assert batches[0][0].shape == (16, dp_addmul.num_features)


def test_materialize_respects_transforms_on_arithmetic_datasets() -> None:
    dp_addmul = AddMulDataProvider(
        n_digits=1,
        target_mode="digits",
        n_train=16,
        n_val=8,
        n_test=8,
        batch_size=8,
        seed=0,
        transforms=lambda t: t * 2.0,
    )
    assert materialize_dataset_tensors(dp_addmul.train_loader().dataset) is None

    dp_tuple = TupleArithmeticDataProvider(
        n_digits=1,
        n_tuples=2,
        target_mode="digits",
        n_train=16,
        n_val=8,
        n_test=8,
        batch_size=8,
        seed=0,
        transforms=lambda t: t * 2.0,
    )
    assert materialize_dataset_tensors(dp_tuple.train_loader().dataset) is None


def test_make_fast_loader_handles_scalar_int_labels() -> None:
    ds = _ScalarLabelDataset(n=20)
    dl = DataLoader(ds, batch_size=8, shuffle=False)
    fast = make_fast_loader(dl, device="cpu", batch_size=8, shuffle=False)
    assert isinstance(fast, FastTensorLoader)
    assert len(fast) == 3
    bx, by = next(iter(fast))
    assert bx.shape == (8, 4)
    assert by.shape == (8,)
