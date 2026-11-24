"""Double-Addition (TupleArithmetic) synthetic dataset (Csordas et al., ICLR 2021).

Tests functional reuse (P_reuse): performing the same operation (modulo-100
addition by default) twice using separate input and output units:
inputs ``(a, b)`` and ``(c, d)`` -> outputs ``(a + b) % 100`` and ``(c + d) % 100``.

Deterministic RNG parity with ``are_neural_nets_modular/dataset/tuple_arithmetic.py``:
- ``RandomState(0x12345678 + set_index)`` with ``SETS = ["train", "test", "valid"]``
- Default ``n_digits = 2, n_tuples = 2`` -> 80-D one-hot input, 40-D (2 x 2 x 10) output logits.
- When ``restrict_tuple`` is set (e.g. ``0`` or ``1``), the non-selected tuple's
  input slice is zeroed out (matching ``FFTupleArithmeticDatasetInterface``).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from positronixlab.dataprovider.addmul_dataprovider import _split_digits
from positronixlab.dataprovider.dataprovider import DataProvider


class TupleArithmeticLoss(nn.Module):
    """Per-digit cross-entropy for ``TupleArithmetic`` with optional single-tuple restriction."""

    def __init__(self, n_tuples: int = 2, n_digits: int = 2, restrict_tuple: Optional[int] = None) -> None:
        super().__init__()
        self.n_tuples = n_tuples
        self.n_digits = n_digits
        self.restrict_tuple = restrict_tuple

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, n_tuples * n_digits * 10), targets: (B, n_tuples, n_digits)
        reshaped = logits.reshape(logits.shape[0], self.n_tuples, self.n_digits, 10)
        ref = targets.long()
        if self.restrict_tuple is not None and self.restrict_tuple >= 0:
            reshaped = reshaped[:, self.restrict_tuple : self.restrict_tuple + 1]
            ref = ref[:, self.restrict_tuple : self.restrict_tuple + 1]
        return F.cross_entropy(reshaped.reshape(-1, 10), ref.reshape(-1))


def tuple_arithmetic_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    n_tuples: int = 2,
    n_digits: int = 2,
) -> Dict[str, float]:
    """Return total and per-tuple exact-match accuracy."""
    preds = logits.reshape(logits.shape[0], n_tuples, n_digits, 10).argmax(dim=-1)
    tuple_ok = (preds == targets.long()).all(dim=-1)  # (B, n_tuples)
    res: Dict[str, float] = {"total": float(tuple_ok.all(dim=-1).float().mean().item())}
    for t in range(n_tuples):
        res[f"tuple_{t}"] = float(tuple_ok[:, t].float().mean().item())
    return res


class TupleArithmeticDataset(Dataset):
    """Synthetic double-addition / multi-tuple arithmetic dataset."""

    SETS: Tuple[str, str, str] = ("train", "test", "valid")
    _CACHE: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def __init__(
        self,
        split: str = "train",
        n_digits: int = 2,
        n_tuples: int = 2,
        n_samples: int = 100_000,
        op: str = "add",
        restrict_tuple: Optional[int] = None,
    ) -> None:
        super().__init__()
        if split not in self.SETS:
            raise ValueError(f"split must be one of {self.SETS}, got {split!r}")
        if op not in ("add", "mul"):
            raise ValueError(f"op must be 'add' or 'mul', got {op!r}")
        if restrict_tuple is not None and not (0 <= restrict_tuple < n_tuples):
            raise ValueError(f"restrict_tuple must be in [0, {n_tuples}), got {restrict_tuple}")

        self.split = split
        self.n_digits = n_digits
        self.n_tuples = n_tuples
        self.n_samples = n_samples
        self.op = op
        self.restrict_tuple = restrict_tuple
        self.maxnum = 10**n_digits

        cache_key = f"{split}_{n_digits}_{n_tuples}_{n_samples}_{op}"
        if cache_key not in self._CACHE:
            rng = np.random.RandomState(0x12345678 + self.SETS.index(split))
            i_list = [rng.randint(0, self.maxnum, (n_samples, 1, 2)) for _ in range(n_tuples)]
            inputs = np.concatenate(i_list, axis=1)  # (N, n_tuples, 2)
            if op == "add":
                outputs = (inputs[:, :, 0] + inputs[:, :, 1]) % self.maxnum
            else:
                outputs = (inputs[:, :, 0] * inputs[:, :, 1]) % self.maxnum
            self._CACHE[cache_key] = (inputs, outputs)

        self.inputs, self.outputs = self._CACHE[cache_key]
        self.in_features = n_tuples * 2 * n_digits * 10
        self.out_features = n_tuples * n_digits * 10

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        in_digits = torch.from_numpy(_split_digits(self.n_digits, self.inputs[idx])).long()
        out_digits = torch.from_numpy(_split_digits(self.n_digits, self.outputs[idx])).long()

        # Shape: (n_tuples, 2, n_digits, 10)
        onehot_in = F.one_hot(in_digits, num_classes=10).float()
        if self.restrict_tuple is not None:
            mask = torch.zeros(self.n_tuples, 1, 1, 1, dtype=onehot_in.dtype)
            mask[self.restrict_tuple] = 1.0
            onehot_in = onehot_in * mask
        return onehot_in.reshape(-1), out_digits


class TupleArithmeticDataProvider(DataProvider):
    """Unified PositronixLab ``DataProvider`` for Csordas et al.'s Double-Addition task."""

    def __init__(
        self,
        n_digits: int = 2,
        n_tuples: int = 2,
        n_train_samples: int = 100_000,
        n_val_samples: int = 10_000,
        n_test_samples: int = 10_000,
        op: str = "add",
        restrict_tuple: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.name = "tuple_arithmetic"
        self.n_digits = n_digits
        self.n_tuples = n_tuples
        self.n_train_samples = n_train_samples
        self.n_val_samples = n_val_samples
        self.n_test_samples = n_test_samples
        self.op = op
        self.restrict_tuple = restrict_tuple

        self.num_features = n_tuples * 2 * n_digits * 10
        self.num_classes = n_tuples * n_digits * 10
        self.x_shape = (self.num_features,)
        self.y_shape = (n_tuples, n_digits)
        self.criterion = TupleArithmeticLoss(
            n_tuples=n_tuples, n_digits=n_digits, restrict_tuple=restrict_tuple
        )

        self.dataset_train = TupleArithmeticDataset(
            split="train",
            n_digits=n_digits,
            n_tuples=n_tuples,
            n_samples=n_train_samples,
            op=op,
            restrict_tuple=restrict_tuple,
        )
        self.dataset_val = TupleArithmeticDataset(
            split="valid",
            n_digits=n_digits,
            n_tuples=n_tuples,
            n_samples=n_val_samples,
            op=op,
            restrict_tuple=restrict_tuple,
        )
        self.dataset_test = TupleArithmeticDataset(
            split="test",
            n_digits=n_digits,
            n_tuples=n_tuples,
            n_samples=n_test_samples,
            op=op,
            restrict_tuple=restrict_tuple,
        )
        self.dataset = self.dataset_train

    def with_restrict_tuple(self, restrict_tuple: Optional[int]) -> "TupleArithmeticDataProvider":
        """Return a new provider restricted to a single tuple index (e.g. ``0`` or ``1``)."""
        return TupleArithmeticDataProvider(
            n_digits=self.n_digits,
            n_tuples=self.n_tuples,
            n_train_samples=self.n_train_samples,
            n_val_samples=self.n_val_samples,
            n_test_samples=self.n_test_samples,
            op=self.op,
            restrict_tuple=restrict_tuple,
            batch_size=self.batch_size,
            split=self.dataset_split,
            num_workers=self.num_workers,
            transforms=self.transforms,
            indexed_dataset=self.indexed_dataset,
        )
