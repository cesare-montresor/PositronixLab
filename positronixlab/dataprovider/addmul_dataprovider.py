"""Addition / Multiplication synthetic dataset (Csordas et al., ICLR 2021).

Tests functional specialization (P_specialize): two distinct operations
(``add``, ``mul`` modulo ``10**n_digits``) sharing the exact same input and
output units, selected by a 2-way one-hot operation indicator.

Deterministic RNG parity with ``are_neural_nets_modular/dataset/double_op.py``:
- ``RandomState(0x12345678 + set_index)`` with ``SETS = ["train", "test", "valid"]``
- Least-significant-digit-first encoding (``split_digits``)
- Default ``n_digits = 2`` -> 42-D one-hot input, 20-D (2 x 10) output logits.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from positronixlab.dataprovider.dataprovider import DataProvider


def _split_digits(n_digits: int, nums: np.ndarray) -> np.ndarray:
    """Split integer array into least-significant-digit-first ``uint8`` digits."""
    digits = []
    cur = nums.copy()
    for _ in range(n_digits):
        digits.append(cur % 10)
        cur = cur // 10
    return np.stack(digits, axis=-1).astype(np.uint8)


class AddMulLoss(nn.Module):
    """Per-digit cross-entropy loss for ``AddMul`` (logits ``(B, n_digits*10)``, targets ``(B, n_digits)``)."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits.reshape(-1, 10), targets.long().reshape(-1))


def addmul_exact_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Exact-match accuracy across all output digits."""
    preds = logits.reshape(targets.shape[0], -1, 10).argmax(dim=-1)
    correct = (preds == targets.long()).all(dim=-1).float().mean()
    return float(correct.item())


class AddMulDataset(Dataset):
    """Synthetic modulo-``10**n_digits`` addition/multiplication dataset."""

    OP_IDS: Tuple[str, str] = ("add", "mul")
    SETS: Tuple[str, str, str] = ("train", "test", "valid")
    _CACHE: Dict[str, Dict[str, np.ndarray]] = {}

    def __init__(
        self,
        split: str = "train",
        n_samples: int = 100_000,
        n_digits: int = 2,
        restrict: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        if split not in self.SETS:
            raise ValueError(f"split must be one of {self.SETS}, got {split!r}")
        self.split = split
        self.n_digits = n_digits
        self.n_samples = n_samples
        self.restrict = list(restrict) if restrict is not None else None
        self.name_class = list(self.OP_IDS)
        self.num_class = len(self.OP_IDS)

        cache_key = f"{split}_{n_digits}_{n_samples}"
        if cache_key not in self._CACHE:
            rng = np.random.RandomState(0x12345678 + self.SETS.index(split))
            args = rng.randint(0, 10**n_digits, (n_samples, 2))
            ops = rng.randint(0, 2, (n_samples,))
            mod = 10**n_digits
            add_res = (args[:, 0] + args[:, 1]) % mod
            mul_res = (args[:, 0] * args[:, 1]) % mod
            res = np.where(ops == 0, add_res, mul_res)
            self._CACHE[cache_key] = {
                "args": args,
                "op": ops.astype(np.uint8),
                "res": res,
            }

        data = self._CACHE[cache_key]
        if self.restrict:
            mask = np.zeros(len(data["op"]), dtype=bool)
            for op_name in self.restrict:
                if op_name not in self.OP_IDS:
                    raise ValueError(f"Unknown op {op_name!r}; expected one of {self.OP_IDS}")
                mask |= data["op"] == self.OP_IDS.index(op_name)
            self.args = data["args"][mask]
            self.ops = data["op"][mask]
            self.res = data["res"][mask]
        else:
            self.args = data["args"]
            self.ops = data["op"]
            self.res = data["res"]

        self.in_features = n_digits * 2 * 10 + 2
        self.out_features = n_digits * 10

    def __len__(self) -> int:
        return int(self.ops.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        arg_digits = torch.from_numpy(_split_digits(self.n_digits, self.args[idx])).long()
        op_val = int(self.ops[idx])
        res_digits = torch.from_numpy(_split_digits(self.n_digits, np.asarray(self.res[idx]))).long()

        onehot_args = F.one_hot(arg_digits, num_classes=10).reshape(-1).float()
        onehot_op = F.one_hot(torch.tensor(op_val, dtype=torch.long), num_classes=2).float()
        x = torch.cat([onehot_args, onehot_op], dim=0)
        return x, res_digits


class AddMulDataProvider(DataProvider):
    """Unified PositronixLab ``DataProvider`` for Csordas et al.'s Add/Mul task."""

    def __init__(
        self,
        n_digits: int = 2,
        n_train_samples: int = 100_000,
        n_val_samples: int = 10_000,
        n_test_samples: int = 10_000,
        restrict: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.name = "addmul"
        self.n_digits = n_digits
        self.n_train_samples = n_train_samples
        self.n_val_samples = n_val_samples
        self.n_test_samples = n_test_samples
        self.restrict = list(restrict) if restrict is not None else None

        self.num_features = n_digits * 2 * 10 + 2
        self.num_classes = n_digits * 10
        self.x_shape = (self.num_features,)
        self.y_shape = (n_digits,)
        self.criterion = AddMulLoss()

        self.dataset_train = AddMulDataset(
            split="train", n_samples=n_train_samples, n_digits=n_digits, restrict=self.restrict
        )
        self.dataset_val = AddMulDataset(
            split="valid", n_samples=n_val_samples, n_digits=n_digits, restrict=self.restrict
        )
        self.dataset_test = AddMulDataset(
            split="test", n_samples=n_test_samples, n_digits=n_digits, restrict=self.restrict
        )
        self.dataset = self.dataset_train

    def with_restrict(self, restrict: Optional[List[str]]) -> "AddMulDataProvider":
        """Return a new provider restricted to a subset of operations (e.g. ``['add']``)."""
        return AddMulDataProvider(
            n_digits=self.n_digits,
            n_train_samples=self.n_train_samples,
            n_val_samples=self.n_val_samples,
            n_test_samples=self.n_test_samples,
            restrict=restrict,
            batch_size=self.batch_size,
            split=self.dataset_split,
            num_workers=self.num_workers,
            transforms=self.transforms,
            indexed_dataset=self.indexed_dataset,
        )
