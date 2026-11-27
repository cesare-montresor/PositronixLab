"""Addition / Multiplication synthetic compositional dataset (Csordas et al., ICLR 2021).

Tests functional specialization (P_specialize): two distinct operations
(``add``, ``mul`` modulo ``10**n_digits``) sharing the exact same input and
output units, selected by a 2-way one-hot operation indicator.

Supports both:
- ``target_mode="digits"`` (original multi-digit per-digit cross-entropy,
  ``n_digits=2`` -> 42-D one-hot input, 20-D output logits, ``AddMulLoss``)
- ``target_mode="integer"`` (single-head integer classification in ``[0, 10**n_digits)``,
  ``nn.CrossEntropyLoss``, matching ``csordas:add_mul`` MLP benchmarks)
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple, Unpack

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from positronixlab.dataprovider.dataprovider import (
    CompositionalDataProvider,
    DataProviderOptions,
    SubtaskSplit,
    TransformLike,
    apply_transform,
)

TargetMode = Literal["digits", "integer"]


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
        *,
        restrict_op: Optional[str] = None,
        target_mode: TargetMode = "digits",
        seed: Optional[int] = None,
        transforms: Optional[TransformLike] = None,
    ) -> None:
        super().__init__()
        if split not in self.SETS:
            raise ValueError(f"split must be one of {self.SETS}, got {split!r}")
        if target_mode not in ("digits", "integer"):
            raise ValueError(f"target_mode must be 'digits' or 'integer', got {target_mode!r}")

        if restrict_op is not None and restrict is None:
            restrict = [restrict_op]

        self.split = split
        self.n_digits = n_digits
        self.n_samples = n_samples
        self.restrict = list(restrict) if restrict is not None else None
        self.target_mode: TargetMode = target_mode
        self.transforms = transforms
        self.name_class = list(self.OP_IDS)
        self.num_class = len(self.OP_IDS)

        eff_seed = (0x12345678 + self.SETS.index(split)) if seed is None else int(seed)
        mod = 10**n_digits

        if restrict_op is not None:
            # Direct single-op generation (guarantees full n_samples for the subtask)
            if restrict_op not in self.OP_IDS:
                raise ValueError(
                    f"restrict_op must be None, 'add', or 'mul', got {restrict_op!r}"
                )
            rng = np.random.RandomState(eff_seed)
            args = rng.randint(0, mod, (n_samples, 2))
            op_idx = self.OP_IDS.index(restrict_op)
            ops = np.full((n_samples,), op_idx, dtype=np.uint8)
            res = ((args[:, 0] + args[:, 1]) % mod) if op_idx == 0 else ((args[:, 0] * args[:, 1]) % mod)
            self.args = args
            self.ops = ops
            self.res = res
        else:
            cache_key = f"{split}_{n_digits}_{n_samples}_{eff_seed}"
            if cache_key not in self._CACHE:
                rng = np.random.RandomState(eff_seed)
                args = rng.randint(0, mod, (n_samples, 2))
                ops = rng.randint(0, 2, (n_samples,))
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
                        raise ValueError(
                            f"Unknown op {op_name!r}; expected one of {self.OP_IDS}"
                        )
                    mask |= data["op"] == self.OP_IDS.index(op_name)
                self.args = data["args"][mask]
                self.ops = data["op"][mask]
                self.res = data["res"][mask]
            else:
                self.args = data["args"]
                self.ops = data["op"]
                self.res = data["res"]

        self.in_features = n_digits * 2 * 10 + 2
        self.out_features = (n_digits * 10) if target_mode == "digits" else mod

    def __len__(self) -> int:
        return int(self.ops.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        arg_digits = torch.from_numpy(_split_digits(self.n_digits, self.args[idx])).long()
        op_val = int(self.ops[idx])

        onehot_args = F.one_hot(arg_digits, num_classes=10).reshape(-1).float()
        onehot_op = F.one_hot(torch.tensor(op_val, dtype=torch.long), num_classes=2).float()
        x = torch.cat([onehot_args, onehot_op], dim=0)
        if self.transforms is not None:
            x = apply_transform(self.transforms, x)

        if self.target_mode == "integer":
            y = torch.tensor(int(self.res[idx]), dtype=torch.long)
        else:
            y = torch.from_numpy(_split_digits(self.n_digits, np.asarray(self.res[idx]))).long()
        return x, y


class AddMulDataProvider(
    CompositionalDataProvider,
    plugin_name="addmul",
    aliases=("csordas:add_mul",),
):
    """Unified PositronixLab ``CompositionalDataProvider`` for Csordas et al.'s Add/Mul task."""

    source = "csordas"
    task_type = "arithmetic"

    def __init__(
        self,
        n_digits: int = 2,
        n_train_samples: int = 100_000,
        n_val_samples: int = 10_000,
        n_test_samples: int = 10_000,
        restrict: Optional[List[str]] = None,
        *,
        n_train: Optional[int] = None,
        n_val: Optional[int] = None,
        n_test: Optional[int] = None,
        target_mode: TargetMode = "digits",
        seed: Optional[int] = None,
        batch_size: int = 64,
        **kwargs: Unpack[DataProviderOptions],
    ) -> None:
        if n_train is not None:
            n_train_samples = n_train
        if n_val is not None:
            n_val_samples = n_val
        if n_test is not None:
            n_test_samples = n_test

        super().__init__(
            name="addmul",
            batch_size=batch_size,
            seed=0x12345678 if seed is None else seed,
            **kwargs,
        )
        self.n_digits = n_digits
        self.n_train_samples = n_train_samples
        self.n_val_samples = n_val_samples
        self.n_test_samples = n_test_samples
        self.restrict = list(restrict) if restrict is not None else None
        self.target_mode: TargetMode = target_mode

        self.num_features = n_digits * 2 * 10 + 2
        self.num_classes = (n_digits * 10) if target_mode == "digits" else (10**n_digits)
        self.x_shape = (self.num_features,)
        self.y_shape = (n_digits,) if target_mode == "digits" else ()
        self.criterion = AddMulLoss() if target_mode == "digits" else nn.CrossEntropyLoss()

        split_kw = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            indexed_dataset=self.indexed_dataset,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )

        def _build_split(sub_name: str, sub_restrict: Optional[List[str]], sub_op: Optional[str], offset: int) -> SubtaskSplit:
            base_s = None if seed is None else (seed + offset * 10)
            return SubtaskSplit(
                name=sub_name,
                dataset_train=AddMulDataset(
                    split="train",
                    n_samples=n_train_samples,
                    n_digits=n_digits,
                    restrict=sub_restrict,
                    restrict_op=sub_op,
                    target_mode=target_mode,
                    seed=base_s,
                    transforms=self.transforms,
                ),
                dataset_val=AddMulDataset(
                    split="valid",
                    n_samples=n_val_samples,
                    n_digits=n_digits,
                    restrict=sub_restrict,
                    restrict_op=sub_op,
                    target_mode=target_mode,
                    seed=None if base_s is None else base_s + 1,
                    transforms=self.transforms,
                ),
                dataset_test=AddMulDataset(
                    split="test",
                    n_samples=n_test_samples,
                    n_digits=n_digits,
                    restrict=sub_restrict,
                    restrict_op=sub_op,
                    target_mode=target_mode,
                    seed=None if base_s is None else base_s + 2,
                    transforms=self.transforms,
                ),
                **split_kw,
            )

        self.subtasks["full"] = _build_split("full", self.restrict, None, 0)
        self.subtasks["add"] = _build_split("add", ["add"], "add", 1)
        self.subtasks["mul"] = _build_split("mul", ["mul"], "mul", 2)
        self._eval_matrix = {
            "add": ["full", "add", "mul"],
            "mul": ["full", "add", "mul"],
        }
        self._sync_base_split()

    def with_restrict(self, restrict: Optional[List[str]]) -> "AddMulDataProvider":
        """Return a new provider restricted to a subset of operations (e.g. ``['add']``)."""
        return AddMulDataProvider(
            n_digits=self.n_digits,
            n_train_samples=self.n_train_samples,
            n_val_samples=self.n_val_samples,
            n_test_samples=self.n_test_samples,
            restrict=restrict,
            target_mode=self.target_mode,
            batch_size=self.batch_size,
            split=self.dataset_split,
            num_workers=self.num_workers,
            transforms=self.transforms,
            indexed_dataset=self.indexed_dataset,
            shuffle_train=self.shuffle_train,
            drop_last=self.drop_last,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            seed=self.seed,
        )
