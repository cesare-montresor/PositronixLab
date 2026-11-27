"""Double-Addition (TupleArithmetic) synthetic compositional dataset (Csordas et al., ICLR 2021).

Tests functional reuse (P_reuse): performing the same operation (modulo-``10**n_digits``
addition/multiplication) across ``n_tuples`` independent input/output slots.

Supports both:
- ``target_mode="digits"`` (original per-digit targets ``(n_tuples, n_digits)``,
  ``TupleArithmeticLoss``, ``tuple_arithmetic_accuracy``)
- ``target_mode="integer"`` (single-head integer target in ``[0, 10**n_digits)`` for
  restricted tuple or tuple 0, ``nn.CrossEntropyLoss``, matching ``csordas:tuple_arithmetic``)
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from positronixlab.dataprovider.addmul_dataprovider import TargetMode, _split_digits
from positronixlab.dataprovider.dataprovider import (
    CompositionalDataProvider,
    SubtaskSplit,
    TransformLike,
    apply_transform,
)


class TupleArithmeticLoss(nn.Module):
    """Per-digit cross-entropy for ``TupleArithmetic`` with optional single-tuple restriction."""

    def __init__(
        self,
        n_tuples: int = 2,
        n_digits: int = 2,
        restrict_tuple: int | None = None,
    ) -> None:
        super().__init__()
        self.n_tuples = n_tuples
        self.n_digits = n_digits
        self.restrict_tuple = restrict_tuple

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
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
) -> dict[str, float]:
    """Return total and per-tuple exact-match accuracy."""
    preds = logits.reshape(logits.shape[0], n_tuples, n_digits, 10).argmax(dim=-1)
    tuple_ok = (preds == targets.long()).all(dim=-1)
    res: dict[str, float] = {"total": float(tuple_ok.all(dim=-1).float().mean().item())}
    for t in range(n_tuples):
        res[f"tuple_{t}"] = float(tuple_ok[:, t].float().mean().item())
    return res


class TupleArithmeticDataset(Dataset):
    """Synthetic double-addition / multi-tuple arithmetic dataset."""

    SETS: tuple[str, str, str] = ("train", "test", "valid")
    _CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def __init__(
        self,
        split: str = "train",
        n_digits: int = 2,
        n_tuples: int = 2,
        n_samples: int = 100_000,
        op: Literal["add", "mul"] = "add",
        restrict_tuple: int | None = None,
        *,
        target_mode: TargetMode = "digits",
        seed: int | None = None,
        transforms: TransformLike | None = None,
    ) -> None:
        super().__init__()
        if split not in self.SETS:
            raise ValueError(f"split must be one of {self.SETS}, got {split!r}")
        if op not in ("add", "mul"):
            raise ValueError(f"op must be 'add' or 'mul', got {op!r}")
        if restrict_tuple is not None and not (0 <= restrict_tuple < n_tuples):
            raise ValueError(
                f"restrict_tuple must be in [0, {n_tuples}), got {restrict_tuple}"
            )
        if target_mode not in ("digits", "integer"):
            raise ValueError(f"target_mode must be 'digits' or 'integer', got {target_mode!r}")

        self.split = split
        self.n_digits = n_digits
        self.n_tuples = n_tuples
        self.n_samples = n_samples
        self.op = op
        self.restrict_tuple = restrict_tuple
        self.target_mode: TargetMode = target_mode
        self.transforms = transforms
        self.maxnum = 10**n_digits

        eff_seed = (0x12345678 + self.SETS.index(split)) if seed is None else int(seed)
        cache_key = f"{split}_{n_digits}_{n_tuples}_{n_samples}_{op}_{eff_seed}"
        if cache_key not in self._CACHE:
            rng = np.random.RandomState(eff_seed)
            i_list = [rng.randint(0, self.maxnum, (n_samples, 1, 2)) for _ in range(n_tuples)]
            inputs = np.concatenate(i_list, axis=1)  # (N, n_tuples, 2)
            if op == "add":
                outputs = (inputs[:, :, 0] + inputs[:, :, 1]) % self.maxnum
            else:
                outputs = (inputs[:, :, 0] * inputs[:, :, 1]) % self.maxnum
            self._CACHE[cache_key] = (inputs, outputs)

        self.inputs, self.outputs = self._CACHE[cache_key]
        self.in_features = n_tuples * 2 * n_digits * 10
        self.out_features = (
            n_tuples * n_digits * 10 if target_mode == "digits" else self.maxnum
        )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        in_digits = torch.from_numpy(_split_digits(self.n_digits, self.inputs[idx])).long()
        onehot_in = F.one_hot(in_digits, num_classes=10).float()
        if self.restrict_tuple is not None:
            mask = torch.zeros(self.n_tuples, 1, 1, 1, dtype=onehot_in.dtype)
            mask[self.restrict_tuple] = 1.0
            onehot_in = onehot_in * mask
        x = onehot_in.reshape(-1)
        if self.transforms is not None:
            x = apply_transform(self.transforms, x)

        if self.target_mode == "integer":
            t_idx = self.restrict_tuple if self.restrict_tuple is not None else 0
            y = torch.tensor(int(self.outputs[idx, t_idx]), dtype=torch.long)
        else:
            y = torch.from_numpy(_split_digits(self.n_digits, self.outputs[idx])).long()
        return x, y


class TupleArithmeticDataProvider(
    CompositionalDataProvider,
    plugin_name="tuple_arithmetic",
    aliases=("csordas:tuple_arithmetic",),
):
    """Unified PositronixLab ``CompositionalDataProvider`` for Csordas et al.'s Double-Addition task."""

    source = "csordas"
    task_type = "arithmetic"

    def __init__(
        self,
        n_digits: int = 2,
        n_tuples: int = 2,
        n_train_samples: int = 100_000,
        n_val_samples: int = 10_000,
        n_test_samples: int = 10_000,
        op: Literal["add", "mul"] = "add",
        restrict_tuple: int | None = None,
        *,
        n_train: int | None = None,
        n_val: int | None = None,
        n_test: int | None = None,
        target_mode: TargetMode = "digits",
        seed: int | None = None,
        batch_size: int = 64,
        **kwargs: Any,
    ) -> None:
        if n_train is not None:
            n_train_samples = n_train
        if n_val is not None:
            n_val_samples = n_val
        if n_test is not None:
            n_test_samples = n_test

        super().__init__(
            name="tuple_arithmetic",
            batch_size=batch_size,
            seed=0x12345678 if seed is None else seed,
            **kwargs,
        )
        self.n_digits = n_digits
        self.n_tuples = n_tuples
        self.n_train_samples = n_train_samples
        self.n_val_samples = n_val_samples
        self.n_test_samples = n_test_samples
        self.op = op
        self.restrict_tuple = restrict_tuple
        self.target_mode: TargetMode = target_mode

        self.num_features = n_tuples * 2 * n_digits * 10
        self.num_classes = (
            n_tuples * n_digits * 10 if target_mode == "digits" else (10**n_digits)
        )
        self.x_shape = (self.num_features,)
        self.y_shape = (n_tuples, n_digits) if target_mode == "digits" else ()
        self.criterion = (
            TupleArithmeticLoss(
                n_tuples=n_tuples, n_digits=n_digits, restrict_tuple=restrict_tuple
            )
            if target_mode == "digits"
            else nn.CrossEntropyLoss()
        )

        split_kw: dict[str, Any] = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            indexed_dataset=self.indexed_dataset,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )

        def _make_split(sub_name: str, sub_restrict: int | None, offset: int) -> SubtaskSplit:
            base_s = None if seed is None else (seed + offset)
            return SubtaskSplit(
                name=sub_name,
                dataset_train=TupleArithmeticDataset(
                    split="train",
                    n_digits=n_digits,
                    n_tuples=n_tuples,
                    n_samples=n_train_samples,
                    op=op,
                    restrict_tuple=sub_restrict,
                    target_mode=target_mode,
                    seed=base_s,
                    transforms=self.transforms,
                ),
                dataset_val=TupleArithmeticDataset(
                    split="valid",
                    n_digits=n_digits,
                    n_tuples=n_tuples,
                    n_samples=n_val_samples,
                    op=op,
                    restrict_tuple=sub_restrict,
                    target_mode=target_mode,
                    seed=None if base_s is None else base_s + 1,
                    transforms=self.transforms,
                ),
                dataset_test=TupleArithmeticDataset(
                    split="test",
                    n_digits=n_digits,
                    n_tuples=n_tuples,
                    n_samples=n_test_samples,
                    op=op,
                    restrict_tuple=sub_restrict,
                    target_mode=target_mode,
                    seed=None if base_s is None else base_s + 2,
                    transforms=self.transforms,
                ),
                **split_kw,
            )

        self.subtasks["full"] = _make_split("full", restrict_tuple, 0)
        for t in range(n_tuples):
            self.subtasks[f"tuple_{t}"] = _make_split(f"tuple_{t}", t, 10 * (t + 1))

        self._eval_matrix = {
            f"tuple_{t}": ["full"] + [f"tuple_{j}" for j in range(n_tuples)]
            for t in range(n_tuples)
        }
        self._sync_base_split()

    def with_restrict_tuple(
        self, restrict_tuple: int | None
    ) -> TupleArithmeticDataProvider:
        """Return a new provider restricted to a single tuple index (e.g. ``0`` or ``1``)."""
        return TupleArithmeticDataProvider(
            n_digits=self.n_digits,
            n_tuples=self.n_tuples,
            n_train_samples=self.n_train_samples,
            n_val_samples=self.n_val_samples,
            n_test_samples=self.n_test_samples,
            op=self.op,  # type: ignore[arg-type]
            restrict_tuple=restrict_tuple,
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
