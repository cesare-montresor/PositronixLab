"""Synthetic & classification compositional datasets from Csordas et al. (ICLR 2021).

Reference: ``libs/are_neural_nets_modular/dataset/``
- :class:`TupleArithmeticDataProvider` (Double Addition / Multiplication — zero disk download)
- :class:`AddMulDataProvider` (Add vs. Mul — zero disk download)
- :class:`ClassRemovalDataProvider` (Leave-one-class-out wrapper for CIFAR-10 or any dataset)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from positronixlab.dataprovider.sota.base import CompositionalDataProvider, SubtaskSplit


def _digits_to_onehot(digits: np.ndarray, n_classes: int = 10) -> torch.Tensor:
    t = torch.from_numpy(digits.astype(np.int64))
    return torch.nn.functional.one_hot(t, num_classes=n_classes).float()


class TupleArithmeticDataset(Dataset):
    """Double-Addition / Double-Multiplication dataset (Csordas et al., 2021).

    Generates ``n_tuples`` independent arithmetic pairs of ``n_digits``-digit
    integers modulo ``10**n_digits``.

    When ``restrict_tuple`` is ``None`` (full task), both tuples are active.
    When ``restrict_tuple`` is ``0`` or ``1``, the other tuple's input channels
    are zeroed out and the target is that tuple's answer (matching
    ``FFTupleArithmeticDatasetInterface``).
    """

    def __init__(
        self,
        n_samples: int = 4096,
        n_tuples: int = 2,
        n_digits: int = 2,
        op: str = "add",
        restrict_tuple: Optional[int] = None,
        seed: int = 0x12345678,
    ) -> None:
        if op not in ("add", "mul"):
            raise ValueError(f"op must be 'add' or 'mul', got {op!r}")
        self.n_samples = n_samples
        self.n_tuples = n_tuples
        self.n_digits = n_digits
        self.op = op
        self.restrict_tuple = restrict_tuple

        rng = np.random.RandomState(seed)
        max_val = 10**n_digits
        a = rng.randint(0, max_val, size=(n_samples, n_tuples))
        b = rng.randint(0, max_val, size=(n_samples, n_tuples))
        res = (a + b) % max_val if op == "add" else (a * b) % max_val

        # Convert integers to digit arrays: (N, n_tuples, n_digits)
        def _to_digits(arr: np.ndarray) -> np.ndarray:
            cols = [(arr // (10**d)) % 10 for d in range(n_digits - 1, -1, -1)]
            return np.stack(cols, axis=-1)

        a_dig = _to_digits(a)
        b_dig = _to_digits(b)
        res_dig = _to_digits(res)

        # Inputs: (N, n_tuples, 2, n_digits, 10) -> flattened (N, n_tuples * 2 * n_digits * 10)
        in_oh = _digits_to_onehot(np.stack([a_dig, b_dig], axis=2))  # (N, T, 2, D, 10)
        if restrict_tuple is not None:
            mask = torch.zeros(n_tuples, dtype=torch.bool)
            mask[restrict_tuple] = True
            in_oh[:, ~mask] = 0.0
            # Classification target: integer value in [0, 10^n_digits) for restricted tuple
            self.y = torch.from_numpy(res[:, restrict_tuple].astype(np.int64))
        else:
            # Default classification target: tuple 0's result (or multi-digit via y_digits)
            self.y = torch.from_numpy(res[:, 0].astype(np.int64))

        self.x = in_oh.reshape(n_samples, -1)
        self.y_digits = torch.from_numpy(res_dig.astype(np.int64))  # (N, n_tuples, n_digits)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class TupleArithmeticDataProvider(CompositionalDataProvider):
    """Csordas Double-Addition / Multiplication compositional benchmark."""

    source = "csordas"
    task_type = "arithmetic"

    def __init__(
        self,
        n_train: int = 4096,
        n_val: int = 512,
        n_test: int = 512,
        n_tuples: int = 2,
        n_digits: int = 1,
        op: str = "add",
        batch_size: int = 64,
        seed: int = 0x12345678,
    ) -> None:
        super().__init__(name="csordas:tuple_arithmetic", batch_size=batch_size)
        self.n_tuples = n_tuples
        self.n_digits = n_digits
        self.num_features = n_tuples * 2 * n_digits * 10
        self.num_classes = 10**n_digits

        def _make_split(sub_name: str, restrict: Optional[int], offset: int) -> SubtaskSplit:
            return SubtaskSplit(
                name=sub_name,
                dataset_train=TupleArithmeticDataset(
                    n_train, n_tuples, n_digits, op, restrict, seed + offset
                ),
                dataset_val=TupleArithmeticDataset(
                    n_val, n_tuples, n_digits, op, restrict, seed + offset + 1
                ),
                dataset_test=TupleArithmeticDataset(
                    n_test, n_tuples, n_digits, op, restrict, seed + offset + 2
                ),
                batch_size=batch_size,
            )

        self.subtasks["full"] = _make_split("full", None, 0)
        for t in range(n_tuples):
            self.subtasks[f"tuple_{t}"] = _make_split(f"tuple_{t}", t, 10 * (t + 1))

        self._eval_matrix = {
            f"tuple_{t}": ["full"] + [f"tuple_{j}" for j in range(n_tuples)]
            for t in range(n_tuples)
        }


class AddMulDataset(Dataset):
    """Csordas Add/Mul synthetic dataset (``libs/are_neural_nets_modular/dataset/double_op.py``).

    Inputs: one-hot digits of ``a``, ``b`` + 2-dim one-hot operator (``0=add, 1=mul``).
    Shape: ``(N, n_digits * 2 * 10 + 2)``, target: ``(a op b) % 10^n_digits``.
    """

    def __init__(
        self,
        n_samples: int = 4096,
        n_digits: int = 1,
        restrict_op: Optional[str] = None,  # None | "add" | "mul"
        seed: int = 0x12345678,
    ) -> None:
        self.n_samples = n_samples
        self.n_digits = n_digits
        rng = np.random.RandomState(seed)
        max_val = 10**n_digits
        a = rng.randint(0, max_val, size=n_samples)
        b = rng.randint(0, max_val, size=n_samples)
        if restrict_op == "add":
            ops = np.zeros(n_samples, dtype=np.int64)
        elif restrict_op == "mul":
            ops = np.ones(n_samples, dtype=np.int64)
        elif restrict_op is None:
            ops = rng.randint(0, 2, size=n_samples)
        else:
            raise ValueError(f"restrict_op must be None, 'add', or 'mul', got {restrict_op!r}")

        res = np.where(ops == 0, (a + b) % max_val, (a * b) % max_val)

        def _to_digits(arr: np.ndarray) -> np.ndarray:
            cols = [(arr // (10**d)) % 10 for d in range(n_digits - 1, -1, -1)]
            return np.stack(cols, axis=-1)

        dig_oh = _digits_to_onehot(np.stack([_to_digits(a), _to_digits(b)], axis=1))
        op_oh = torch.nn.functional.one_hot(torch.from_numpy(ops), num_classes=2).float()
        self.x = torch.cat([dig_oh.reshape(n_samples, -1), op_oh], dim=-1)
        self.y = torch.from_numpy(res.astype(np.int64))

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class AddMulDataProvider(CompositionalDataProvider):
    """Csordas Add/Mul compositional benchmark with ``full``, ``add``, and ``mul`` subtasks."""

    source = "csordas"
    task_type = "arithmetic"

    def __init__(
        self,
        n_train: int = 4096,
        n_val: int = 512,
        n_test: int = 512,
        n_digits: int = 1,
        batch_size: int = 64,
        seed: int = 0x12345678,
    ) -> None:
        super().__init__(name="csordas:add_mul", batch_size=batch_size)
        self.n_digits = n_digits
        self.num_features = n_digits * 2 * 10 + 2
        self.num_classes = 10**n_digits

        for offset, (sub_name, op) in enumerate(
            [("full", None), ("add", "add"), ("mul", "mul")]
        ):
            self.subtasks[sub_name] = SubtaskSplit(
                name=sub_name,
                dataset_train=AddMulDataset(n_train, n_digits, op, seed + offset * 10),
                dataset_val=AddMulDataset(n_val, n_digits, op, seed + offset * 10 + 1),
                dataset_test=AddMulDataset(n_test, n_digits, op, seed + offset * 10 + 2),
                batch_size=batch_size,
            )

        self._eval_matrix = {
            "add": ["full", "add", "mul"],
            "mul": ["full", "add", "mul"],
        }


class ClassRemovalDataProvider(CompositionalDataProvider):
    """Leave-one-class-out wrapper matching ``Cifar10ClassRemovalTask`` (Csordas et al.).

    Wraps any classification train/val/test ``Dataset`` (or a synthetic fallback
    if none is passed) and creates subtasks ``"all"`` and ``"without_0" .. "without_{C-1}"``.
    """

    source = "csordas"
    task_type = "classification"

    def __init__(
        self,
        dataset_train: Dataset,
        dataset_val: Dataset,
        dataset_test: Optional[Dataset] = None,
        num_classes: int = 10,
        batch_size: int = 64,
        name: str = "csordas:class_removal",
    ) -> None:
        super().__init__(name=name, batch_size=batch_size)
        self.num_classes = num_classes

        def _extract_labels(ds: Dataset) -> List[int]:
            if hasattr(ds, "targets"):
                t = getattr(ds, "targets")
                return [int(v) for v in (t.tolist() if isinstance(t, torch.Tensor) else t)]
            if hasattr(ds, "y"):
                y = getattr(ds, "y")
                return [int(v) for v in (y.tolist() if isinstance(y, torch.Tensor) else y)]
            return [int(ds[i][1]) for i in range(len(ds))]

        train_labels = _extract_labels(dataset_train)
        val_labels = _extract_labels(dataset_val)
        test_labels = _extract_labels(dataset_test) if dataset_test is not None else None

        self.subtasks["all"] = SubtaskSplit(
            name="all",
            dataset_train=dataset_train,
            dataset_val=dataset_val,
            dataset_test=dataset_test or dataset_val,
            batch_size=batch_size,
        )

        for cls_id in range(num_classes):
            tr_idx = [i for i, y in enumerate(train_labels) if y != cls_id]
            va_idx = [i for i, y in enumerate(val_labels) if y != cls_id]
            te_ds = (
                Subset(dataset_test, [i for i, y in enumerate(test_labels) if y != cls_id])
                if (dataset_test is not None and test_labels is not None)
                else Subset(dataset_val, va_idx)
            )
            self.subtasks[f"without_{cls_id}"] = SubtaskSplit(
                name=f"without_{cls_id}",
                dataset_train=Subset(dataset_train, tr_idx),
                dataset_val=Subset(dataset_val, va_idx),
                dataset_test=te_ds,
                batch_size=batch_size,
            )

        self._eval_matrix = {
            f"without_{c}": ["all", f"without_{c}"] for c in range(num_classes)
        }
