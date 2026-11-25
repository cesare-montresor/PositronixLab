"""Synthetic & classification compositional datasets from Csordas et al. (ICLR 2021).

Delegates to the canonical implementations in ``positronixlab.dataprovider``
with Csordas single-head classification benchmark defaults (``n_digits=1``,
``target_mode="integer"``) for ``DataProviderLoader``.
"""

from __future__ import annotations

from typing import Literal, Optional, Unpack

from positronixlab.dataprovider.addmul_dataprovider import (
    AddMulDataProvider as _BaseAddMulDataProvider,
    AddMulDataset as _BaseAddMulDataset,
)
from positronixlab.dataprovider.class_removal_dataprovider import (
    ClassRemovalDataProvider,
)
from positronixlab.dataprovider.dataprovider import DataProviderOptions
from positronixlab.dataprovider.tuple_arithmetic_dataprovider import (
    TupleArithmeticDataProvider as _BaseTupleArithmeticDataProvider,
    TupleArithmeticDataset as _BaseTupleArithmeticDataset,
)


class TupleArithmeticDataset(_BaseTupleArithmeticDataset):
    """Csordas single-head Double-Addition / Multiplication dataset."""

    def __init__(
        self,
        n_samples: int = 4096,
        n_tuples: int = 2,
        n_digits: int = 1,
        op: Literal["add", "mul"] = "add",
        restrict_tuple: Optional[int] = None,
        seed: int = 0x12345678,
    ) -> None:
        super().__init__(
            split="train",
            n_digits=n_digits,
            n_tuples=n_tuples,
            n_samples=n_samples,
            op=op,
            restrict_tuple=restrict_tuple,
            target_mode="integer",
            seed=seed,
        )


class TupleArithmeticDataProvider(_BaseTupleArithmeticDataProvider, register=False):
    """Csordas Double-Addition / Multiplication compositional benchmark."""

    def __init__(
        self,
        n_train: int = 4096,
        n_val: int = 512,
        n_test: int = 512,
        n_tuples: int = 2,
        n_digits: int = 1,
        op: Literal["add", "mul"] = "add",
        batch_size: int = 64,
        seed: int = 0x12345678,
        **kwargs: Unpack[DataProviderOptions],
    ) -> None:
        super().__init__(
            n_digits=n_digits,
            n_tuples=n_tuples,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            op=op,
            target_mode="integer",
            seed=seed,
            batch_size=batch_size,
            **kwargs,
        )
        self.name = "csordas:tuple_arithmetic"


class AddMulDataset(_BaseAddMulDataset):
    """Csordas single-head Add/Mul synthetic dataset."""

    def __init__(
        self,
        n_samples: int = 4096,
        n_digits: int = 1,
        restrict_op: Optional[str] = None,
        seed: int = 0x12345678,
    ) -> None:
        super().__init__(
            split="train",
            n_samples=n_samples,
            n_digits=n_digits,
            restrict_op=restrict_op,
            target_mode="integer",
            seed=seed,
        )


class AddMulDataProvider(_BaseAddMulDataProvider, register=False):
    """Csordas Add/Mul compositional benchmark with ``full``, ``add``, and ``mul`` subtasks."""

    def __init__(
        self,
        n_train: int = 4096,
        n_val: int = 512,
        n_test: int = 512,
        n_digits: int = 1,
        batch_size: int = 64,
        seed: int = 0x12345678,
        **kwargs: Unpack[DataProviderOptions],
    ) -> None:
        super().__init__(
            n_digits=n_digits,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            target_mode="integer",
            seed=seed,
            batch_size=batch_size,
            **kwargs,
        )
        self.name = "csordas:add_mul"


__all__ = [
    "AddMulDataProvider",
    "AddMulDataset",
    "ClassRemovalDataProvider",
    "TupleArithmeticDataProvider",
    "TupleArithmeticDataset",
]
