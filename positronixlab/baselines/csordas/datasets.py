"""Paper-calibrated dataset references for Csordas et al. (ICLR 2021).

Reference: ``libs/are_neural_nets_modular/``

Every dataset from the paper is exposed here as a typed builder function
pointing to the canonical ``positronixlab.dataprovider`` class with the
original paper defaults, alongside :data:`PAPER_CONFIGS`:

- :func:`addmul` — Modulo-100 Add vs. Mul (``n_digits=2``, 100k/10k/10k, per-digit loss)
- :func:`tuple_arithmetic` — Double-Addition / Multiplication (``n_digits=2``, ``n_tuples=2``, 100k/10k/10k)
- :func:`cifar10_class_removal` — CIFAR-10 leave-one-class-out (10 subtasks ``without_0..9``)
- :func:`permuted_mnist` — Permuted MNIST (deterministic pixel permutations ``0..10``)
- :func:`scan` — SCAN sequence-to-sequence benchmark
- :func:`deepmind_math` — DeepMind Mathematics dataset
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Unpack

from positronixlab.dataprovider.addmul_dataprovider import (
    AddMulDataProvider as _BaseAddMulDataProvider,
    AddMulDataset as _BaseAddMulDataset,
    AddMulLoss,
    TargetMode,
    addmul_exact_accuracy,
)
from positronixlab.dataprovider.cifar10_dataprovider import CIFAR10DataProvider
from positronixlab.dataprovider.class_removal_dataprovider import (
    ClassRemovalDataProvider,
)
from positronixlab.dataprovider.config import DataConfig
from positronixlab.dataprovider.dataprovider import DataProviderOptions
from positronixlab.dataprovider.math_dataprovider import MathDataProvider
from positronixlab.dataprovider.permuted_mnist_dataprovider import (
    PermutedMnistDataProvider,
)
from positronixlab.dataprovider.scan_dataprovider import SCANDataProvider
from positronixlab.dataprovider.tuple_arithmetic_dataprovider import (
    TupleArithmeticDataProvider as _BaseTupleArithmeticDataProvider,
    TupleArithmeticDataset as _BaseTupleArithmeticDataset,
    TupleArithmeticLoss,
    tuple_arithmetic_accuracy,
)


# ---------------------------------------------------------------------------
# Paper-calibrated builders (original Csordas et al. ICLR 2021 settings)
# ---------------------------------------------------------------------------


def addmul(
    *,
    n_digits: int = 2,
    n_train: int = 100_000,
    n_val: int = 10_000,
    n_test: int = 10_000,
    target_mode: TargetMode = "digits",
    batch_size: int = 128,
    **kwargs: Unpack[DataProviderOptions],
) -> _BaseAddMulDataProvider:
    """Csordas et al. Add/Mul task (``n_digits=2``, 100k/10k/10k, per-digit CE)."""
    return _BaseAddMulDataProvider(
        n_digits=n_digits,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        target_mode=target_mode,
        batch_size=batch_size,
        **kwargs,
    )


def tuple_arithmetic(
    *,
    n_digits: int = 2,
    n_tuples: int = 2,
    n_train: int = 100_000,
    n_val: int = 10_000,
    n_test: int = 10_000,
    op: Literal["add", "mul"] = "add",
    target_mode: TargetMode = "digits",
    batch_size: int = 128,
    **kwargs: Unpack[DataProviderOptions],
) -> _BaseTupleArithmeticDataProvider:
    """Csordas et al. Double-Addition / TupleArithmetic task (``n_digits=2``, ``n_tuples=2``)."""
    return _BaseTupleArithmeticDataProvider(
        n_digits=n_digits,
        n_tuples=n_tuples,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        op=op,
        target_mode=target_mode,
        batch_size=batch_size,
        **kwargs,
    )


def cifar10_class_removal(
    *,
    batch_size: int = 128,
    **kwargs: Unpack[DataProviderOptions],
) -> ClassRemovalDataProvider:
    """Csordas et al. CIFAR-10 leave-one-class-out benchmark (``all`` + ``without_0..9``)."""
    cifar = CIFAR10DataProvider(batch_size=batch_size, **kwargs)
    return ClassRemovalDataProvider.from_provider(
        cifar,
        name="csordas:cifar10_class_removal",
        batch_size=batch_size,
    )


def permuted_mnist(
    *,
    permutation: int = 0,
    flatten: bool = True,
    normalize: bool = True,
    batch_size: int = 128,
    **kwargs: Unpack[DataProviderOptions],
) -> PermutedMnistDataProvider:
    """Csordas et al. Permuted MNIST transfer/reuse benchmark (permutations ``0..10``)."""
    return PermutedMnistDataProvider(
        permutation=permutation,
        flatten=flatten,
        normalize=normalize,
        batch_size=batch_size,
        **kwargs,
    )


def scan(
    *,
    batch_size: int = 128,
    **kwargs: Unpack[DataProviderOptions],
) -> SCANDataProvider:
    """Csordas et al. SCAN compositional generalization dataset."""
    return SCANDataProvider(batch_size=batch_size, **kwargs)


def deepmind_math(
    *,
    batch_size: int = 128,
    **kwargs: Unpack[DataProviderOptions],
) -> MathDataProvider:
    """Csordas et al. DeepMind Mathematics dataset."""
    return MathDataProvider(batch_size=batch_size, **kwargs)


PAPER_CONFIGS: Dict[str, DataConfig] = {
    "addmul": DataConfig(
        name="csordas_addmul",
        dataset_name="addmul",
        batch_size=128,
        dataset_kwargs={
            "n_digits": 2,
            "n_train": 100_000,
            "n_val": 10_000,
            "n_test": 10_000,
            "target_mode": "digits",
        },
    ),
    "tuple_arithmetic": DataConfig(
        name="csordas_tuple_arithmetic",
        dataset_name="tuple_arithmetic",
        batch_size=128,
        dataset_kwargs={
            "n_digits": 2,
            "n_tuples": 2,
            "n_train": 100_000,
            "n_val": 10_000,
            "n_test": 10_000,
            "op": "add",
            "target_mode": "digits",
        },
    ),
    "permuted_mnist": DataConfig(
        name="csordas_permuted_mnist",
        dataset_name="permuted_mnist",
        batch_size=128,
        flatten=True,
        dataset_kwargs={"permutation": 0, "normalize": True},
    ),
    "scan": DataConfig(
        name="csordas_scan",
        dataset_name="scan",
        batch_size=128,
    ),
    "deepmind_math": DataConfig(
        name="csordas_deepmind_math",
        dataset_name="deepmind_math",
        batch_size=128,
    ),
}


# ---------------------------------------------------------------------------
# DataProviderLoader compatibility wrappers (single-head integer mode)
# ---------------------------------------------------------------------------


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
    "AddMulLoss",
    "ClassRemovalDataProvider",
    "PAPER_CONFIGS",
    "TupleArithmeticDataProvider",
    "TupleArithmeticDataset",
    "TupleArithmeticLoss",
    "addmul",
    "addmul_exact_accuracy",
    "cifar10_class_removal",
    "deepmind_math",
    "permuted_mnist",
    "scan",
    "tuple_arithmetic",
    "tuple_arithmetic_accuracy",
]
