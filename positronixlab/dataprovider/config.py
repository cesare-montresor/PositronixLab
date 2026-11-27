from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from positronixlab.pipeline.config import BaseConfig

DatasetName = Literal[
    "iris",
    "mnist",
    "fashion_mnist",
    "fmnist",
    "cifar10",
    "cifar100",
    "covertype",
    "higgs",
    "scan",
    "deepmind_math",
    "cvr_odd_one_out",
    "cvr_simclr",
    "syneval_odd_one_out",
    "addmul",
    "tuple_arithmetic",
    "permuted_mnist",
    "class_removal",
    "csordas:add_mul",
    "csordas:tuple_arithmetic",
    "csordas:class_removal",
    "csordas:permuted_mnist",
    "csordas:deepmind_math",
    "csordas:scan",
    "lepori:cvr_inside_contact",
    "lepori:cvr_inside_count",
    "lepori:cvr_contact_count",
    "lepori:cvr_simclr",
    "lepori:syneval_sv_singular",
    "lepori:syneval_sv_plural",
    "lepori:syneval_pro_ant_singular",
    "lepori:syneval_pro_ant_plural",
]


@dataclass
class DataConfig(BaseConfig):
    """Composable dataset config for ``DataProviderFactory``."""

    dataset_name: DatasetName | str = "fashion_mnist"
    batch_size: int = 32
    split: list[float] | None = None
    num_workers: int | None = None
    root: str | None = None
    flatten: bool = False
    indexed_dataset: bool = False
    shuffle_train: bool = True
    drop_last: bool = True
    persistent_workers: bool = True
    pin_memory: bool = False
    seed: int = 42
    dataset_kwargs: dict[str, Any] = field(default_factory=dict)
