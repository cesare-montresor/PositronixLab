from __future__ import annotations

from dataclasses import dataclass

from positronixlab.pipeline.config import BaseConfig


@dataclass
class DataConfig(BaseConfig):
    """Composable dataset config for `DataProviderFactory`."""

    dataset_name: str = "fashion_mnist"
    batch_size: int = 32
