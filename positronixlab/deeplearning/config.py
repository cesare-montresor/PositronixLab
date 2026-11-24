from __future__ import annotations

from dataclasses import dataclass

from positronixlab.pipeline.config import BaseConfig


@dataclass
class TrainConfig(BaseConfig):
    """Composable training hyperparameter config for `DeeplearningTrain`."""

    optimizer: str = "adamw"
    learning_rate: float = 0.001
    epochs: int = 10
    train_epochs: int = 10
