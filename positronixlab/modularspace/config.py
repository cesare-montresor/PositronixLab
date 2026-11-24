from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from positronixlab.pipeline.config import BaseConfig


@dataclass
class LayerSpaceConfig(BaseConfig):
    """Composable hyperparameter config for `LayerSpace` / `SpaceStep` / `ModularOverSpace`."""

    space_epochs: int = 1

    # Core space geometry & optimization
    modular_space_size: int = 6
    modular_space_lr: float = 0.01
    modular_space_active_q: float = 0.80
    modular_space_distance: str = "cosine"
    modular_modularize_decay: float = 0.95
    modular_overspace_size: int = 10

    # Composable SpaceStep stage selectors
    modular_space_scoring: str = "quantile"
    modular_space_reducer: Optional[str] = None
    modular_space_apply_activation: bool = False
    modular_space_weighting: str = "raw"
    modular_space_contrast: str = "distance_log"

    # Composable loss weights (0.0 = skipped)
    modular_space_loss_global: float = 1.0
    modular_space_loss_on: float = 1.0
    modular_space_loss_off: float = 1.0
    modular_space_loss_cluster: float = 1.0
    modular_space_loss_centers: float = 0.0
    modular_space_loss_batch: float = 0.0
    modular_space_loss_sigreg: float = 1.0

    # Method-specific knobs
    modular_space_tau: float = 0.1
    modular_space_freq_alpha: float = 0.0
    modular_space_sigreg_method: str = "weak"
    modular_space_sigreg_target: str = "centers"
    modular_space_sigreg_dim: int = 64
    modular_cnn_locality: float = 0.5
