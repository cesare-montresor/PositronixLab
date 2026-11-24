from __future__ import annotations

from dataclasses import dataclass

from positronixlab.pipeline.config import BaseConfig


@dataclass
class MetaModelConfig(BaseConfig):
    """Composable hyperparameter config for `MetaModel` and `MetaCriterion`."""

    task: str = "classification"
    criterion_type: str = "cross_entropy"
    model_type: str = "fc"

    # Input layer hierarchy
    input_data_shape: int = 784
    input_hidden_dim: int = 32
    input_num_layers: int = 2
    input_reduce_every: int = 0
    input_dropout_every: int = 0
    input_batch_norm: bool = True
    input_residual: bool = True

    # Output layer hierarchy
    output_data_shape: int = 10
    output_hidden_dim: int = 32
    output_num_layers: int = 2
    output_reduce_every: int = -1
    output_dropout_every: int = 0
    output_batch_norm: bool = True
    output_residual: bool = True

    # Block & CNN parameters
    activation: str = "relu"
    dropout: float = 0.5
    cnn_kernel_size: int = 3
