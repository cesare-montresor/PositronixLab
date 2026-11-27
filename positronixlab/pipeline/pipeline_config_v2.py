from dataclasses import dataclass

from positronixlab.pipeline.pipeline_config import PipelineConfig


@dataclass
class PipelineConfigV2(PipelineConfig):
    """
    Configuration parameters for a single trial using the modular `LayerSpace`
    and composable `SpaceStep` pipeline (flat fields for Ax sweeps & ablations).
    """

    # Modularity flags
    exp_metric_modularity: bool = True
    exp_modularizer: bool = True

    # General task & model architecture
    task: str = "classification"
    criterion_type: str = "cross_entropy"
    model_type: str = "fc"
    train_epochs: int = 10
    space_epochs: int = 1

    # Dataset
    dataset_name: str = "fashion_mnist"

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

    # Optimization & training hyperparameters
    optimizer: str = "adamw"
    activation: str = "relu"
    dropout: float = 0.5
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 10

    # Modular space core parameters
    modular_space_size: int = 6
    modular_space_lr: float = 0.01
    modular_space_active_q: float = 0.80
    modular_space_distance: str = "cosine"
    modular_modularize_decay: float = 0.95
    modular_overspace_size: int = 10

    # Composable space-step stage selectors (Ax choices)
    modular_space_scoring: str = "quantile"
    modular_space_reducer: str | None = None
    modular_space_apply_activation: bool = False
    modular_space_weighting: str = "raw"
    modular_space_contrast: str = "distance_log"

    # Composable loss weights (0.0 = skipped for ablations)
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

    # Multi-objective thresholds & CNN features
    nas_min_acc: float = 0.8
    nas_min_mod_score: float = 0.2
    cnn_kernel_size: int = 3
    modular_cnn_locality: float = 0.5
