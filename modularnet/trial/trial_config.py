from dataclasses import dataclass, asdict
from modularnet.utils.utils import load_data, save_data, signature_settings

@dataclass
class TrialConfig:
    """
    A class to hold options for the MetaModel.
    Using dataclass removes __init__ boilerplate and provides type safety.
    """
    name: str = "default_config"

    exp_metric_modularity: bool = True
    exp_modularizer: bool = True

    task: str = "classification"
    criterion_type: str = "cross_entropy"
    model_type: str = "fc"
    train_epochs: int = 10
    space_epochs: int = 1

    dataset_name: str = "fashion_mnist"
    modular_space_name: str = "fashion_mnist_cnn"
    modular_space_size: int = 6
    modular_space_lr: float = 0.01
    modular_space_active_q: float = 0.80
    modular_space_distance: str = "cosine"
    modular_modularize_decay: float = 0.95
    
    modular_overspace_size: int = 10

    modular_space_loss_on: float = 1
    modular_space_loss_off: float = 1
    modular_space_loss_centers: float = 1
    modular_space_loss_batch: float = 1
    modular_space_loss_alignment: float = 1
    
    input_data_shape: int = 784
    input_hidden_dim: int = 32

    input_num_layers: int = 2
    input_reduce_every: int = 0
    input_dropout_every: int = 0
    input_batch_norm: bool = True
    input_residual: bool = True

    output_data_shape: int = 10
    output_hidden_dim: int = 32
    output_num_layers: int = 2
    output_reduce_every: int = -1
    output_dropout_every: int = 0
    output_batch_norm: bool = True
    output_residual: bool = True

    optimizer: str = "adamw"
    activation: str = "relu"
    dropout: float = 0.5
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 10

    nas_min_acc: float = 0.8
    nas_min_mod_score: float = 0.2
    cnn_kernel_size: int = 3

    def signature(self):
        return signature_settings(self)

    def to_dict(self):
        """
        Convert the dataclass to a dictionary.
        """
        return asdict(self)

    def save(self, filepath):
        """
        Save the options to a YAML file.
        asdict(self) converts the dataclass to a dictionary automatically.
        """
        save_data(filepath, self.to_dict() )

    @classmethod
    def load(cls, filepath):
        """
        Load options from a YAML file and return a TrialConfig instance.
        """
        options = load_data(filepath)
        # We use **options to unpack the dictionary into the dataclass kwargs
        return cls(**options)