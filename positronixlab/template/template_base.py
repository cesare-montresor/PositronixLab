import copy
import yaml
import os

from enum import Enum
from typing import Any, Dict, List, Optional, Union
import yaml

from ax.api.configs import (
    RangeParameterConfig,
    ChoiceParameterConfig,
)

from positronixlab.template.template_variant import TemplateVariant, TemplateVariantList


class ParameterType(str, Enum):
    INT = "int"
    FLOAT = "float"
    STRING = "str"
    BOOL = "bool"



class TemplateBase:
    """
    Defines the parameter space, fixed parameters, and variants for NAS.
    Separates tunable parameters (self.params) from fixed parameters (self.fixed)
    to keep the Ax search space clean and free of noise.
    """

    CONFIG_PATH = "configs/"

    def __init__(self, name:str=None):
        self.name = name if name is not None else self.__class__.__name__.lower()
        self.params: Dict[str, Dict[str, Any]] = {}  # Tunable search space parameters
        self.fixed: Dict[str, Any] = {}              # Fixed hyperparameters (dataset, task, etc.)
        self.variants: TemplateVariantList = TemplateVariantList()

    # -------------------------------------------------------------------------
    # Search Space Building Methods (Tunable Parameters)
    # -------------------------------------------------------------------------
    def add_range(
        self,
        name: str,
        bounds: List[Union[int, float]],
        parameter_type: Union[ParameterType, str] = ParameterType.FLOAT,
        log_scale: bool = False,
    ) -> None:
        p_type = parameter_type.value if isinstance(parameter_type, ParameterType) else str(parameter_type)
        self.params[name] = {
            "name": name,
            "type": "range",
            "bounds": bounds,
            "value_type": p_type,
            "log_scale": log_scale,
        }

    def add_choice(
        self,
        name: str,
        values: List[Any],
        parameter_type: Union[ParameterType, str] = ParameterType.STRING,
        is_ordered: Optional[bool] = None,
        dependents: Optional[Dict[str, Any]] = None,
    ) -> None:
        p_type = parameter_type.value if isinstance(parameter_type, ParameterType) else str(parameter_type)
        param: Dict[str, Any] = {
            "name": name,
            "type": "choice",
            "values": values,
            "value_type": p_type,
            "is_ordered": is_ordered,
        }
        if dependents is not None:
            param["dependents"] = dependents
        self.params[name] = param

    # -------------------------------------------------------------------------
    # Fixed Parameters (Saved in self.fixed to avoid polluting the search space)
    # -------------------------------------------------------------------------
    def add_fixed(
        self,
        name: str,
        value: Any,
        parameter_type: Optional[Union[ParameterType, str]] = None,
    ) -> None:
        """Stores fixed parameters directly in self.fixed."""
        self.fixed[name] = value

    def get_fixed(self, keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Pulls fixed parameters. If specific keys are requested (e.g. ['dataset_name', 'task']),
        returns only those. Otherwise returns all fixed parameters.
        """
        if keys is not None:
            return {k: self.fixed[k] for k in keys if k in self.fixed}
        return dict(self.fixed)

    # -------------------------------------------------------------------------
    # Variant Management
    # -------------------------------------------------------------------------
    def add_variant(
        self,
        name: str,
        fixed_features: Optional[Dict[str, Any]] = None,
        description: str = "",
    ) -> TemplateVariant:
        variant = TemplateVariant(name=name, fixed_features=fixed_features, description=description)
        self.variants.add_variant(variant)
        return variant

    def get_variant(self, name: str) -> TemplateVariant:
        return self.variants.get_variant(name)

    def list_variants(self) -> List[str]:
        return self.variants.list_variants()

    # -------------------------------------------------------------------------
    # Ax Search Space Conversion (Tunable Parameters Only)
    # -------------------------------------------------------------------------
    def to_parameter_configs(
        self,
    ) -> List[Union[RangeParameterConfig, ChoiceParameterConfig]]:
        """
        Converts tunable parameters (self.params) into modern Ax ParameterConfig objects
        (RangeParameterConfig, ChoiceParameterConfig).
        Fixed parameters are excluded to avoid search space noise.
        """
        configs = []

        for name, spec in self.params.items():
            p_type = spec.get("type")
            if p_type == "range":
                configs.append(
                    RangeParameterConfig(
                        name=name,
                        bounds=tuple(spec["bounds"]),
                        parameter_type=spec.get("value_type", "float"),
                        log_scale=spec.get("log_scale", False),
                    )
                )
            elif p_type == "choice":
                configs.append(
                    ChoiceParameterConfig(
                        name=name,
                        values=spec["values"],
                        parameter_type=spec.get("value_type", "str"),
                        is_ordered=spec.get("is_ordered"),
                    )
                )

        return configs

    # -------------------------------------------------------------------------
    # Hashing & Serialization
    # -------------------------------------------------------------------------
    

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "params": self.params,
            "fixed": self.fixed,
            "variants": self.variants.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemplateBase":
        tpl = cls(name=data.get("name", "template"))
        tpl.params = data.get("params", {})
        tpl.fixed = data.get("fixed", {})
        if "variants" in data:
            tpl.variants = TemplateVariantList.from_dict(data["variants"])
        return tpl

    def save(self, filepath: Optional[str] = None) -> str:
        if filepath is None:
            filepath = os.path.join(self.CONFIG_PATH, f"template_{self.name}.yaml")
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, sort_keys=False)
        return filepath

    @classmethod
    def load(cls, filepath: str) -> "TemplateBase":
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)






class TemplateDL(TemplateBase):
    
    
    def __init__(self, name:str=None):
        self.name = name if name is not None else self.__class__.__name__.lower()
        self.variant = None
        self.params = {}

                
        
        self.add_fixed('nas_min_acc', 0.95, "float")
                
        self.add_fixed('task', "classification", "str")  # Default task is classification
        self.add_fixed('criterion_type', "cross_entropy", "str")  # Default task is classification
        self.add_fixed('model_type', "fc", "str")  # Default model type is fully connected (fc])
        self.add_range('train_epochs', [1,10], "int")  # Default number of training epochs
        #self.add_fixed('space_epochs', 1, "int")  # Default number of space epochs ( optimized for unchaning model weights)

        self.add_fixed('template_name', self.name, "str")  # Default input data shape for MNIST
        self.add_fixed('dataset_name', 'mnist', "str")  # Default input data shape for MNIST
        


        self.add_fixed('input_data_shape', 784, "int") 
        self.add_range('input_hidden_dim', [64,512], "int") 

        self.add_range('input_num_layers', [2,10], "int")
        self.add_range('input_reduce_every', [2,8], "int")  # Default input size reduced every N layer. <1 no reduction
        self.add_choice('input_dropout_every', [-1, 0, 1], "int") # 0 first only, -1 none, N>0 every N layer])
        self.add_choice('input_batch_norm', [True,False], "bool")
        self.add_choice('input_residual', [True,False], "bool")


        self.add_fixed('output_data_shape', 10, "int") # -1: auto
        self.add_range('output_hidden_dim', [64,256], "int") # -1: auto

        self.add_range('output_num_layers', [2,4], "int") # -1: auto
        self.add_range('output_reduce_every', [0,2], "int")  # Default input size reduced every N layer. <1 no reduction
        self.add_choice('output_dropout_every', [-1,0,1], "int") # 0 first only, -1 none, N>0 every N layer])
        self.add_choice('output_batch_norm', [True,False], "bool")
        self.add_choice('output_residual', [True,False], "bool")


        self.add_choice('activation', ['relu', 'leaky_relu'], "str")
        self.add_range('dropout', [0.1,0.5], "float")
        self.add_range('learning_rate', [0.0001, 0.01], "float", log_scale=True)
        self.add_range('batch_size', [32,128], "int")

    """
        if variant == 'a':
            template.add_choice('exp_metric_modularity', [False], 'bool')
            template.add_choice('exp_modularizer', [False], 'bool')
        elif variant == 'b':
            template.add_choice('exp_metric_modularity', [True], 'bool')
            template.add_choice('exp_modularizer', [False], 'bool')
        elif variant == 'c':
            template.add_choice('exp_metric_modularity', [False], 'bool')
            template.add_choice('exp_modularizer', [True], 'bool')
        elif variant == 'd':
            template.add_choice('exp_metric_modularity', [True], 'bool')
            template.add_choice('exp_modularizer', [True], 'bool')
    """    
        
class TemplateAtlas(TemplateDL):
    def __init__(self, name: str=None):
        super().__init__(name)

        self.add_choice('exp_metric_modularity', [True, False], 'bool')
        self.add_choice('exp_modularizer', [True, False], 'bool')

        self.add_fixed('nas_min_mod_score', 0.2, "float")
        
        
        self.add_range('modular_space_size', [3,8], "int")  # Default modular space size
        self.add_range('modular_space_lr', [0.0001, 0.01], "float", log_scale=True) # Default modular space size
        self.add_range('modular_space_active_q', [0.6,0.95], "float") # Default modular space size
        self.add_choice('modular_space_distance', ['euclidean', 'cosine'], "str") # Default modular space size
        self.add_range('modular_modularize_decay', [0.8, 0.99], "float") # Default modular space size


        self.add_range('modular_space_loss_on', [0, 1], "float")
        self.add_range('modular_space_loss_off', [0, 1], "float")
        self.add_range('modular_space_loss_centers', [0, 1], "float")
        self.add_range('modular_space_loss_batch', [0, 1], "float")
        self.add_range('modular_space_loss_alignment', [0, 1], "float")


class TemplateCNN(TemplateAtlas):

    def __init__(self, name: str=None):
        super().__init__(name)   
        
        self.add_fixed('model_type', "cnn", "str")  # Default model type is convolutional neural network (cnn)
        self.add_choice('cnn_kernel_size', [3,5,7], "int")
        self.add_range('modular_cnn_locality', [0.8, 0.99], "float") # Default modular space size


class TemplateFC(TemplateAtlas):

    def __init__(self, name: str=None):
        super().__init__(name)   
        
        self.add_fixed('model_type', "fc", "str")  # Default model type is fully connected (fc)
        self.add_range('input_hidden_dim', [64,512], "int")

