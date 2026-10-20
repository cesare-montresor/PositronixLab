
from __future__ import annotations
import copy
import yaml
import os
from positronixlab.utils.utils import signature_settings


class TemplateBase:

    VARIANTS = [ 'a', 'b', 'c', 'd' ]
    
    
    def __init__(self, name:str=None):
        self.name = name if name is not None else self.__class__.__name__.lower()
        self.variant = None
        self.params = {}

        self.add_choice('exp_metric_modularity', [True, False], 'bool')
        self.add_choice('exp_modularizer', [True, False], 'bool')
        
        self.add_fixed('nas_min_acc', 0.95, "float")
        self.add_fixed('nas_min_mod_score', 0.2, "float")
                
        self.add_fixed('task', "classification", "str")  # Default task is classification
        self.add_fixed('criterion_type', "cross_entropy", "str")  # Default task is classification
        self.add_fixed('model_type', "fc", "str")  # Default model type is fully connected (fc])
        self.add_range('train_epochs', [1,10], "int")  # Default number of training epochs
        #self.add_fixed('space_epochs', 1, "int")  # Default number of space epochs ( optimized for unchaning model weights)

        self.add_fixed('template_name', self.name, "str")  # Default input data shape for MNIST
        self.add_fixed('dataset_name', 'mnist', "str")  # Default input data shape for MNIST
        
        self.add_fixed('modular_space_name', self.name, "str")  # Default modular space name
        self.add_range('modular_space_size', [3,8], "int")  # Default modular space size
        self.add_range('modular_space_lr', [0.0001, 0.01], "float", log_scale=True) # Default modular space size
        self.add_range('modular_space_active_q', [0.6,0.95], "float") # Default modular space size
        self.add_choice('modular_space_distance', ['euclidean', 'cosine'], "str") # Default modular space size
        self.add_range('modular_modularize_decay', [0.8, 0.99], "float") # Default modular space size


        self.add_range('modular_space_loss_active', [0, 1], "float")
        self.add_range('modular_space_loss_inactive', [0, 1], "float")
        self.add_range('modular_space_loss_centers', [0, 1], "float")
        self.add_range('modular_space_loss_batch', [0, 1], "float")
        self.add_range('modular_space_loss_alignment', [0, 1], "float")


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


    def filename(self):
        if self.variant is not None:
            return f"template_{self.name}_{self.variant}.json"
        else:
            return f"template_{self.name}.json"
        
    def get_value(self, name: str, default=None):
        value = self.params.get(name, None)
        if value is None: return default
        val = value.get('value', None)
        if val is not None: return val
        return value.get('values', default)
        
    def get_params(self):
        return list(self.params.values())

    def add_range(self, name: str, bounds, parameter_type, log_scale = False):
        #param = RangeParameterConfig(name, bounds, parameter_type, step_size, scaling)
        param = {"name": name, "value_type": parameter_type, "type": "range", "bounds": bounds, "log_scale": log_scale}
        self.params[name] = param
    
    def add_choice(self, name: str, values: list[float] | list[int] | list[str] | list[bool], parameter_type, is_ordered: bool | None = None, dependent_parameters: dict = None):
        #param = ChoiceParameterConfig(name, values, parameter_type, is_ordered, dependent_parameters)
        param = {"name": name, "value_type": parameter_type, "type": "choice", "values": values, "is_ordered": is_ordered, "dependents": dependent_parameters}
        self.params[name] = param

    def add_fixed(self, name: str, value: list[float] | list[int] | list[str] | list[bool], parameter_type ):
        #param = ChoiceParameterConfig(name, values, parameter_type, is_ordered, dependent_parameters)
        param = {"name": name, "value_type": parameter_type, "type": "fixed", "value": value}
        self.params[name] = param

    def remove(self, name: str):
        if name in self.params:
            del self.params[name]

    
    def signature(self):
        sign = signature_settings(self.params)
        
        # print(f"Template {self.name} signature: {sign}\n\n---\n\n{self.params}\n\n---\n\n")
        return sign
    
    def clone(self) -> TemplateBase:
        return copy.deepcopy(self)
    
    def get_variant(self, variant: str=None, inplace: bool = False) -> TemplateBase:
        template = self if inplace else self.clone()
        
        if variant == None:
            template.add_choice('exp_metric_modularity', [True, False], 'bool') 
            template.add_choice('exp_modularizer', [False, True], 'bool')
            template.variant = None
            return template


        if variant not in TemplateBase.VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        
        template.variant = variant

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
        
        

        return template
        

    @classmethod
    def from_dict(cls, data: dict) -> TemplateBase:
        tpl = cls(name=data.get('name', 'default_template'))
        tpl.params = data.get('params', {})
        tpl.variant = data.get('variant', None)
        return tpl
    
    def to_dict(self):
        return {'name': self.name, 'params': self.params, 'variant': self.variant}

    def save(self, filepath=None) -> None:
        """
        Save the options to a YAML file.
        """
        if filepath is None:
            filepath = f"{TemplateBase.CONFIG_PATH}/template_{self.name}.yaml"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            
            yaml.dump(self.to_dict(), f)
    
    @classmethod
    def load(cls, filepath) -> TemplateBase:
        """
        Load options from a YAML file and return a MetaModelTemplate instance.
        """
        with open(filepath, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        return cls.from_dict(data)


    


class TemplateCNN(TemplateBase):

    def __init__(self, name: str=None):
        super().__init__(name)   
        
        self.add_fixed('model_type', "cnn", "str")  # Default model type is convolutional neural network (cnn)
        self.add_choice('cnn_kernel_size', [3,5,7], "int")
        self.add_range('modular_cnn_locality', [0.8, 0.99], "float") # Default modular space size


class TemplateFC(TemplateBase):

    def __init__(self, name: str=None):
        super().__init__(name)   
        
        self.add_fixed('model_type', "fc", "str")  # Default model type is fully connected (fc)
        self.add_range('input_hidden_dim', [64,512], "int")

