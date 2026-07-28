
from pymoo.core import callback
from torch import nn
import torch

from modularnet.spacemath.common import  is_activation, is_conv, is_linear, is_layer

from typing import TYPE_CHECKING, Callable

from modularnet.spacemath.spacemath import ActivationType, LayerType, SpaceMath
if TYPE_CHECKING:
    from modularnet.modularobserver.modularobserver import ModularObserver



LayerCallback = Callable[['LayerObserver', torch.Tensor, torch.Tensor], torch.Tensor | None]
WeightCallback = Callable[['LayerObserver', torch.nn.Parameter, torch.Tensor], torch.Tensor | None]



class LayerObserver:
    

    def __init__(self, observer: 'ModularObserver', layer:nn.Module, layer_id:str, depth:int, num_units:int):
        self.observer = observer
        self.name = None
        self.param_ids = []
        self.layer_id = layer_id
        self.layer_type = LayerType.get_layer_type(layer)
        self.activation = None  # observed activation function, if any
        self.activation_type = None  # observed activation function, if any

        self.depth = depth
        self.num_units = num_units

        # shape information
        self.batch_size = None
        self.input_shape = None
        self.output_shape = None

        #model layers
        self.layer = layer


        # hooks
        self.forward_layer_hook = None
        self.backward_layer_hook = None

        self.forward_weights_hook = None
        self.backward_weights_hook = None

    def to_dict(self):
        return {
            "layer_id": self.layer_id,
            "layer_type": self.layer_type,
            "activation_type": self.activation_type,
            "depth": self.depth,
            "num_units": self.num_units,
            "batch_size": self.batch_size,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape
        }
    
    def set_activation(self, activation_fn: nn.Module = None):
        if activation_fn is None:
            self.activation = None
            self.activation_type = None
        else:
            self.activation = activation_fn
            self.activation_type = ActivationType.get_activation_type(activation_fn)

    def on_grad_weights(self, callback: WeightCallback = None):
        [param.register_hook(lambda grad: callback(self, param, grad)) for param in self.layer.parameters()]

    def on_forward_layer(self, callback: LayerCallback = None):
        if callback is None:
            self.forward_layer_hook.remove() if self.forward_layer_hook else None
            self.forward_layer_hook = None
        else:
            self.forward_layer_hook = self.layer.register_forward_hook(lambda module, input, output: callback(self, input, output))

    def on_backward_layer(self, callback: LayerCallback = None):
        if callback is None:
            self.backward_layer_hook.remove() if self.backward_layer_hook else None
            self.backward_layer_hook = None
        else:
            self.backward_layer_hook = self.layer.register_full_backward_hook(lambda module, input, output: callback(self, input, output))
    

