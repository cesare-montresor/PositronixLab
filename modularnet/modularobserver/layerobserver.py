
from pymoo.core import callback
from torch import nn
import torch

from modularnet.modularobserver.modelmapper import ModelNode
from modularnet.spacemath.common import  is_activation, is_conv, is_linear, is_layer

from typing import TYPE_CHECKING, Callable

from modularnet.spacemath.spacemath import ActivationType, LayerType, SpaceMath
if TYPE_CHECKING:
    from modularnet.modularobserver.modularobserver import ModularObserver



LayerCallback = Callable[['LayerObserver', torch.Tensor, torch.Tensor], torch.Tensor | None]
WeightCallback = Callable[['LayerObserver', torch.nn.Parameter, torch.Tensor], torch.Tensor | None]



class LayerObserver:
    

    def __init__(self, observer: 'ModularObserver', layer_id: str, node: 'ModelNode'):
        self.observer = observer
        self.node = node
        self.layer_id = layer_id
        self.layer = node.module_ref

        # hooks
        self.forward_layer_hook = None
        self.backward_layer_hook = None

        self.forward_weights_hook = None
        self.backward_weights_hook = None


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
    

