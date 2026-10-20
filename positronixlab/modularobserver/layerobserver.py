
from pymoo.core import callback
from torch import nn
import torch

from positronixlab.modularobserver.modelmapper import ModelNode
from positronixlab.spacemath.common import  is_activation, is_conv, is_linear, is_layer

from typing import TYPE_CHECKING, Callable

from positronixlab.spacemath.spacemath import ActivationType, LayerType, SpaceMath
if TYPE_CHECKING:
    from positronixlab.modularobserver.modularobserver import ModularObserver



PreForwardCallback = Callable[[nn.Module, torch.Tensor], torch.Tensor | None]
ForwardCallback = Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor | None]
WeightCallback = Callable[[nn.Module, torch.nn.Parameter, torch.Tensor], torch.Tensor | None]



class LayerObserver:
    

    def __init__(self, observer: 'ModularObserver', layer_id: str, node: 'ModelNode'):
        self.observer = observer
        self.node = node
        self.layer_id = layer_id
        self.layer = node.module_ref

        # hooks
        self.pre_forward_layer_hook = None
        self.forward_layer_hook = None
        self.backward_layer_hook = None

        self.forward_weights_hook = None
        self.backward_weights_hook = None


    def on_grad_weights(self, callback: WeightCallback = None):
        [param.register_hook(lambda grad: callback(self.layer, param, grad)) for param in self.layer.parameters()]

    def on_pre_forward_layer(self, callback: PreForwardCallback = None):
        if callback is None:
            self.pre_forward_layer_hook.remove() if self.pre_forward_layer_hook else None
            self.pre_forward_layer_hook = None
        else:
            self.pre_forward_layer_hook = self.layer.register_forward_pre_hook(callback)

    def on_forward_layer(self, callback: ForwardCallback = None):
        if callback is None:
            self.forward_layer_hook.remove() if self.forward_layer_hook else None
            self.forward_layer_hook = None
        else:
            self.forward_layer_hook = self.layer.register_forward_hook(callback)

    def on_backward_layer(self, callback: ForwardCallback = None):
        if callback is None:
            self.backward_layer_hook.remove() if self.backward_layer_hook else None
            self.backward_layer_hook = None
        else:
            self.backward_layer_hook = self.layer.register_full_backward_hook(callback)
    

