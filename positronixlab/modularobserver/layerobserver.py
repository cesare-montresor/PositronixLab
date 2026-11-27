
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from positronixlab.modularobserver.modelmapper import ModelNode

if TYPE_CHECKING:
    from positronixlab.modularobserver.modularobserver import ModularObserver



PreForwardCallback = Callable[..., Any]
ForwardCallback = Callable[..., Any]
WeightCallback = Callable[[nn.Module, torch.nn.Parameter, torch.Tensor], torch.Tensor | None]



class LayerObserver:
    

    def __init__(self, observer: 'ModularObserver | None', layer_id: str, node: 'ModelNode'):
        self.observer = observer
        self.node = node
        self.layer_id = layer_id
        self.layer = node.module_ref

        # hooks
        self.pre_forward_layer_hook: RemovableHandle | None = None
        self.forward_layer_hook: RemovableHandle | None = None
        self.backward_layer_hook: RemovableHandle | None = None

        self.forward_weights_hook: RemovableHandle | None = None
        self.backward_weights_hook: RemovableHandle | None = None
        self._weight_grad_handles: list[RemovableHandle] = []


    def on_grad_weights(self, callback: WeightCallback | None = None):
        for handle in self._weight_grad_handles:
            handle.remove()
        self._weight_grad_handles = []
        if callback is None:
            return
        for param in self.layer.parameters():
            self._weight_grad_handles.append(
                param.register_hook(lambda grad, p=param: callback(self.layer, p, grad))
            )

    def on_pre_forward_layer(self, callback: PreForwardCallback | None = None):
        if callback is None:
            self.pre_forward_layer_hook.remove() if self.pre_forward_layer_hook else None
            self.pre_forward_layer_hook = None
        else:
            self.pre_forward_layer_hook = self.layer.register_forward_pre_hook(callback)

    def on_forward_layer(self, callback: ForwardCallback | None = None):
        if callback is None:
            self.forward_layer_hook.remove() if self.forward_layer_hook else None
            self.forward_layer_hook = None
        else:
            self.forward_layer_hook = self.layer.register_forward_hook(callback)

    def on_backward_layer(self, callback: ForwardCallback | None = None):
        if callback is None:
            self.backward_layer_hook.remove() if self.backward_layer_hook else None
            self.backward_layer_hook = None
        else:
            self.backward_layer_hook = self.layer.register_full_backward_hook(callback)
    

