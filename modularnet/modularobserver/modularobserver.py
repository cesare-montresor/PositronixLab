
import re
from typing import Dict, List, TYPE_CHECKING

import torch
from torch import nn

from modularnet.spacemath.spacemath import SpaceMath
from modularnet.spacemath.common import is_activation, is_conv, is_linear, is_layer
from modularnet.modularobserver.layerobserver import LayerCallback, WeightCallback
from modularnet.trial.trial_config import TrialConfig

if TYPE_CHECKING:
    from modularnet.modularobserver.layerobserver import LayerObserver


class ModularObserver:
    def __init__(self, config: TrialConfig, model: nn.Module=None, device:torch.device=None, useLayers=True, useActivations=True):
        self.config = config
        self.device = device if device else model.device if model is not None else torch.device("cpu")
        self.useLayers = useLayers
        self.useActivations = useActivations
        self.observers: List['LayerObserver'] = []
        self.by_layer: Dict[nn.Module, 'LayerObserver'] = {}
        self.model: nn.Module = None

        self.forward_hook = None
        self.backward_hook = None

        if model is not None:
            self.observe(model)


    def on_forward(self, callback=None):
        if self.model is None:
            raise ValueError("Model not set. Call observe(model) first.")
        if callback is None and self.forward_hook is not None:
            self.forward_hook.remove()
            self.forward_hook = None
        elif callback is not None:
            self.forward_hook = self.model.register_forward_hook( lambda module, input, output: callback(self, input, output) )

    def on_backward(self, callback=None):
        if self.model is None:
            raise ValueError("Model not set. Call observe(model) first.")
        if callback is None and self.backward_hook is not None:
            self.backward_hook.remove()
            self.backward_hook = None
        elif callback is not None:
            self.backward_hook = self.model.register_full_backward_hook(lambda module, input, output: callback(self, input, output))

        
    def on_forward_layer(self, callback: LayerCallback=None):
        [observer.on_forward_layer(callback) for observer in self.observers]
        
    def on_backward_layer(self, callback: LayerCallback=None):
        [observer.on_backward_layer(callback) for observer in self.observers]

    def on_grad_weights(self, callback: WeightCallback=None):
        [observer.on_grad_weights(callback) for observer in self.observers] 

    def layer_id(self, lo: 'LayerObserver'):
        for i, m in enumerate(self.observers):
            if m == lo: return str(i)
        return str(-1)

    def get_layer(self, lo_id:str):
        for i, m in enumerate(self.observers):
            if str(i) == str(lo_id): return m
        return None
    
    
    def observe_activation(self, activation_fn:nn.Module):
        if not self.useActivations: return False
        if not SpaceMath.is_activation(activation_fn): return False
        self.observers[-1].set_activation(activation_fn)
        return True
        
    def observe_layer(self, module: nn.Module, num_units:int, depth:int): 
        if not self.useLayers: return False
        if not is_layer(module): return False
        lo = LayerObserver(self, module, str(len(self.observers)), depth, num_units)
        self.observers.append(lo)
        self.by_layer[module] = lo
        return True
    
    def num_units(self, parent_module: nn.Module, child_module: nn.Module):
        if hasattr(child_module,'out_features'): return child_module.out_features
        elif hasattr(child_module,'out_channels'): return child_module.out_channels
        elif hasattr(parent_module,'out_features'): return parent_module.out_features
        elif hasattr(parent_module,'out_channels'): return parent_module.out_channels
        
        return None

    def observe(self, model: nn.Module):
        self.model = model
        self.observers = []
        self._observe(model)
        return self.observers

    def _observe(self, parent: nn.Module, num_units:int|None=None, depth=1,):
        for _, child in parent.named_children():
            # child.register_full_backward_hook(self.backward_hook)
            num_units = self.num_units(parent, child) or num_units
            
            if self.observe_activation(child): pass 
            elif self.observe_layer(child, num_units, depth):
                depth +=1
            else:
                num_units = self._observe(child, num_units, depth)
        
        return num_units

    def clear(self):
        self.on_forward_layer(None)
        self.on_backward_layer(None)
        
        self.on_grad_weights(None)
        
        self.on_forward(None)
        self.on_backward(None)
        
        self.observers = []
        self.model = None
        self.forward_hook = None
        self.backward_hook = None