
import re
from typing import List

import torch
from torch import nn

from modularnet.common.utils import is_activation, is_conv, is_layer
from modularnet.config.trial_config import TrialConfig
from modularnet.modularobserver.layerobserver_pubsub import LayerObserver


class ModularObserver:
    def __init__(self, config: TrialConfig, device:torch.device=None, useLayers=True, useActivations=True):
        self.config = config
        self.device = device if device else torch.device('cpu')
        self.useLayers = useLayers
        self.useActivations = useActivations
        self.observers = []

       
    def mos_id(self, mo:LayerObserver):
        for i, m in enumerate(self.observers):
            if m == mo: return str(i)
        return str(-1)

    def mos_get(self, mos_id:str):
        for i, m in enumerate(self.observers):
            if str(i) == mos_id: return m
        return None
    

        
    # DEPS: modularobserver
    def set_pruning_masks(self, masks:List[torch.Tensor]=None):
        if masks is None: masks = []
        
        for mo, mask in zip(self.mos, masks):
            if mo == self.mos[-1]: continue #print("set_pruning_masks: skip last layer"); continue
            if mask is None: print("set_pruning_masks: empty mask"); continue
            if not isinstance(mask, torch.Tensor): print("set_pruning_masks: not a tensor!"); continue
            mos_id = self.mos.index(mo)
            if len(mask) != len(self.positions[mos_id]): print("set_pruning_masks: size mismatch!"); continue
            mo.set_pruning_mask(mask)

    
    def observe_activation(self, module:nn.Module):
        if not self.useActivations: return False
        if not is_activation(module): return False
        self.observers[-1].observed_activation = module
        return True
        
    def observe_layer(self, module: nn.Module, num_units:int, depth:int): 
        if not self.useLayers: return False
        if not is_layer(module): return False
        lo = LayerObserver(self, num_units, module, depth, self.device)
        self.observers.append(lo)
        return True
    
    def num_units(self, parent_module: nn.Module, child_module: nn.Module):
        if hasattr(child_module,'out_features'): return child_module.out_features
        elif hasattr(child_module,'out_channels'): return child_module.out_channels
        elif hasattr(parent_module,'out_features'): return parent_module.out_features
        elif hasattr(parent_module,'out_channels'): return parent_module.out_channels
        
        return None

    def watch(self, parent: nn.Module):
        self.observers = []
        self._watch(parent)
        return self.observers

    def _watch(self, parent: nn.Module, num_units:int|None=None, depth=1,):
        for name, child in parent.named_children():
            # child.register_full_backward_hook(self.backward_hook)
            num_units = self.num_units(parent, child) or num_units
            
            if self.observe_activation(child): pass 
            elif self.observe_layer(child, num_units, depth):
                depth +=1
            else:
                num_units = self._watch(child, num_units, depth)
        
        return num_units

    def find_activation_layer(self, module: nn.Module):
        """ Recursively searches for an activation layer in the given module's children. """
        for _, child in module.named_children():
            if is_activation(child): return child
            result = self.find_activation_layer(child)  # Recursive search
            if result: return result
                
        return None  # No activation found
    
    def add_observer(self, observed:nn.Module, depth:int):
        num_units = observed.out_features if isinstance(observed, nn.Linear) else observed.out_channels
        observer = ModularObserver(self.space, num_units, observed, depth, self.device)
        self.observers.append(observer)
        return observer

        