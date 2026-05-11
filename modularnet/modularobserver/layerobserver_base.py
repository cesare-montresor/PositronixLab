if True==False: from modularnet.modularspace.modularspace import ModularSpace

from calendar import c

import torch
from torch import nn

from modularnet.config.trial_config import TrialConfig

from modularnet.common.utils import  is_activation, is_conv, is_linear, is_layer
from modularnet.modularobserver.modularobserver import ModularObserver

class BaseLayerObserver:
    OBSERVABLE_LAYERS = {
        'conv': is_conv,
        'linear': is_linear,
    }

    @classmethod
    def layer_type(cls, module: nn.Module):
        for layer_type, check_fn in cls.OBSERVABLE_LAYERS.items():
            if check_fn(module):
                return layer_type
        return 'unknown'

    def __init__(self, observer: 'ModularObserver', observed:nn.Module, num_units: int, depth:int):
        self.observer = observer
        
        self.batch_size = None
        self.input_shape = None
        self.output_shape = None
        
        # space
        self.num_units = num_units 

        self.depth = depth
        
        
        #model layers
        self.observed = observed

        self.config = self.observer.config


        # hooks
        self.forward_hook = None
        self.backward_hook = None




        
        self.enable_hooks()
        
        
    def enable_hooks(self):
        self.disable_hooks()
        self.forward_hook = self.observed.register_forward_hook(self.forward_callback)
        self.backward_hook = self.observed.register_full_backward_hook(self.backward_callback)
        
    
    def disable_hooks(self):
        if self.forward_hook is not None: 
            self.forward_hook.remove()
            self.forward_hook = None
        
        if self.backward_hook is not None:
            self.backward_hook.remove()
            self.backward_hook = None
   

    def set_forward_masks(self, mask:torch.Tensor=None):
        if mask is None:
            self.forward_mask = None
        if len(mask) != self.num_units: 
            raise ValueError(f"Mask length {len(mask)} does not match input size {self.num_units}") 

        self.forward_mask = mask.detach().to(device=self.observed.device).clone()


    def infer_shape(self, x_in:torch.Tensor, x_out:torch.Tensor):
        if self.batch_size is None:
            self.batch_size = x_in[0].shape[0]
        if self.input_shape is None:
            self.input_shape = x_in[0].shape[1:]
        if self.output_shape is None:
            self.output_shape = x_out.shape[1:]

    def forward_callback(self, layer:nn.Module, x_in:torch.Tensor, x_out:torch.Tensor):
        # debug
        self.infer_shape(x_in, x_out)

        
        if self.forward_mask is not None:
            x_out_masked = x_out.clone() * self.forward_mask
            return x_out_masked
        

    def backward_callback(self, layer:nn.Module, grads_in:tuple[torch.Tensor], grads_out:tuple[torch.Tensor]):
        pass
