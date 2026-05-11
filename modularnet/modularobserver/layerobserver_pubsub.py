if True==False: from modularnet.modularspace.modularspace import ModularSpace

from calendar import c

import torch
from torch import nn

from modularnet.config.trial_config import TrialConfig

from modularnet.common.utils import  is_activation, is_conv, is_linear, is_layer
from modularnet.modularobserver.layerobserver_base import BaseLayerObserver
from modularnet.modularobserver.modularobserver import ModularObserver
from modularnet.plugin.plugin_transport import ActivationTap


class LayerObserverPubSub(BaseLayerObserver):

    def __init__(self, observer: 'ModularObserver', num_units: int, observed:nn.Module, depth:int):
        super().__init__(observer, observed, num_units, depth)
        
        self.is_linear = self.layer_type(self.observed) == 'linear'
        self.is_conv = self.layer_type(self.observed) == 'conv'
         
        self.publisher = ActivationTap(observed, self.observer.config.activation_tap_mode, self.observer.config.activation_tap_buffer_size)
        # debug
        

    def forward_callback(self, layer:nn.Module, x_in:torch.Tensor, x_out:torch.Tensor):
        
        x_out_masked = super().forward_callback(layer, x_in, x_out)
        
        


        if x_out_masked is not None:
            return x_out_masked
        
        
       


    def backward_callback(self, layer:nn.Module, grads_in:tuple[torch.Tensor], grads_out:tuple[torch.Tensor]):
        if not self.observer.config.modularizer:return 
        sig_steep = 5

        mos_id = self.observer.mos_id(self)
        layer_loss = self.observer.last_space_loss.layers_loss[mos_id]

        # read precomputed distance to positive center
        center_dist = layer_loss.center_dist
        if center_dist is None: return 

        # normalize and invert distances 
        if self.observer.config.modular_space_distance == 'euclidean':
            center_dist_norm = 1 - torch.sigmoid((center_dist * sig_steep) - sig_steep / 2)
        else:
            center_dist_norm = 1 - center_dist  # cosine distances are already in [0,1]


        # combine with rescale_factor
        center_dist_norm = center_dist_norm.squeeze(1)  # make broadcastable
        grad_scale = torch.lerp(torch.ones_like(center_dist_norm), center_dist_norm, self.space.rescale_factor)
        grad_scale = grad_scale.mean(dim=0)

        return

        print("oook")
        # apply scaling to gradients
        if self.is_linear:
            fc = self.observed
            if fc.weight.grad is not None:
                fc.weight.grad *= grad_scale.unsqueeze(1)
            if fc.bias is not None and fc.bias.grad is not None:
                fc.bias.grad *= grad_scale

        elif self.is_conv:
            conv = self.observed
            if conv.weight.grad is not None:
                conv.weight.grad *= grad_scale.view(-1, 1, 1, 1)
            if conv.bias is not None and conv.bias.grad is not None:
                conv.bias.grad *= grad_scale

        print("nooook!")
        #in_shape = grads_in[0].shape
        #out_shape = grads_out[0].shape
        #center_shape = center_dist_norm.shape
        #weight_shape = self.observed.weight.shape
        #bias_shape = self.observed.bias.shape

        #print(in_shape, out_shape, center_shape, weight_shape)
        
