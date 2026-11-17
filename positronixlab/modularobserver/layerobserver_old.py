if True==False: from positronixlab.modularspace.modularspace import ModularSpace

from calendar import c

import torch
from torch import nn

from positronixlab.pipeline import PipelineConfig

from positronixlab.modularobserver.layerobserver import LayerObserver
from positronixlab.modularobserver.modularobserver import ModularObserver



class LayerObserverPubSub(LayerObserver):

    def __init__(self, observer: 'ModularObserver', num_units: int, observed:nn.Module, depth:int):
        super().__init__(observer, observed, num_units, depth)
        
        self.is_linear = self.get_layer_type(self.layer) == 'linear'
        self.is_conv = self.get_layer_type(self.layer) == 'conv'

        

    def forward_callback(self, layer:nn.Module, x_in:torch.Tensor, x_out:torch.Tensor):
        
        x_out_masked = super().forward_callback(layer, x_in, x_out)
        
        


        if x_out_masked is not None:
            return x_out_masked
        
        
       


    def backward_callback(self, layer:nn.Module, grads_in:tuple[torch.Tensor], grads_out:tuple[torch.Tensor]):
        if not self.observer.config.modularizer:return 
        sig_steep = 5

        mos_id = self.observer.layer_id(self)
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
            fc = self.layer
            if fc.weight.grad is not None:
                fc.weight.grad *= grad_scale.unsqueeze(1)
            if fc.bias is not None and fc.bias.grad is not None:
                fc.bias.grad *= grad_scale

        elif self.is_conv:
            conv = self.layer
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
        



    
    def set_forward_masks(self, mask:torch.Tensor=None):
        if mask is None:
            self.forward_mask = None
        if len(mask) != self.num_units: 
            raise ValueError(f"Mask length {len(mask)} does not match input size {self.num_units}") 

        self.forward_mask = mask.detach().to(device=self.layer.device).clone()


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

    
    