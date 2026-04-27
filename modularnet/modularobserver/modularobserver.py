if True==False: from modularnet.modularspace.modularspace import ModularSpace

import torch
from torch import nn

        

class ModularObserver():
    def __init__(self,  space: 'ModularSpace', num_units:int, observed:nn.Module, depth:int, device:torch.device):
        
        self.device = device
        
        self.batch_size = None
        self.input_shape = None
        self.output_shape = None
        
        # space
        self.num_units = num_units ##OLD: input_size
        self.space = space

        self.depth = depth
        self.pruning_mask = None 
        
        #model layers
        self.observed = observed
        self.observed_activation = None

        self.is_linear = isinstance(observed, nn.Linear)
        self.is_conv2d = isinstance(observed, nn.Conv2d)
         
        # debug
        self.debug = self.space.debug
        #self.plot_activations = False
        #torch.autograd.set_detect_anomaly(True)


        # stats
        #self.stats_positions_count = torch.zeros(input_size, requires_grad=False, device=self.device)
        #self.stats_data_count = 0

        ##self.stats_dist_on = []
        #self.stats_dist_center = []

        
        # metrics
        #self.metric_covariance = False
        #self.metric_count = True
        #self.metric_count_total = 0
        #self.activations = []
        #self.counts = {comb:0 for comb in combinations(range(self.input_size), 2)} 

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

    def value(self):
        return self.depth #math.log(1+self.depth)
    
   

    def set_pruning_mask(self, mask:torch.Tensor=None):
        if mask is None:
            self.pruning_mask = None
        if len(mask) != self.num_units: 
            raise ValueError(f"Mask length {len(mask)} does not match input size {self.num_units}") 

        self.pruning_mask = mask.detach().to(device=self.device).clone()


    def infer_shape(self, x_in:torch.Tensor, x_out:torch.Tensor):
        if self.batch_size is None:
            self.batch_size = x_in[0].shape[0]
        if self.input_shape is None:
            self.input_shape = x_in[0].shape[1:]
        if self.output_shape is None:
            self.output_shape = x_out.shape[1:]

    def forward_callback(self, layer:nn.Module, x_in:torch.Tensor, x_out:torch.Tensor):
        self.infer_shape(x_in, x_out)

        
        self.space.save_activations(self, x_out) 
        
        if self.pruning_mask is not None:
            x_out = x_out.clone() * self.pruning_mask
            return x_out

        
        
       


    def backward_callback(self, layer:nn.Module, grads_in:tuple[torch.Tensor], grads_out:tuple[torch.Tensor]):
        if not self.space.modularizer:return 
        sig_steep = 5

        mos_id = self.space.mos_id(self)
        layer_loss = self.space.last_space_loss.layers_loss[mos_id]

        # read precomputed distance to positive center
        center_dist = layer_loss.center_dist
        if center_dist is None: return 

        # normalize and invert distances 
        if self.space.config.modular_space_distance == 'euclidean':
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

        elif self.is_conv2d:
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
        
