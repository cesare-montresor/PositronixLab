
from collections import OrderedDict

import torch
from torch import nn
from torch import Tensor
import math
from modularnet.utils.utils import assert_tensor, info_tensor, is_conv
import einops
from modularnet.spacemath.spacemath import SpaceMath

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modularnet.modularspace.modularspace import ModularSpace
    from modularnet.modularobserver.modularobserver import ModularObserver




class SpaceLoss(nn.Module):
    eps = 1e-8

    def __init__(self, space:'ModularSpace', quantile_x=0.9, debug = False, loss_dist_center=True, loss_dist_batch=True, batch_id:str=None):
        super().__init__()
        
        self.config = space.config
        self.space = space
        self.loss_dist_center = loss_dist_center
        self.loss_dist_batch = loss_dist_batch
        self.debug = debug
        
        self.did_forward = False
        self.quantile_x = quantile_x if quantile_x is not None else self.config.modular_space_active_q
        

        self.loss = None
        self.layers_loss = OrderedDict() # type:  OrderedDict[str, LayerLoss]
        for i, mo in enumerate(self.space.mos):
            layer_criterion = LayerLoss(self, mo, quantile_x=quantile_x, depth=mo.depth, debug=debug)
            mos_id = self.space.mos_id(mo)
            self.layers_loss[mos_id] = layer_criterion
        
        self.batch_id = batch_id    
        
        self.reset()

    def reset(self):
        self.did_forward = False
        self.loss = None
        for layer_loss in self.layers_loss.values():
            layer_loss.reset()

    def activations_layer(self, mos_id:str, x:Tensor):
        layer_loss = self.layers_loss[mos_id]
        return layer_loss.add_activation(x)

    def forward_all_layers(self): 
        if not self.can_forward(): return False
        for layer_loss in self.layers_loss.values():
            layer_loss.forward()
        return True

    def forward_layer(self, mos_id:str):
        layer_loss = self.layers_loss[mos_id]
        return layer_loss.forward()

    def forward(self):
        self.forward_all_layers()
        if self.did_forward: return self.loss

        layer_losses =  list(self.layers_loss.values()) # [mo.criterion for mo in mos]
        last_losses = [ll.loss for ll in layer_losses]
        centers_on = [layer_loss.center_on for layer_loss in layer_losses]
            
        centers_on = torch.vstack(centers_on).reshape(len(centers_on), -1, self.space.space_size)
        active_cu = einops.rearrange(centers_on, ' b l s -> l b s')
        #if self.debug:info_tensor(good, "good")
        active_centers = torch.mean(active_cu, dim=1).unsqueeze(1)#.detach()
        #if self.debug:info_tensor(good_centers, "good_centers")


        if self.config.modular_space_distance == 'euclidean':
            assert_tensor(active_centers, "active_centers")
            assert_tensor(active_cu, "active_cu")
            align_dist = (active_centers - active_cu).spow(2).sum(dim=1).clamp(min=self.eps).sqrt()
            assert_tensor(align_dist, "align_dist")
        else:
            active_cu_n = torch.nn.functional.normalize(active_cu, dim=2, eps=self.eps)
            active_centers_n = torch.nn.functional.normalize(active_centers, dim=2, eps=self.eps)
            align_dist = 1 - (active_cu_n * active_centers_n).sum(dim=2)
        
        
        
        #if self.debug:info_tensor(good_dist, "good_dist")
        alignment_loss = align_dist.mean()
        loss = torch.vstack(last_losses)
        loss = loss.sum() / len(last_losses)
        #if self.debug:info_tensor(last_loss, "layer_loss")

        align_factor = self.config.modular_space_loss_alignment

        self.loss = loss + (alignment_loss * align_factor)
        self.did_forward = True
        return self.loss

    
    def can_forward(self):
        return all( layer_loss.did_forward or layer_loss.can_forward() for layer_loss in self.layers_loss.values() )

class LayerLoss(nn.Module):
    eps = 1e-8
    inf = 1e8

    def __init__(self, space_loss: SpaceLoss, observer:'ModularObserver', quantile_x=0.8, depth = 1, debug = False, cnn_locality_weight=0.8):
        super().__init__()

        self.space_loss = space_loss
        self.config = space_loss.config
        self.space = self.space_loss.space

        self.observer = observer
        
        self.cnn_locality_weight = cnn_locality_weight # TODO: bind to config for CNN
        self.quantile_x = quantile_x
        self.depth = depth
        
        self.debug = debug
        self.did_activate = False
        self.did_forward = False
        self.reset()


    def reset(self):
        self.did_activate = False
        self.did_forward = False
        self.scores = None
        self.bs = None
        self.idxs_on = None
        self.idxs_off = None
        self.vals_on = None  
        self.vals_off = None
        self.weight_on = None
        self.weight_off = None
        
        
        self.center_on = None
        self.center_off = None
        self.center_dist = None
        self.dist_on = None
        self.dist_off = None

        self.loss_on = None
        self.loss_off = None
        self.loss_centers = None
        self.loss_batch = None
        self.loss = None
        
        
    def add_activation(self, x: Tensor):
        
        xd = x.detach()
        self.bs = xd.shape[0]
        
        # [-1,0-,1]
        self.scores = SpaceMath.activations_to_scores(self.observer, xd, self.quantile_x, self.cnn_locality_weight)
        
        # [True, False]
        idxs_on, idxs_off = SpaceMath.split_active_group(self.scores)

        # [0,1]
        self.vals_on = (self.scores * idxs_on).unsqueeze(2)
        self.vals_off = -(self.scores * idxs_off).unsqueeze(2)

        self.idxs_on = idxs_on.unsqueeze(2)
        self.idxs_off = idxs_off.unsqueeze(2)

        self.weight_on = (self.vals_on / self.vals_on.sum(dim=1, keepdim=True)).nan_to_num(nan=0.0, posinf=self.inf, neginf=-self.inf)
        self.weight_off = (self.vals_off / self.vals_off.sum(dim=1, keepdim=True)).nan_to_num(nan=0.0, posinf=self.inf, neginf=-self.inf)

        self.did_activate = True
        
        
    def forward(self):
        if not self.can_forward(): return None
        if self.did_forward: return self.loss

        cfg = self.config
        mos_idx = self.space.mos.index(self.observer) 
        layer_positions = self.space.positions[mos_idx]
        
        
        # compute centers by weighted average of positions 
        #layer_positions = torch.nn.functional.normalize(layer_positions, dim=1, eps=self.eps)
        batch_positions = layer_positions.expand( self.bs, -1, -1 )
        self.center_on = (batch_positions * self.weight_on).sum(dim=1)
        self.center_off = (batch_positions * self.weight_off).sum(dim=1)

        # LOSS: contastive CUs. On VS Off 
        batch_centers = self.center_on.unsqueeze(1).expand(-1, batch_positions.shape[1], -1)
        self.center_dist = SpaceMath.distance(batch_positions, batch_centers, dim=2, distance_type=cfg.modular_space_distance, pairwise=False).unsqueeze(1)
        
        self.dist_on = (self.center_dist * self.weight_on).sum(dim=1)
        self.dist_off = (self.center_dist * self.weight_off).sum(dim=1)


        self.loss_on = self.dist_on.mean() # mean for batch / sample 
        self.loss_off = self.dist_off.mean() # mean for batch / sample


        self.loss_on = torch.log( 1+self.loss_on ) * cfg.modular_space_loss_on
        self.loss_off = torch.log(1+self.loss_off) * cfg.modular_space_loss_off
        self.loss = self.loss_on - self.loss_off

        # LOSS: center distance: break simmerity between on and off centers ( ex: random init uniform + random sampling )
        if self.space_loss.loss_dist_center:
            dist_centers = SpaceMath.distance(self.center_on, self.center_off, dim=1, distance_type=cfg.modular_space_distance) # on the same datapoint
            self.loss_centers = dist_centers.mean()
            self.loss -= self.loss_centers * cfg.modular_space_loss_centers

        # LOSS: batch distance: break simmilarity between different samples on the same centers (supposed to be weak, at the time of writing)
        if self.space_loss.loss_dist_batch:
            batch_dist = SpaceMath.distance(self.center_on, self.center_on, dim=0, distance_type=cfg.modular_space_distance) # between datapoints in the batch 
            self.loss_batch = batch_dist.mean()
            self.loss -= self.loss_batch * cfg.modular_space_loss_batch

        self.loss *= 1+math.log(1+self.depth) # increase weight with depth, higher level correlations are more relevant
        

        if self.debug:
            print(self.loss_on, self.loss_off, self.loss_centers, self.loss_batch)
            print(self.loss_on, self.loss_off, self.loss_centers, self.loss_batch)
            self.check_tensors()

        self.did_forward = True
        return self.loss

    def check_tensors(self):
        assert_tensor(self.scores, "scores")
        assert_tensor(self.idxs_on, "idxs_on")
        assert_tensor(self.idxs_off, "idxs_off")
        assert_tensor(self.vals_on, "vals_on")
        assert_tensor(self.vals_off, "vals_off")
        assert_tensor(self.weight_on, "weight_on")
        assert_tensor(self.weight_off, "weight_off")
        assert_tensor(self.center_on, "center_on")
        assert_tensor(self.center_off, "center_off")
        assert_tensor(self.center_dist, "center_dist")
        assert_tensor(self.dist_on, "dist_on")
        assert_tensor(self.dist_off, "dist_off")
        assert_tensor(self.loss_on, "loss_on")
        assert_tensor(self.loss_off, "loss_off")
        if self.space_loss.loss_dist_center:
            assert_tensor(self.loss_centers, "loss_centers")
        if self.space_loss.loss_dist_batch:
            assert_tensor(self.loss_batch, "loss_batch")
        

    
    def can_forward(self):
        return self.did_activate
        