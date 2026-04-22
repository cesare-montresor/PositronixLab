if True==False: from modularnet.modularspace.modularspace import ModularSpace
if True==False: from modularnet.modularspace.modularobserver import ModularObserver

from collections import OrderedDict

import torch
from torch import nn
from torch import Tensor
import math
from modularnet.utils.utils import assert_tensor, info_tensor, is_conv
import einops
import torch.nn.functional as F

from modularnet.utils.torch_extras import spow_, spow, zclamp, zclamp_

class SpaceMath():
    eps = 1e-8
    inf = 1e8
    
    @classmethod
    def distance(cls, x:Tensor, y:Tensor, dim=1, distance_type=None, pairwise=True):
        distances = ['euclidean', 'cosine']
        if distance_type not in distances: distance_type='cosine'

        if distance_type == 'euclidean':
            if pairwise:
                dist = torch.cdist(x,y, p=2).squeeze()
            else:
                dist = zclamp(x - y).pow(2).sum(dim=dim).clamp(min=cls.eps).sqrt()

        elif distance_type == 'cosine':
            x_norm = F.normalize(x, p=2, dim=dim)
            y_norm = F.normalize(y, p=2, dim=dim)
            if pairwise:
                dist = 1 - torch.mm(x_norm, y_norm.T)
            else:
                dist = 1 - F.cosine_similarity(x_norm, y_norm, dim=dim)
        else:
            raise Exception(f"Unknown distance type: {distance_type}")
        
        #if len(dist.shape)>2:
            #dist.diagonal().zero_() # distance with itself should be zero, numerical stability
        
        dist = dist.clamp(min=0.0) # numerical stability: euclidean
        return dist
       

    @classmethod
    def activations_to_scores(cls, obs: 'ModularObserver', xd:Tensor, x_quantile=0.9, cnn_locality_weight=0.5):
        q = None
        act = obs.observed_activation
        
        if is_conv(obs.observed):
            q, scores = cls.get_quantile_k_kernels_per_sample(xd, x_quantile, locality_weight=cnn_locality_weight)[1].unsqueeze(1).detach()
            scores = cls.normalize_by_threshold(scores, q)
            return scores
                
        if isinstance(act, nn.Softmax): 
            q = torch.max(xd, dim=1).values.unsqueeze(1).detach()
            scores = torch.ones_like(xd) * -1.0
            scores[xd>=q] = 1.0
            return scores
        
        if act is not None:
            q = torch.quantile(act(xd), x_quantile, interpolation='higher', dim=1).unsqueeze(1).detach()
        else:
            q = torch.quantile(xd, x_quantile, interpolation='higher', dim=1).unsqueeze(1).detach()

        scores = cls.normalize_by_threshold(xd, q)

        return scores
    
    @classmethod
    def normalize_by_threshold(cls, scores:Tensor, q:Tensor):
        idxs_on, idxs_off = cls.split_active_group(scores, threshold=q)
        scores_on = scores * idxs_on # keep X >= q untouched
        scores_off = ((scores - q) - cls.eps) * idxs_off # brings X < q below zero
        scores_max = torch.abs(torch.amax(scores_on,dim=1)).unsqueeze(1) + cls.eps
        scores_min = torch.abs(torch.amin(scores_off,dim=1)).unsqueeze(1) + cls.eps
        scores = (scores_on / scores_max ) + (scores_off / scores_min )
        scores = scores.detach()
        return scores

    @classmethod
    def split_active_group(cls,x:Tensor, threshold=0.0):
        idxs_on  = (x >= threshold)
        idxs_off = (~idxs_on)
        return idxs_on, idxs_off
    

    @classmethod
    def count_active(cls, idxs_on, idxs_off):
        cnt_on = idxs_on.sum(dim=1)
        cnt_off = idxs_off.sum(dim=1)
        
        good = True
        if (cnt_on==0).any():
            print("WARNING: No active neurons")
            good = False

        if (cnt_off==0).any():
            print("WARNING: No inactive neurons")
            good = False
            
        return good

    

    @classmethod
    def rank_kernels_by_activation_per_sample(  cls,
                                                feature_maps: torch.Tensor,
                                                locality_weight: float = 0.5,
                                                spatial_radius: int = 3,
                                                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fast GPU-optimized method to rank convolutional kernels for each sample in the batch.
        
        Args:
            feature_maps: Tensor of shape (batch_size, num_kernels, height, width)
            locality_weight: Weight factor for locality score (0 to 1)
            spatial_radius: Radius for computing local response
        
        Returns:
            sorted_indices: Indices of kernels sorted by importance for each sample
                        Shape: (batch_size, num_kernels)
            kernel_scores: Corresponding importance scores for each sample
                        Shape: (batch_size, num_kernels)
        """
        B, C, H, W = feature_maps.shape
        
        # 1. Compute Per-Sample Activation Magnitude
        # Using L2 norm for each sample separately
        magnitude_scores = torch.norm(feature_maps.reshape(B, C, -1), dim=2)  # Shape: (B, C)
        
        # 2. Compute Local Response Concentration per sample
        padding = spatial_radius // 2
        max_pooled = F.max_pool2d(
            feature_maps,
            kernel_size=spatial_radius,
            stride=1,
            padding=padding
        )
        
        # Compute ratio of local maximum to total activation for each sample
        local_response = (max_pooled.abs() / (torch.sum(feature_maps.abs(), dim=(2, 3), keepdim=True) + cls.eps))
        locality_scores = local_response.mean(dim=(2, 3))  # Shape: (B, C)
        
        # 3. Normalize scores per sample
        magnitude_scores = magnitude_scores / (magnitude_scores.max(dim=1, keepdim=True)[0] + cls.eps)
        locality_scores = locality_scores / (locality_scores.max(dim=1, keepdim=True)[0] + cls.eps)
        
        # 4. Compute final scores per sample
        combined_scores = ((1 - locality_weight) * magnitude_scores) + (locality_weight * locality_scores)
        
        # 5. Sort independently for each sample
        sorted_indices = torch.argsort(combined_scores, dim=1, descending=True)
        
        return sorted_indices, combined_scores

    @classmethod
    def get_quantile_k_kernels_per_sample(cls,
                                feature_maps: torch.Tensor,
                                quantile: float,
                                locality_weight: float = 0.5,
                                spatial_radius: int = 3) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns kernels above the quantile threshold for each sample.
        
        Args:
            feature_maps: Tensor of shape (batch_size, num_kernels, height, width)
            quantile: Float between 0 and 1, threshold for kernel selection
            locality_weight: Weight factor for locality score
            spatial_radius: Radius for computing local response
            
        Returns:
            top_k_indices: Indices of selected kernels per sample
                        Shape: (batch_size, k), where k varies by quantile
            top_k_scores: Scores of selected kernels per sample
                        Shape: (batch_size, k)
            top_k_features: Feature maps of selected kernels per sample
                        Shape: (batch_size, k, height, width)
        """
        sorted_indices, scores = cls.rank_kernels_by_activation_per_sample(
            feature_maps, locality_weight, spatial_radius
        )
        
        # Calculate quantile threshold for each sample
        thresholds = torch.quantile(scores, quantile, dim=1)
        
        # Create mask for selected kernels and apply to sorted indices and scores
        selected_mask = scores >= thresholds.unsqueeze(1)
        k = selected_mask.sum(dim=1).max().item()  # Get max number of selected kernels
        
        # Get top-k indices and scores using the mask
        top_k_indices = torch.zeros((feature_maps.shape[0], k), 
                                dtype=torch.long, 
                                device=feature_maps.device)
        top_k_scores = torch.zeros((feature_maps.shape[0], k), 
                                device=feature_maps.device)
        
        for b in range(feature_maps.shape[0]):
            sample_indices = torch.where(selected_mask[b])[0]
            num_selected = len(sample_indices)
            top_k_indices[b, :num_selected] = sorted_indices[b, sample_indices]
            top_k_scores[b, :num_selected] = scores[b, sample_indices]
        
        # Get corresponding feature maps
        batch_indices = torch.arange(feature_maps.shape[0], device=feature_maps.device)
        batch_indices = batch_indices.view(-1, 1).expand(-1, k)
        # top_k_features = feature_maps[batch_indices, top_k_indices]

        


        return top_k_indices, top_k_scores #, top_k_features
    



