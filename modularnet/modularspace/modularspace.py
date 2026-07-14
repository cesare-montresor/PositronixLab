
from typing import  Dict

import torch
from modularnet.spacemath.spacemath import ActivationType, SpaceMath, compile
from modularnet.trial.trial_config import TrialConfig


class ModularSpace():
    
    def __init__(self, config: TrialConfig, observed_layers: dict[str, dict], device:torch.device=None, stats_enabled:bool=False):
        self.config = config
        self.observed_layers = observed_layers
        self.device = device if device else torch.device("cpu")
        self.stats_enabled = stats_enabled

        self.positions: Dict[str, torch.Tensor] = {}
        self.stats: Dict[str, torch.Tensor] = {}
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}


        self._init_layers()
        self._init_space()

    def _init_layers(self):
        for layer_id, layer_info in self.observed_layers.items():
            act_type = layer_info.get("activation_type", None)
            if act_type is None: continue
            self.observed_layers[layer_id]["activation"] = ActivationType.get_function(act_type)

    def _init_space(self):
        for layer_id, layer_info in self.observed_layers.items():
            num_cus = layer_info["num_units"]

            # positions: [num_cus, modular_space_size]
            positions = torch.randn(num_cus, self.config.modular_space_size, device=self.device, requires_grad=True)
            self.positions[layer_id] = positions
            
            # stats: [num_cus, 4] for count_on, count_off, scores_on, scores_off
            if self.stats_enabled:
                stats = torch.zeros(num_cus, 4, device=self.device)
                self.stats[layer_id] = stats
            
            # optimizer: dedicated for each positions
            self.optimizers[layer_id] = torch.optim.Adam(
                (positions,),
                lr=self.config.modular_space_lr,
            )
    
    def get_space(self, layer_id: str):
        layer_info = self.observed_layers.get(layer_id, None)
        positions = self.positions.get(layer_id, None)
        stats = self.stats.get(layer_id, None)
        optimizer = self.optimizers.get(layer_id, None)
        if None in (layer_info, positions, optimizer, stats):
            raise ValueError(f"Layer {layer_id} not found in ModularSpace.")
        return layer_info, positions, optimizer, stats
    
    def process_activations(self, activations: torch.Tensor, layer_id: str):
        layer_info, positions, optimizer, stats = self.get_space(layer_id)
        scores = self._process_activations(activations, layer_info["layer_type"], layer_info["activation"])
        return self.process_scores(scores, positions, optimizer, stats)

    @compile
    def _process_activations(self, activations: torch.Tensor, layer_type: str, activation_fn):
        # [1 |0| -1]
        scores = SpaceMath.activations_to_scores( 
            activations, layer_type, activation_fn, 
            self.config.modular_space_active_q, cnn_locality_weight=0.5)
        
        return scores
    
    @compile
    def process_scores(self, scores: torch.Tensor, positions: torch.Tensor, optimizer: torch.optim.Optimizer, stats: torch.Tensor):
        bs = scores.shape[0]
        
        # [True, False]
        idxs: tuple[torch.Tensor, torch.Tensor] = SpaceMath.split_active_group(scores)
        idxs_on, idxs_off = idxs

        # [0,1]
        scores_on = (scores * idxs_on).unsqueeze(2)
        scores_off = -(scores * idxs_off).unsqueeze(2)

        if self.stats_enabled:
            stats[:, 0] += idxs_on.sum(dim=0).detach()  # count_on
            stats[:, 1] += idxs_off.sum(dim=0).detach()  # count_off
            stats[:, 2] += scores_on.sum(dim=0).detach()  # scores_on
            stats[:, 3] += scores_off.sum(dim=0).detach()  # scores_off


        idxs_on = idxs_on.unsqueeze(2)
        idxs_off = idxs_off.unsqueeze(2)
        
        # compute centers by weighted average of positions 
        batch_positions = positions.expand( bs, -1, -1 )
        
        # performing (batch_positions * weight_on/off).sum(dim=1) but more efficently
        scores_stack = torch.cat([scores_on, scores_off], dim=2)                    # (bs, N, 2)
        weights = scores_stack / scores_stack.sum(dim=1, keepdim=True).clamp(min=self.eps)  # (bs, N, 2)
        centers = torch.bmm(weights.transpose(1, 2), batch_positions)               # (bs, 2, D)
        center_on, center_off = centers[:, 0], centers[:, 1]
       
        # LOSS: contastive CUs. On VS Off 
        batch_centers = center_on.unsqueeze(1).expand(-1, batch_positions.shape[1], -1)
        center_dist = SpaceMath.distance(batch_positions, batch_centers, dim=2, distance_type=self.config.modular_space_distance, pairwise=False)
        center_dist = center_dist.unsqueeze(2)  # (bs, N, 2)
        dists = torch.bmm(weights.transpose(1, 2), center_dist)
        
        dist_on, dist_off = dists[:, 0, 0], dists[:, 1, 0]


        loss_on = dist_on.mean() # mean for batch / sample 
        loss_off = dist_off.mean() # mean for batch / sample


        loss_on  = torch.log( 1 + loss_on ) * float(self.config.modular_space_loss_on)
        loss_off = torch.log( 1 + loss_off ) * float(self.config.modular_space_loss_off)
        loss = loss_on - loss_off

        # LOSS: center distance: break simmerity between on and off centers ( ex: random init uniform + random sampling )
        dist_centers = SpaceMath.distance(center_on, center_off, dim=1, distance_type=self.config.modular_space_distance) # on the same datapoint
        loss_centers = dist_centers.mean()
        loss -= loss_centers * float(self.config.modular_space_loss_centers)

        # LOSS: batch distance: break simmetry between different samples on the same centers (supposed to be weak, at the time of writing)
        # batch_dist = SpaceMath.distance(center_on, center_on, dim=0, distance_type=self.config.modular_space_distance) # between datapoints in the batch 
        # loss_batch = batch_dist.mean()
        # loss -= loss_batch * float(self.config.modular_space_loss_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    

        return (loss, loss_on, loss_off, loss_centers), (positions, scores)


    