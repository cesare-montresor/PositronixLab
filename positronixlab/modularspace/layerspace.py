
from typing import  Dict

import torch
from torch import Tensor
from positronixlab.spacemath.common import staticmethod_compile
from positronixlab.spacemath.spacemath import ActivationType, SpaceMath
from positronixlab.trial.trial_config import TrialConfig


class LayerSpace():
    
    def __init__(self, config: TrialConfig, observed_layers: dict[str, dict], device:torch.device=None, stats_enabled:bool=False):
        self.config = config
        self.observed_layers = observed_layers
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stats_enabled = stats_enabled

        self.positions: Dict[str, torch.Tensor] = {}
        self.stats: Dict[str, torch.Tensor] = {}
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}


        self._init_layers()
        self._init_space()

    def _init_layers(self):
        for layer_id, layer_info in self.observed_layers.items():
            act_type = layer_info.get("activation_type", None)
            act_fn = None if act_type is None else ActivationType.get_function(act_type)
            self.observed_layers[layer_id]["activation"] = act_fn

    def _init_space(self):
        for layer_id, layer_info in self.observed_layers.items():
            num_cus = layer_info["out_units"]

            # positions: [num_cus, modular_space_size]
            positions = torch.randn(num_cus, self.config.modular_space_size, device=self.device, requires_grad=True)
            torch._dynamo.mark_static_address(positions)
            self.positions[layer_id] = positions
            
            # stats: [num_cus, 4] for count_on, count_off, scores_on, scores_off
            if self.stats_enabled:
                stats = torch.zeros(num_cus, 4, device=self.device)
                torch._dynamo.mark_static_address(stats)
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
    
    def process_activations(self, layer_id: str, activations: torch.Tensor, labels: torch.Tensor|None )-> tuple[tuple, Tensor, Tensor | None]:
        layer_info = self.get_space(layer_id)[0]
        scores = self._process_activations(self.config.modular_space_active_q, activations, layer_info["layer_type"], layer_info["activation"])
        scores = scores.detach().clone()  # Detach and clone to avoid in-place modifications
        return self.process_scores(layer_id, scores)

    @staticmethod_compile
    #@staticmethod
    def _process_activations(active_q: float, activations: torch.Tensor, layer_type: str, activation_fn) -> Tensor:
        # [1 |0| -1]
        scores = SpaceMath.activations_to_scores( 
            activations, layer_type, activation_fn, 
            active_q, cnn_locality_weight=0.5)
        
        return scores
    

    def process_scores(self, layer_id: str, scores: torch.Tensor, labels: torch.Tensor|None):
        _, positions, optimizer, stats = self.get_space(layer_id)
        if not self.stats_enabled: stats = None
        losses = self._process_scores(self.config, scores, positions, stats, labels)
        loss= losses[0]  # total loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    

        loss_names = ["loss", "loss_on", "loss_off", "loss_centers"] 
        losses = {name : loss.detach().cpu().numpy() for name, loss in zip( loss_names, losses)}
        return losses, positions.detach().cpu().numpy(), None if stats is None else stats.detach().cpu().numpy() 

    @staticmethod_compile
    #@staticmethod
    def _process_scores(config: TrialConfig, scores: torch.Tensor, positions: torch.Tensor, stats: torch.Tensor, labels: torch.Tensor|None)-> tuple[torch.Tensor, tuple[ torch.Tensor, torch.Tensor, torch.Tensor]]:
        bs = scores.shape[0]
        
        # [True, False]
        idxs: tuple[torch.Tensor, torch.Tensor] = SpaceMath.split_active_group(scores)
        idxs_on, idxs_off = idxs

        # [0,1]
        scores_on = (scores * idxs_on).unsqueeze(2)
        scores_off = -(scores * idxs_off).unsqueeze(2)

        if stats is not None:
            stats[:, 0] += idxs_on.sum(dim=0).detach()                    # (N,)  
            stats[:, 1] += idxs_off.sum(dim=0).detach()                   # (N,)  
            stats[:, 2] += scores_on.sum(dim=0).squeeze(-1).detach()      # (N,1) -> (N,)
            stats[:, 3] += scores_off.sum(dim=0).squeeze(-1).detach()     # (N,1) -> (N,)

        
        # compute centers by weighted average of positions 
        batch_positions = positions.expand( bs, -1, -1 )
        
        # performing (batch_positions * weight_on/off).sum(dim=1) but more efficently
        scores_stack = torch.cat([scores_on, scores_off], dim=2)                    # (bs, N, 2)
        weights = scores_stack / scores_stack.sum(dim=1, keepdim=True).clamp(min=SpaceMath.eps)  # (bs, N, 2)
        centers = torch.bmm(weights.transpose(1, 2), batch_positions)               # (bs, 2, D)
        center_on, center_off = centers[:, 0], centers[:, 1]
       
        # LOSS: contastive CUs. On VS Off 
        batch_centers = center_on.unsqueeze(1).expand(-1, batch_positions.shape[1], -1)
        center_dist: torch.Tensor = SpaceMath.distance(batch_positions, batch_centers, dim=2, distance_type=config.modular_space_distance, pairwise=False)
        center_dist = center_dist.unsqueeze(2)  # (bs, N, 2)
        dists = torch.bmm(weights.transpose(1, 2), center_dist)
        
        dist_on, dist_off = dists[:, 0, 0], dists[:, 1, 0]


        loss_on = dist_on.mean() # mean for batch / sample 
        loss_off = dist_off.mean() # mean for batch / sample


        loss_on  = torch.log( 1 + loss_on ) * float(config.modular_space_loss_on)
        loss_off = torch.log( 1 + loss_off ) * float(config.modular_space_loss_off)
        loss = loss_on - loss_off

        # LOSS: center distance: break simmerity between on and off centers ( ex: random init uniform + random sampling )
        dist_centers: torch.Tensor = SpaceMath.distance(center_on, center_off, dim=1, distance_type=config.modular_space_distance) # on the same datapoint
        loss_centers = dist_centers.mean()
        loss -= loss_centers * float(config.modular_space_loss_centers)

  

        return loss, loss_on, loss_off, loss_centers


    