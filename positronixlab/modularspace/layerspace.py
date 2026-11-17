
from builtins import staticmethod
from typing import  Dict

import math
import torch
from torch import Tensor
from torch.nn import functional as F
from positronixlab.spacemath.common import staticmethod_compile
from positronixlab.spacemath.spacemath import ActivationType, SpaceMath
from positronixlab.pipeline import PipelineConfig
from positronixlab.spacemath.sigreg import sigreg


class LayerSpace():
    
    def __init__(self, config: PipelineConfig, observed_layers: dict[str, dict], device:torch.device=None, stats_enabled:bool=False):
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
    
    def process_activations(self, layer_id: str, activations: torch.Tensor, clusters_labels: torch.Tensor|None )-> tuple[tuple, Tensor, Tensor | None]:
        layer_info = self.get_space(layer_id)[0]
        scores = self._process_activations(self.config.modular_space_active_q, activations, layer_info["layer_type"], layer_info["activation"])
        scores = scores.detach().clone()  # Detach and clone to avoid in-place modifications
        return self.process_scores(layer_id, scores, clusters_labels)

    @staticmethod_compile
    #@staticmethod
    def _process_activations(active_q: float, activations: torch.Tensor, layer_type: str, activation_fn) -> Tensor:
        # [1 |0| -1]
        scores = SpaceMath.activations_to_scores( 
            activations, layer_type, activation_fn, 
            active_q, cnn_locality_weight=0.5)
        
        return scores
    
    
    def process_scores(self, layer_id: str, scores: torch.Tensor, clusters_labels: torch.Tensor | None):
        _, positions, optimizer, stats = self.get_space(layer_id)
        if not self.stats_enabled:
            stats = None
            
        membership = self._membership(clusters_labels, positions.device, positions.dtype)
        
        losses = self._process_scores(self.config, scores, positions, stats, membership)
        loss = losses[0]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_names = ["loss", "loss_on", "loss_off", "loss_cluster", "loss_sigreg"]
        losses_dict = {name: l.detach().cpu().numpy() for name, l in zip(loss_names, losses)}
        return losses_dict, positions.detach().cpu().numpy(), None if stats is None else stats.detach().cpu().numpy()

    @staticmethod
    def _membership(labels, device, dtype):
        """Crea una maschera one-hot (K, N) basata sulle etichette dei cluster."""
        if labels is None:
            return None
        lab = torch.as_tensor(labels, device=device).long()
        K = int(lab.max().item()) + 1
        if K < 2:
            return None
        M = F.one_hot(lab.clamp_min(0), K).T.to(dtype)
        return M * (lab >= 0).to(dtype).unsqueeze(0)  # Esclude le etichette di rumore (-1)
    
    @staticmethod_compile
    def _contrast_vectorized(config: PipelineConfig, scores: torch.Tensor, positions: torch.Tensor, M: torch.Tensor):
        """
        Compiled static graph for heavy tensor algebra. 
        All shapes here (bs, N, K, D) are strictly static.
        """
        bs, N = scores.shape
        D = positions.shape[1]
        
        idxs_on, idxs_off = SpaceMath.split_active_group(scores)
        w_on_base = scores * idxs_on                    
        w_off_base = -(scores * idxs_off)               
        
        # Broadcasting: (bs, K, N)
        w_on = w_on_base.unsqueeze(1) * M.unsqueeze(0)
        w_off = w_off_base.unsqueeze(1) * M.unsqueeze(0)
        
        mass_on = w_on.sum(2)
        mass_off = w_off.sum(2)
        mixed = ((mass_on > SpaceMath.eps) & (mass_off > SpaceMath.eps)).to(scores.dtype)
        
        w_on_norm = w_on / mass_on.clamp(min=SpaceMath.eps).unsqueeze(2)
        w_off_norm = w_off / mass_off.clamp(min=SpaceMath.eps).unsqueeze(2)
        
        batch_pos = positions.unsqueeze(0).expand(bs, -1, -1)
        centers_on = torch.bmm(w_on_norm, batch_pos)
        
        # Inlined distance calculation to avoid external graph breaks
        is_cosine = str(config.modular_space_distance).lower().startswith("cos")
        if is_cosine:
            pos_norm = F.normalize(positions, p=2, dim=-1)
            cent_norm = F.normalize(centers_on, p=2, dim=-1)
            sim = torch.matmul(cent_norm, pos_norm.T)  # (bs, K, N)
            d = 1.0 - sim
        else:
            d = torch.cdist(centers_on, positions)     # (bs, K, N)
            
        dist_on = (w_on_norm * d).sum(2)
        dist_off = (w_off_norm * d).sum(2)
        
        m_count = mixed.sum().clamp(min=1.0)
        loss_on = (dist_on * mixed).sum() / m_count
        loss_off = (dist_off * mixed).sum() / m_count
        
        return loss_on, loss_off, centers_on, mixed

    def _process_scores(self, config: PipelineConfig, scores: torch.Tensor, positions: torch.Tensor, stats: torch.Tensor, membership: torch.Tensor | None):
        """
        Eager execution loop. Handles dynamic masking for SIGReg and stat updates.
        """
        bs, N = scores.shape
        D = positions.shape[1]
        
        if stats is not None:
            idxs_on, idxs_off = SpaceMath.split_active_group(scores)
            stats[:, 0] += idxs_on.sum(dim=0).detach()
            stats[:, 1] += idxs_off.sum(dim=0).detach()
            stats[:, 2] += (scores * idxs_on).sum(dim=0).detach()
            stats[:, 3] += (-(scores * idxs_off)).sum(dim=0).detach()
            
        # 1) Global Contrast (Calls compiled graph)
        M_global = torch.ones(1, N, device=positions.device, dtype=positions.dtype)
        l_on_g, l_off_g, centers_on_g, mixed_g = self._contrast_vectorized(config, scores, positions, M_global)
        
        l_on = torch.log(1 + l_on_g) * float(config.modular_space_loss_on)
        l_off = torch.log(1 + l_off_g) * float(config.modular_space_loss_off)
        loss = l_on - l_off
        
        centers_for_sigreg = centers_on_g
        mixed_for_sigreg = mixed_g
        loss_cluster = scores.new_zeros(())
        
        # 2) Cluster Contrast (Calls compiled graph)
        w_c = float(getattr(config, "modular_space_loss_cluster", 0.0))
        if membership is not None and w_c > 0:
            l_on_c, l_off_c, centers_on_c, mixed_c = self._contrast_vectorized(config, scores, positions, membership)
            
            l_on_c = torch.log(1 + l_on_c) * float(config.modular_space_loss_on)
            l_off_c = torch.log(1 + l_off_c) * float(config.modular_space_loss_off)
            loss_cluster = l_on_c - l_off_c
            loss = loss + w_c * loss_cluster
            
            centers_for_sigreg = centers_on_c
            mixed_for_sigreg = mixed_c
            
        # 3) SIGReg (Dynamic shape logic stays in eager mode)
        loss_sigreg = scores.new_zeros(())
        w_s = float(getattr(config, "modular_space_loss_sigreg", 0.0))
        if w_s > 0:
            active_mask = mixed_for_sigreg > 0
            if active_mask.sum() > 1:
                active_centers = centers_for_sigreg[active_mask]  # Dynamic shape triggers graph break if compiled
                
                is_cosine = str(config.modular_space_distance).lower().startswith("cos")
                if is_cosine:
                    X_sig = F.normalize(active_centers, dim=-1) * math.sqrt(D)
                else:
                    X_sig = active_centers
                    
                loss_sigreg = sigreg(X_sig, method=str(config.modular_space_sigreg_method))
                loss = loss + w_s * loss_sigreg

        return loss, l_on, l_off, loss_cluster, loss_sigreg