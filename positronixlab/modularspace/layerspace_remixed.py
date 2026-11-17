import math
import torch
import torch.nn.functional as F

from positronixlab.spacemath.common import staticmethod_compile
from positronixlab.spacemath.spacemath import SpaceMath
from positronixlab.spacemath.sigreg import sigreg
from positronixlab.pipeline.pipeline_config import PipelineConfig

class LayerSpaceMixin:

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