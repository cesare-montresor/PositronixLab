"""
Minimal change to your original _process_scores:
  1. remove center_off / loss_centers
  2. repeat the same on/off contrast once per cluster (units of that cluster only)
  3. add sigreg on the positions

New PipelineConfig fields:
    modular_space_loss_cluster   float   weight of the per-cluster term
    modular_space_loss_sigreg    float   weight of sigreg (0 = off)
    modular_space_sigreg_method  str     "weak" | "strong"
Old modular_space_loss_centers is no longer used.

Not run against your stack. Note: no @staticmethod_compile on _process_scores any more
(the cluster loop has data-dependent shapes). If you want compile, put it on _contrast only.
"""
import math
import torch
import torch.nn.functional as F

# from .sigreg import sigreg
# from ... import SpaceMath


from typing import  Dict

import torch
from torch import Tensor
from positronixlab.spacemath.common import staticmethod_compile
from positronixlab.spacemath.spacemath import  SpaceMath
from positronixlab.spacemath.sigreg import sigreg
from positronixlab.pipeline.pipeline_config import PipelineConfig



class LayerSpaceMixin:  # your real class

    def process_scores(self, layer_id: str, scores: torch.Tensor, clusters_labels: torch.Tensor | None):
        _, positions, optimizer, stats = self.get_space(layer_id)
        if not self.stats_enabled:
            stats = None
        losses = self._process_scores(self.config, scores, positions, stats, clusters_labels)
        loss = losses[0]  # total loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_names = ["loss", "loss_on", "loss_off", "loss_cluster", "loss_sigreg"]
        losses = {name: l.detach().cpu().numpy() for name, l in zip(loss_names, losses)}
        return losses, positions.detach().cpu().numpy(), None if stats is None else stats.detach().cpu().numpy()

    # ------------------------------------------------------------------
    @staticmethod
    def _contrast(config, scores: torch.Tensor, positions: torch.Tensor):
        """
        Your original contrast, applied to one group of units.
          scores    (bs, n)  in [-1, 1]  (>=0 active, <0 inactive)
          positions (n, D)
        Returns (loss_on, loss_off): batch means of
          dist_on  = weighted spread of the active units around the active centre
          dist_off = weighted distance of the inactive units from the active centre
        Datapoints where the group is all-active or all-inactive are skipped
        (a pure group has nothing to split).
        """
        bs, n = scores.shape
        idxs_on, idxs_off = SpaceMath.split_active_group(scores)
        w_on = scores * idxs_on                       # (bs, n)  [0, 1]
        w_off = -(scores * idxs_off)                  # (bs, n)  [0, 1]
        mixed = ((w_on.sum(1) > 0) & (w_off.sum(1) > 0)).to(scores.dtype)   # (bs,)

        w_on = w_on / w_on.sum(1, keepdim=True).clamp(min=SpaceMath.eps)
        w_off = w_off / w_off.sum(1, keepdim=True).clamp(min=SpaceMath.eps)

        batch_positions = positions.expand(bs, -1, -1)                       # (bs, n, D)
        center_on = torch.bmm(w_on.unsqueeze(1), batch_positions)            # (bs, 1, D)
        batch_centers = center_on.expand(-1, n, -1)
        d = SpaceMath.distance(batch_positions, batch_centers, dim=2,
                               distance_type=config.modular_space_distance, pairwise=False)   # (bs, n)

        dist_on = (w_on * d).sum(1)                   # (bs,)
        dist_off = (w_off * d).sum(1)                 # (bs,)

        m = mixed.sum().clamp(min=1.0)
        return (dist_on * mixed).sum() / m, (dist_off * mixed).sum() / m

    @staticmethod
    def _combine(config, loss_on, loss_off):
        loss_on = torch.log(1 + loss_on) * float(config.modular_space_loss_on)
        loss_off = torch.log(1 + loss_off) * float(config.modular_space_loss_off)
        return loss_on, loss_off

    # ------------------------------------------------------------------
    def _process_scores(self, config, scores: torch.Tensor, positions: torch.Tensor,
                        stats: torch.Tensor, clusters_labels: torch.Tensor | None):
        idxs_on, idxs_off = SpaceMath.split_active_group(scores)

        # --- stats block: unchanged ---------------------------------------
        if stats is not None:
            scores_on = scores * idxs_on
            scores_off = -(scores * idxs_off)
            stats[:, 0] += idxs_on.sum(dim=0).detach()
            stats[:, 1] += idxs_off.sum(dim=0).detach()
            stats[:, 2] += scores_on.sum(dim=0).detach()
            stats[:, 3] += scores_off.sum(dim=0).detach()

        # --- 1) whole layer: your original loss, without loss_centers -------
        l_on, l_off = self._contrast(config, scores, positions)
        l_on, l_off = self._combine(config, l_on, l_off)
        loss_on, loss_off = l_on, l_off
        loss = l_on - l_off

        # --- 2) same thing, once per cluster --------------------------------
        loss_cluster = scores.new_zeros(())
        if clusters_labels is not None:
            labels = torch.as_tensor(clusters_labels, device=scores.device).long()
            ks = [int(k) for k in torch.unique(labels) if k >= 0]            # -1 = noise
            terms = []
            for k in ks:
                idx = (labels == k).nonzero(as_tuple=True)[0]
                if idx.numel() < 2:
                    continue
                c_on, c_off = self._contrast(config, scores[:, idx], positions[idx])
                c_on, c_off = self._combine(config, c_on, c_off)
                terms.append(c_on - c_off)
            if terms:
                loss_cluster = torch.stack(terms).mean()
                loss = loss + float(config.modular_space_loss_cluster) * loss_cluster

        # --- 3) sigreg on the positions -------------------------------------
        loss_sigreg = scores.new_zeros(())
        w_s = float(config.modular_space_loss_sigreg)
        if w_s > 0:
            X = positions
            if str(config.modular_space_distance).lower().startswith("cos"):
                # unit-norm rows have cov = I/D; rescale so the sigreg target (I) is right
                X = F.normalize(positions, dim=-1) * math.sqrt(positions.shape[1])
            loss_sigreg = sigreg(X, method=str(config.modular_space_sigreg_method))
            loss = loss + w_s * loss_sigreg

        return loss, loss_on, loss_off, loss_cluster, loss_sigreg
