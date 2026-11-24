from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple
import torch
from torch import Tensor
import torch.nn.functional as F


def split_active_group(scores: Tensor, threshold: float = 0.0) -> Tuple[Tensor, Tensor]:
    """Split unit scores (bs, U) into boolean (idxs_on, idxs_off) masks."""
    idxs_on = scores >= threshold
    idxs_off = ~idxs_on
    return idxs_on, idxs_off


@torch.no_grad()
def stats_update_(stats: Tensor, scores: Tensor, threshold: float = 0.0) -> Tensor:
    """
    In-place update of running per-unit stats (U, 4):
      [:, 0] count_on
      [:, 1] count_off
      [:, 2] sum of positive scores
      [:, 3] sum of negative score magnitudes
    """
    idxs_on, idxs_off = split_active_group(scores, threshold=threshold)
    stats[:, 0] += idxs_on.sum(dim=0)
    stats[:, 1] += idxs_off.sum(dim=0)
    stats[:, 2] += (scores * idxs_on).clamp_min(0.0).sum(dim=0)
    stats[:, 3] += (-(scores * idxs_off)).clamp_min(0.0).sum(dim=0)
    return stats


def stats_activation_frequency(stats: Tensor) -> Tensor:
    """Per-unit activation frequency in [0, 1] from (U, 4) stats."""
    total = (stats[:, 0] + stats[:, 1]).clamp_min(1.0)
    return (stats[:, 0] / total).detach()


def mask_global_scope(
    num_units: int,
    device: torch.device,
    dtype: torch.dtype,
    unit_mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Build the global (1, U) scope mask.
    If `unit_mask` (U,) is provided (e.g. active cluster slots in OverSpace),
    inactive slots are masked to 0.
    """
    if unit_mask is not None:
        return unit_mask.to(device=device, dtype=dtype).reshape(1, num_units)
    return torch.ones(1, num_units, device=device, dtype=dtype)


def mask_cluster_membership(
    labels: Optional[Tensor],
    num_units: Optional[int] = None,
    *,
    device: torch.device,
    dtype: torch.dtype,
    unit_mask: Optional[Tensor] = None,
) -> Optional[Tensor]:
    """
    Build a (K, U) one-hot cluster membership mask from integer labels (U,).
    Negative labels (e.g. -1 noise) are excluded. Returns None if K < 2.
    """
    if labels is None:
        return None
    lab = torch.as_tensor(labels, device=device).long()
    if num_units is not None and lab.numel() != num_units:
        raise ValueError(f"Expected {num_units} cluster labels, got {lab.numel()}.")
    if lab.numel() == 0:
        return None
    K = int(lab.max().item()) + 1
    if K < 2:
        return None
    M = F.one_hot(lab.clamp_min(0), K).T.to(dtype)  # (K, U)
    valid = (lab >= 0).to(dtype).unsqueeze(0)
    if unit_mask is not None:
        valid = valid * unit_mask.to(device=device, dtype=dtype).unsqueeze(0)
    return M * valid


def weight_raw(
    scores: Tensor,
    *,
    threshold: float = 0.0,
    **kwargs,
) -> Tuple[Tensor, Tensor]:
    """Use raw score magnitudes for active/inactive weights (bs, U)."""
    if threshold == 0.0:
        w_on = scores.clamp_min(0.0)
        w_off = (-scores).clamp_min(0.0)
    else:
        idxs_on, idxs_off = split_active_group(scores, threshold=threshold)
        w_on = ((scores - threshold) * idxs_on).clamp_min(0.0)
        w_off = ((threshold - scores) * idxs_off).clamp_min(0.0)
    return w_on, w_off


def weight_inv_freq(
    scores: Tensor,
    *,
    stats: Optional[Tensor] = None,
    freq: Optional[Tensor] = None,
    freq_alpha: float = 0.5,
    threshold: float = 0.0,
    min_freq: float = 1e-3,
    **kwargs,
) -> Tuple[Tensor, Tensor]:
    """IDF-like inverse-frequency weighting on active units so hubs don't dominate centers."""
    w_on, w_off = weight_raw(scores, threshold=threshold)
    if freq is None and stats is not None:
        freq = stats_activation_frequency(stats)
    if freq is not None and freq_alpha > 0.0:
        g = freq.clamp_min(min_freq).pow(-freq_alpha)
        w_on = w_on * (g / g.mean().clamp_min(1e-8))
    return w_on, w_off


def weight_uniform(
    scores: Tensor,
    *,
    threshold: float = 0.0,
    **kwargs,
) -> Tuple[Tensor, Tensor]:
    """Binary uniform weights (1.0 for active/inactive units, ignoring magnitude)."""
    idxs_on, idxs_off = split_active_group(scores, threshold=threshold)
    return idxs_on.to(scores.dtype), idxs_off.to(scores.dtype)


WEIGHTERS: Dict[str, Callable[..., Tuple[Tensor, Tensor]]] = {
    "raw": weight_raw,
    "inv_freq": weight_inv_freq,
    "uniform": weight_uniform,
}


def get_weighter(name: str) -> Callable[..., Tuple[Tensor, Tensor]]:
    """Look up a weighting function by name from `WEIGHTERS`."""
    key = str(name).strip().lower()
    if key not in WEIGHTERS:
        raise KeyError(f"Unknown weighter '{name}'. Available: {sorted(WEIGHTERS)}")
    return WEIGHTERS[key]
