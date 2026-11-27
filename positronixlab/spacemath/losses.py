from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import Tensor

from positronixlab.spacemath.sigreg import sigreg
from positronixlab.spacemath.space_type import (
    SpaceType,
    distance_elementwise,
    distance_pairwise,
    space_sigreg_prepare,
)


def loss_centers_repulsion(
    centers_on: Tensor,
    centers_off: Tensor,
    mixed: Tensor | None = None,
    *,
    space_type: str = SpaceType.COSINE,
    eps: float = 1e-8,
) -> Tensor:
    """
    Break symmetry between active and inactive centers on the same sample/scope.
    Returns negative mean distance so minimizing it pushes center_on and center_off apart.
    """
    dist = distance_elementwise(centers_on, centers_off, space_type=space_type, dim=-1, eps=eps)
    if mixed is not None:
        return -((dist * mixed).sum() / mixed.sum().clamp_min(1.0))
    return -dist.mean()


def loss_batch_repulsion(
    centers_on: Tensor,
    mixed: Tensor | None = None,
    *,
    space_type: str = SpaceType.COSINE,
    eps: float = 1e-8,
) -> Tensor:
    """
    Encourage different samples in the batch to occupy distinct regions.
    Returns negative mean pairwise distance across active batch centers.
    """
    if centers_on.ndim == 3:
        if mixed is not None:
            pts = centers_on[mixed > 0]
        else:
            pts = centers_on.reshape(-1, centers_on.shape[-1])
    else:
        pts = centers_on[mixed > 0] if mixed is not None else centers_on

    if pts.shape[0] <= 1:
        return centers_on.new_zeros(())
    dist = distance_pairwise(pts, pts, space_type=space_type, eps=eps)
    return -dist.mean()


def loss_sigreg(
    x: Tensor,
    *,
    space_type: str = SpaceType.COSINE,
    method: str = "weak",
    sketch_dim: int = 64,
    eps: float = 1e-8,
) -> Tensor:
    """
    Isotropic Gaussian regularization (SIGReg) on (M, D) vectors.
    Automatically rescales unit-norm vectors by sqrt(D) for cosine geometry.
    """
    if x.shape[0] <= 1:
        return x.new_zeros(())
    x_prep = space_sigreg_prepare(x, space_type=space_type, eps=eps)
    return sigreg(x_prep, sketch_dim=int(sketch_dim), method=str(method))


def loss_sigreg_centers(
    centers: Tensor,
    mixed: Tensor,
    *,
    space_type: str = SpaceType.COSINE,
    method: str = "weak",
    sketch_dim: int = 64,
    eps: float = 1e-8,
) -> Tensor:
    """
    Dynamic-shape SIGReg applied only to active (mixed > 0) scope centers.
    Keep outside compiled regions to avoid dynamic-shape graph breaks.
    """
    active_mask = mixed > 0
    if active_mask.sum() <= 1:
        return centers.new_zeros(())
    active_centers = centers[active_mask]
    return loss_sigreg(
        active_centers,
        space_type=space_type,
        method=method,
        sketch_dim=sketch_dim,
        eps=eps,
    )


def loss_layer_alignment(
    centers_per_layer: Sequence[Tensor] | Tensor,
    *,
    space_type: str = SpaceType.COSINE,
    eps: float = 1e-8,
) -> Tensor:
    """
    Cross-layer alignment loss: pulls per-layer active centers toward the
    mean active center across layers for each sample.
    """
    if isinstance(centers_per_layer, Tensor):
        stacked = centers_per_layer  # (L, bs, D)
    else:
        if len(centers_per_layer) == 0:
            return torch.tensor(0.0)
        if len(centers_per_layer) < 2:
            ref = centers_per_layer[0]
            return ref.new_zeros(())
        stacked = torch.stack(list(centers_per_layer), dim=0)

    mean_center = stacked.mean(dim=0, keepdim=True).expand_as(stacked)
    dist = distance_elementwise(stacked, mean_center, space_type=space_type, dim=-1, eps=eps)
    return dist.mean()


LOSSES: dict[str, Callable[..., Tensor]] = {
    "centers": loss_centers_repulsion,
    "batch": loss_batch_repulsion,
    "sigreg": loss_sigreg,
    "sigreg_centers": loss_sigreg_centers,
    "alignment": loss_layer_alignment,
}
