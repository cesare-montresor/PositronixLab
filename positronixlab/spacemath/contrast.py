from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor

from positronixlab.spacemath.common import torch_compile
from positronixlab.spacemath.space_type import (
    SpaceType,
    distance_pairwise,
    space_is_cosine,
    space_normalize,
)


class ContrastResult(NamedTuple):
    """Unified output of any contrastive step over (K, U) scopes."""

    loss: Tensor          # scalar total contrastive loss
    loss_on: Tensor       # scalar active attraction term (or total CE)
    loss_off: Tensor      # scalar inactive repulsion term (0 for CE)
    centers_on: Tensor    # (bs, K, D) weighted centers of active units per scope
    centers_off: Tensor   # (bs, K, D) weighted centers of inactive units per scope
    mixed: Tensor         # (bs, K) 1.0 where scope has both active & inactive mass


def _scoped_weights_and_centers(
    positions: Tensor,
    w_on: Tensor,
    w_off: Tensor,
    scope_mask: Tensor,
    eps: float = 1e-8,
):
    """Normalize active/inactive weights within each scope (K, U) and compute weighted centers (bs, K, D)."""
    M = scope_mask.unsqueeze(0)  # (1, K, U)
    w_on_k = w_on.unsqueeze(1) * M
    w_off_k = w_off.unsqueeze(1) * M

    mass_on = w_on_k.sum(dim=-1)
    mass_off = w_off_k.sum(dim=-1)
    mixed = ((mass_on > eps) & (mass_off > eps)).to(positions.dtype)

    w_on_norm = w_on_k / mass_on.clamp_min(eps).unsqueeze(-1)
    w_off_norm = w_off_k / mass_off.clamp_min(eps).unsqueeze(-1)

    centers_on = torch.matmul(w_on_norm, positions)    # (bs, K, D)
    centers_off = torch.matmul(w_off_norm, positions)  # (bs, K, D)
    return w_on_norm, w_off_norm, centers_on, centers_off, mixed


@torch_compile
def _distance_contrast_core(
    positions: Tensor,
    w_on: Tensor,
    w_off: Tensor,
    scope_mask: Tensor,
    use_cosine: bool,
    use_log: bool,
    weight_on: float,
    weight_off: float,
    eps: float,
) -> ContrastResult:
    """Compiled static-shape core for distance-based on/off contrast."""
    pos = F.normalize(positions, dim=-1, eps=eps) if use_cosine else positions
    w_on_norm, w_off_norm, centers_on, centers_off, mixed = _scoped_weights_and_centers(
        pos, w_on, w_off, scope_mask, eps=eps
    )
    space = SpaceType.COSINE if use_cosine else SpaceType.EUCLIDEAN
    d = distance_pairwise(centers_on, pos, space_type=space, eps=eps)  # (bs, K, U)

    dist_on = (w_on_norm * d).sum(dim=-1)
    dist_off = (w_off_norm * d).sum(dim=-1)

    m_count = mixed.sum().clamp_min(1.0)
    mean_on = (dist_on * mixed).sum() / m_count
    mean_off = (dist_off * mixed).sum() / m_count

    if use_log:
        loss_on = torch.log(1.0 + mean_on) * weight_on
        loss_off = torch.log(1.0 + mean_off) * weight_off
    else:
        loss_on = mean_on * weight_on
        loss_off = mean_off * weight_off

    loss = loss_on - loss_off
    return ContrastResult(loss, loss_on, loss_off, centers_on, centers_off, mixed)


def contrast_distance_log(
    positions: Tensor,
    w_on: Tensor,
    w_off: Tensor,
    scope_mask: Tensor,
    *,
    space_type: str = SpaceType.COSINE,
    weight_on: float = 1.0,
    weight_off: float = 1.0,
    eps: float = 1e-8,
    **kwargs,
) -> ContrastResult:
    """
    Vectorized distance-based on/off contrast with log damping:
      loss = weight_on * log(1 + d_on) - weight_off * log(1 + d_off)
    """
    return _distance_contrast_core(
        positions,
        w_on,
        w_off,
        scope_mask,
        space_is_cosine(space_type),
        True,
        float(weight_on),
        float(weight_off),
        float(eps),
    )


def contrast_distance_linear(
    positions: Tensor,
    w_on: Tensor,
    w_off: Tensor,
    scope_mask: Tensor,
    *,
    space_type: str = SpaceType.COSINE,
    weight_on: float = 1.0,
    weight_off: float = 1.0,
    eps: float = 1e-8,
    **kwargs,
) -> ContrastResult:
    """
    Vectorized distance-based on/off contrast without log damping:
      loss = weight_on * d_on - weight_off * d_off
    """
    return _distance_contrast_core(
        positions,
        w_on,
        w_off,
        scope_mask,
        space_is_cosine(space_type),
        False,
        float(weight_on),
        float(weight_off),
        float(eps),
    )


@torch_compile
def _softmax_ce_contrast_core(
    positions: Tensor,
    w_on: Tensor,
    w_off: Tensor,
    scope_mask: Tensor,
    use_cosine: bool,
    tau: float,
    eps: float,
) -> ContrastResult:
    """Compiled static-shape core for temperature-scaled softmax cross-entropy contrast."""
    space = SpaceType.COSINE if use_cosine else SpaceType.EUCLIDEAN
    P = space_normalize(positions, space_type=space, eps=eps)
    q, w_off_norm, centers_on, centers_off, mixed = _scoped_weights_and_centers(
        P, w_on, w_off, scope_mask, eps=eps
    )

    if use_cosine:
        C = F.normalize(centers_on, p=2, dim=-1, eps=eps)
        logits = torch.matmul(C, P.T) / tau
    else:
        d2 = (
            centers_on.pow(2).sum(dim=-1, keepdim=True)
            - 2.0 * torch.matmul(centers_on, P.T)
            + P.pow(2).sum(dim=-1)
        )
        logits = -d2.clamp_min(0.0) / tau

    neg = -torch.finfo(logits.dtype).max / 2.0
    logits = logits.masked_fill(scope_mask.unsqueeze(0) == 0, neg)
    logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    ce = -(q * logp).sum(dim=-1)  # (bs, K)

    loss = (ce * mixed).sum() / mixed.sum().clamp_min(1.0)
    zero = loss.new_zeros(())
    return ContrastResult(loss, loss, zero, centers_on, centers_off, mixed)


def contrast_softmax_ce(
    positions: Tensor,
    w_on: Tensor,
    w_off: Tensor,
    scope_mask: Tensor,
    *,
    space_type: str = SpaceType.COSINE,
    tau: float = 0.1,
    eps: float = 1e-8,
    **kwargs,
) -> ContrastResult:
    """
    Temperature-scaled softmax cross-entropy contrast per scope:
    pulls active units toward their scope center while pushing inactive units in the scope away.
    """
    return _softmax_ce_contrast_core(
        positions,
        w_on,
        w_off,
        scope_mask,
        space_is_cosine(space_type),
        float(tau),
        float(eps),
    )


CONTRASTS: dict[str, Callable[..., ContrastResult]] = {
    "distance_log": contrast_distance_log,
    "distance_linear": contrast_distance_linear,
    "softmax_ce": contrast_softmax_ce,
}


def get_contrast(name: str) -> Callable[..., ContrastResult]:
    """Look up a contrastive step function by name from `CONTRASTS`."""
    key = str(name).strip().lower()
    if key not in CONTRASTS:
        raise KeyError(f"Unknown contrast '{name}'. Available: {sorted(CONTRASTS)}")
    return CONTRASTS[key]
