from __future__ import annotations

from typing import Callable, Dict, Optional
import torch
from torch import Tensor, nn

from positronixlab.spacemath.reducers import DEFAULT_REDUCER_BY_LAYER, get_reducer


def score_normalize_by_threshold(values: Tensor, threshold: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Split `values` around `threshold` and scale:
      values >= threshold  ->  [0, 1]
      values <  threshold  -> [-1, 0)
    """
    idxs_on = values >= threshold
    idxs_off = ~idxs_on

    scores_on = (values - threshold + eps) * idxs_on
    scores_off = ((values - threshold) - eps) * idxs_off

    scores_max = scores_on.amax(dim=1, keepdim=True).abs() + eps
    scores_min = scores_off.amin(dim=1, keepdim=True).abs() + eps
    return (scores_on / scores_max) + (scores_off / scores_min)


def score_quantile(
    values: Tensor,
    active_q: float = 0.8,
    eps: float = 1e-8,
    **kwargs,
) -> Tensor:
    """Per-sample quantile threshold normalization to [-1, 1]."""
    q = torch.quantile(values, active_q, interpolation="higher", dim=1, keepdim=True).detach()
    return score_normalize_by_threshold(values, q, eps=eps)


def score_top_k(
    values: Tensor,
    active_q: float = 0.8,
    eps: float = 1e-8,
    **kwargs,
) -> Tensor:
    """
    Fast per-sample top-k threshold normalization to [-1, 1].
    Selects roughly the top `(1 - active_q)` fraction of units as active.
    """
    num_units = values.shape[1]
    k = max(1, min(num_units - 1, int(round((1.0 - active_q) * num_units))))
    topk_vals = torch.topk(values, k=k, dim=1, largest=True, sorted=True).values
    threshold = topk_vals[:, -1:].detach()
    return score_normalize_by_threshold(values, threshold, eps=eps)


def score_sign_preserving_row(
    values: Tensor,
    eps: float = 1e-8,
    **kwargs,
) -> Tensor:
    """
    Per-sample sign-preserving normalization around 0.0:
      positive values / max_row(pos)  in [0, 1]
      negative values / max_row(-neg) in [-1, 0)
    """
    pos = values.clamp_min(0.0)
    neg = (-values).clamp_min(0.0)
    pos_scale = pos.amax(dim=1, keepdim=True) + eps
    neg_scale = neg.amax(dim=1, keepdim=True) + eps
    return (pos / pos_scale) - (neg / neg_scale)


def score_sign_preserving_batch(
    values: Tensor,
    eps: float = 1e-8,
    **kwargs,
) -> Tensor:
    """
    Batch-level sign-preserving normalization around 0.0:
      positive values / max_batch(pos)  in [0, 1]
      negative values / max_batch(-neg) in [-1, 0)
    """
    pos = values.clamp_min(0.0)
    neg = (-values).clamp_min(0.0)
    pos_scale = pos.amax() + eps
    neg_scale = neg.amax() + eps
    return (pos / pos_scale) - (neg / neg_scale)


def score_softmax(values: Tensor, **kwargs) -> Tensor:
    """Argmax unit(s) = +1.0, all others = -1.0."""
    q = values.amax(dim=1, keepdim=True).detach()
    return torch.where(values >= q, torch.ones_like(values), -torch.ones_like(values))


SCORERS: Dict[str, Callable[..., Tensor]] = {
    "quantile": score_quantile,
    "top_k": score_top_k,
    "sign_preserving_row": score_sign_preserving_row,
    "sign_preserving_batch": score_sign_preserving_batch,
    "softmax": score_softmax,
}


def get_scorer(name: str) -> Callable[..., Tensor]:
    """Look up a scoring function by name from `SCORERS`."""
    key = str(name).strip().lower()
    if key not in SCORERS:
        raise KeyError(f"Unknown scorer '{name}'. Available: {sorted(SCORERS)}")
    return SCORERS[key]


def activations_to_scores(
    activations: Tensor,
    *,
    layer_type: str = "linear",
    score_method: str = "quantile",
    reducer: Optional[str] = None,
    activation_fn: Optional[nn.Module] = None,
    apply_activation: bool = False,
    active_q: float = 0.8,
    cnn_locality_weight: float = 0.5,
    spatial_radius: int = 3,
    eps: float = 1e-8,
) -> Tensor:
    """Convert raw layer activations into standardized [-1, 1] unit scores (bs, U)."""
    x = activations
    if apply_activation and activation_fn is not None:
        x = activation_fn(x)

    reducer_name = reducer or DEFAULT_REDUCER_BY_LAYER.get(str(layer_type).lower(), "linear")
    reducer_fn = get_reducer(reducer_name)
    values = reducer_fn(
        x,
        locality_weight=cnn_locality_weight,
        spatial_radius=spatial_radius,
        eps=eps,
    )

    effective_method = "softmax" if isinstance(activation_fn, nn.Softmax) else score_method
    scorer_fn = get_scorer(effective_method)
    return scorer_fn(values, active_q=active_q, eps=eps)
