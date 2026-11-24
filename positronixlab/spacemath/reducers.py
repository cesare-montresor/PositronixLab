from __future__ import annotations

from typing import Callable, Dict
import torch
from torch import Tensor
import torch.nn.functional as F


def reduce_linear(activations: Tensor, **kwargs) -> Tensor:
    """Identity reducer for linear/MLP layers: (bs, U) -> (bs, U)."""
    if activations.ndim == 2:
        return activations
    return activations.reshape(activations.shape[0], -1)


def reduce_conv_locality(
    activations: Tensor,
    locality_weight: float = 0.5,
    spatial_radius: int = 3,
    eps: float = 1e-8,
    **kwargs,
) -> Tensor:
    """
    Conv2d feature maps (bs, C, H, W) -> (bs, C) combining L2 magnitude and
    local max-pool concentration per sample.
    """
    B, C = activations.shape[:2]
    magnitude = torch.norm(activations.reshape(B, C, -1), dim=2)

    padding = spatial_radius // 2
    max_pooled = F.max_pool2d(activations, kernel_size=spatial_radius, stride=1, padding=padding)
    denom = activations.abs().sum(dim=(2, 3), keepdim=True) + eps
    locality = (max_pooled.abs() / denom).mean(dim=(2, 3))

    magnitude = magnitude / (magnitude.max(dim=1, keepdim=True)[0] + eps)
    locality = locality / (locality.max(dim=1, keepdim=True)[0] + eps)
    return (1.0 - locality_weight) * magnitude + locality_weight * locality


def reduce_conv_l2(activations: Tensor, **kwargs) -> Tensor:
    """Conv feature maps (bs, C, ...) -> (bs, C) per-channel L2 norm."""
    B, C = activations.shape[:2]
    return torch.norm(activations.reshape(B, C, -1), dim=2)


def reduce_conv_mean_abs(activations: Tensor, **kwargs) -> Tensor:
    """Conv feature maps (bs, C, ...) -> (bs, C) per-channel mean absolute value."""
    B, C = activations.shape[:2]
    return activations.reshape(B, C, -1).abs().mean(dim=2)


def reduce_conv_max(activations: Tensor, **kwargs) -> Tensor:
    """Conv feature maps (bs, C, ...) -> (bs, C) per-channel spatial max."""
    B, C = activations.shape[:2]
    return activations.reshape(B, C, -1).amax(dim=2)


REDUCERS: Dict[str, Callable[..., Tensor]] = {
    "linear": reduce_linear,
    "conv_locality": reduce_conv_locality,
    "conv_l2": reduce_conv_l2,
    "conv_mean_abs": reduce_conv_mean_abs,
    "conv_max": reduce_conv_max,
}

DEFAULT_REDUCER_BY_LAYER: Dict[str, str] = {
    "linear": "linear",
    "conv": "conv_locality",
}


def get_reducer(name: str) -> Callable[..., Tensor]:
    """Look up a reducer function by name from `REDUCERS`."""
    key = str(name).strip().lower()
    if key not in REDUCERS:
        raise KeyError(f"Unknown reducer '{name}'. Available: {sorted(REDUCERS)}")
    return REDUCERS[key]
