from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor


def reduce_linear(activations: Tensor, **kwargs) -> Tensor:
    """Identity reducer for linear/MLP layers: (bs, U) -> (bs, U); (bs, T, U) -> mean over T."""
    if activations.ndim == 2:
        return activations
    if activations.ndim == 3:
        return activations.mean(dim=1)
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
    local max-pool concentration per sample. Already-reduced 2D (bs, C) passes through.
    """
    if activations.ndim == 2:
        return activations
    if activations.ndim == 3:
        return torch.norm(activations, dim=2)
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
    if activations.ndim == 2:
        return activations
    B, C = activations.shape[:2]
    return torch.norm(activations.reshape(B, C, -1), dim=2)


def reduce_conv_mean_abs(activations: Tensor, **kwargs) -> Tensor:
    """Conv feature maps (bs, C, ...) -> (bs, C) per-channel mean absolute value."""
    if activations.ndim == 2:
        return activations
    B, C = activations.shape[:2]
    return activations.reshape(B, C, -1).abs().mean(dim=2)


def reduce_conv_max(activations: Tensor, **kwargs) -> Tensor:
    """Conv feature maps (bs, C, ...) -> (bs, C) per-channel spatial max."""
    if activations.ndim == 2:
        return activations
    B, C = activations.shape[:2]
    return activations.reshape(B, C, -1).amax(dim=2)


def reduce_seq_mean(activations: Tensor, **kwargs) -> Tensor:
    """Sequence activations (bs, T, U) -> (bs, U) via mean over sequence dim."""
    if activations.ndim == 2:
        return activations
    if activations.ndim == 3:
        return activations.mean(dim=1)
    return activations.reshape(activations.shape[0], -1)


def reduce_seq_max(activations: Tensor, **kwargs) -> Tensor:
    """Sequence activations (bs, T, U) -> (bs, U) via max over sequence dim."""
    if activations.ndim == 2:
        return activations
    if activations.ndim == 3:
        return activations.amax(dim=1)
    return activations.reshape(activations.shape[0], -1)


def reduce_seq_cls(activations: Tensor, **kwargs) -> Tensor:
    """Sequence activations (bs, T, U) -> (bs, U) taking token 0 ([CLS])."""
    if activations.ndim == 2:
        return activations
    if activations.ndim == 3:
        return activations[:, 0, :]
    return activations.reshape(activations.shape[0], -1)


def reduce_seq_energy(activations: Tensor, **kwargs) -> Tensor:
    """Sequence activations (bs, T, U) -> (bs, U) via L2 norm over sequence dim."""
    if activations.ndim == 2:
        return activations
    if activations.ndim == 3:
        return torch.norm(activations, dim=1)
    return activations.reshape(activations.shape[0], -1)


REDUCERS: dict[str, Callable[..., Tensor]] = {
    "linear": reduce_linear,
    "conv_locality": reduce_conv_locality,
    "conv_l2": reduce_conv_l2,
    "conv_mean_abs": reduce_conv_mean_abs,
    "conv_max": reduce_conv_max,
    "seq_mean": reduce_seq_mean,
    "seq_max": reduce_seq_max,
    "seq_cls": reduce_seq_cls,
    "seq_energy": reduce_seq_energy,
}

DEFAULT_REDUCER_BY_LAYER: dict[str, str] = {
    "linear": "linear",
    "conv": "conv_locality",
    "conv2d": "conv_locality",
    "conv1d": "conv_l2",
    "rnn": "seq_mean",
    "lstm": "seq_mean",
    "transformer": "seq_mean",
}


def get_reducer(name: str) -> Callable[..., Tensor]:
    """Look up a reducer function by name from `REDUCERS`."""
    key = str(name).strip().lower()
    if key not in REDUCERS:
        raise KeyError(f"Unknown reducer '{name}'. Available: {sorted(REDUCERS)}")
    return REDUCERS[key]
