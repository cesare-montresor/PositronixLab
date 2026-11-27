from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


class SpaceType:
    """Supported embedding space geometries (shared by space optimization and clustering)."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    ALL = (COSINE, EUCLIDEAN)

    @classmethod
    def normalize_name(cls, space_type: str) -> str:
        """Canonicalize a space-type string to 'cosine' or 'euclidean'."""
        key = str(space_type).strip().lower()
        if key.startswith("cos"):
            return cls.COSINE
        if key.startswith("euc") or key in ("l2", "euclid"):
            return cls.EUCLIDEAN
        raise ValueError(f"Unknown space_type '{space_type}'. Expected one of {cls.ALL}.")

    @classmethod
    def is_cosine(cls, space_type: str) -> bool:
        """Return True if `space_type` resolves to cosine geometry."""
        return cls.normalize_name(space_type) == cls.COSINE


def space_is_cosine(space_type: str) -> bool:
    """Return True if space_type selects cosine geometry."""
    return SpaceType.is_cosine(space_type)


def space_normalize(positions: Tensor, space_type: str = SpaceType.COSINE, eps: float = 1e-8) -> Tensor:
    """Project positions onto the space geometry (unit sphere for cosine, identity for euclidean)."""
    if space_is_cosine(space_type):
        return F.normalize(positions, p=2, dim=-1, eps=eps)
    return positions


def space_sigreg_prepare(x: Tensor, space_type: str = SpaceType.COSINE, eps: float = 1e-8) -> Tensor:
    """
    Prepare vectors (M, D) for isotropic Gaussian SIGReg target (cov = I).
    Unit-norm rows on the sphere have cov = I/D, so rescale by sqrt(D).
    """
    if space_is_cosine(space_type):
        D = x.shape[-1]
        return F.normalize(x, p=2, dim=-1, eps=eps) * math.sqrt(D)
    return x


def distance_elementwise(
    x: Tensor,
    y: Tensor,
    space_type: str = SpaceType.COSINE,
    dim: int = -1,
    eps: float = 1e-8,
) -> Tensor:
    """Elementwise distance between matching vectors in x and y along `dim`."""
    if space_is_cosine(space_type):
        x_norm = F.normalize(x, p=2, dim=dim, eps=eps)
        y_norm = F.normalize(y, p=2, dim=dim, eps=eps)
        dist = 1.0 - (x_norm * y_norm).sum(dim=dim)
    else:
        sq = (x - y).pow(2).sum(dim=dim)
        sq_safe = torch.where(sq > 0, sq, torch.ones_like(sq))
        dist = torch.where(sq > 0, sq_safe.sqrt(), torch.zeros_like(sq))
    return dist.clamp(min=0.0)


def distance_pairwise(
    x: Tensor,
    y: Tensor,
    space_type: str = SpaceType.COSINE,
    eps: float = 1e-8,
) -> Tensor:
    """
    Pairwise distance between x and y.
    Supports:
      - 2D: x (M, D), y (U, D)         -> (M, U)
      - 3D: x (bs, K, D), y (U, D)     -> (bs, K, U)
      - 3D: x (bs, K, D), y (bs, U, D) -> (bs, K, U)
    """
    if space_is_cosine(space_type):
        x_norm = F.normalize(x, p=2, dim=-1, eps=eps)
        y_norm = F.normalize(y, p=2, dim=-1, eps=eps)
        if y_norm.ndim == 2:
            sim = torch.matmul(x_norm, y_norm.T)
        else:
            sim = torch.matmul(x_norm, y_norm.transpose(-1, -2))
        dist = 1.0 - sim
    else:
        if x.ndim == 3 and y.ndim == 2:
            y_in = y.unsqueeze(0).expand(x.shape[0], -1, -1)
        else:
            y_in = y
        dist = torch.cdist(x, y_in, p=2)
    return dist.clamp(min=0.0)
