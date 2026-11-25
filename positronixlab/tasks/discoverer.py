"""Shared subnetwork discovery interface and control discoverers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from positronixlab.baselines.base import get_maskable_layer_names
from positronixlab.tasks.masks import SubnetworkMask


def unpack_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unpack ``(x, y)`` or ``(x, y, idx)`` batches uniformly."""
    if isinstance(batch, (tuple, list)):
        return batch[0], batch[1]
    raise TypeError(f"Expected (x, y) or (x, y, idx) batch, got {type(batch).__name__}")


class SubnetworkDiscoverer(ABC):
    """Base interface for all functional modularity / subnetwork discovery methods."""

    name: str = "base"

    @abstractmethod
    def discover(
        self,
        model: nn.Module,
        dataloader: Iterable[Any],
        target_layers: Optional[List[str]] = None,
        mask_name: str = "subnetwork",
        **kwargs: Any,
    ) -> SubnetworkMask:
        """Discover a subnetwork mask on *model* using *dataloader*."""


class RandomMaskDiscoverer(SubnetworkDiscoverer):
    """Control baseline producing a random subnetwork mask of target density."""

    name = "random"

    def __init__(
        self,
        density: float = 0.5,
        granularity: str = "weight",
        seed: int = 0,
    ) -> None:
        if not (0.0 <= density <= 1.0):
            raise ValueError(f"density must be in [0, 1], got {density}")
        self.density = density
        self.granularity = granularity
        self.seed = seed

    def discover(
        self,
        model: nn.Module,
        dataloader: Iterable[Any] = (),
        target_layers: Optional[List[str]] = None,
        mask_name: str = "random",
        **kwargs: Any,
    ) -> SubnetworkMask:
        layers = (
            list(target_layers)
            if target_layers is not None
            else get_maskable_layer_names(model)
        )
        modules = dict(model.named_modules())
        gen = torch.Generator()
        gen.manual_seed(self.seed)

        masks: Dict[str, torch.Tensor] = {}
        for layer_name in layers:
            mod = modules.get(layer_name)
            if mod is None or not hasattr(mod, "weight") or mod.weight is None:
                continue
            shape = (mod.weight.shape[0],) if self.granularity == "unit" else tuple(mod.weight.shape)
            masks[layer_name] = torch.rand(shape, generator=gen) < self.density

        return SubnetworkMask(
            name=mask_name,
            granularity=self.granularity,  # type: ignore[arg-type]
            masks=masks,
            metadata={"method": self.name, "density": self.density, "seed": self.seed},
        )
