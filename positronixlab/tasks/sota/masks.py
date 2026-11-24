"""Canonical ``SubnetworkMask`` representation and leak-free ``MaskApplier``.

Unifies unit-level masks (PositronixLab Atlas, NeuroSurgeon ``mask_unit="neuron"``)
and weight-level masks (Csordas et al., Lepori et al., NeuroSurgeon ``mask_unit="weight"``)
under a single contract keyed by canonical dotted PyTorch module names.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Optional, Set, Tuple

import torch
import torch.nn as nn

Granularity = Literal["unit", "weight"]
AblationMode = Literal[
    "retain",
    "zero_ablate",
    "random_ablate",
    "randomly_sampled",
    "complement_sampled",
]


def _broadcast_unit_mask_to_weight(
    unit_mask: torch.Tensor,
    weight: torch.Tensor,
    module: nn.Module,
) -> torch.Tensor:
    """Broadcast a 1D unit mask ``(out_units,)`` to match ``weight.shape``."""
    m = unit_mask.bool()
    if m.shape == weight.shape:
        return m
    if m.ndim != 1:
        raise ValueError(
            f"Expected 1D unit mask for module {type(module).__name__}, "
            f"got shape {tuple(m.shape)} for weight {tuple(weight.shape)}."
        )
    # GPT-2 Conv1D stores weight as (in_features, out_features)
    if type(module).__name__ == "Conv1D" and weight.ndim == 2 and m.shape[0] == weight.shape[1]:
        return m.unsqueeze(0).expand_as(weight).clone()

    if m.shape[0] != weight.shape[0]:
        raise ValueError(
            f"Unit mask length {m.shape[0]} does not match output dimension "
            f"{weight.shape[0]} of weight {tuple(weight.shape)}."
        )
    view_shape = (m.shape[0],) + (1,) * (weight.ndim - 1)
    return m.reshape(view_shape).expand_as(weight).clone()


@dataclass(frozen=True)
class SubnetworkMask:
    """Canonical representation of a discovered functional subnetwork."""

    name: str
    granularity: Granularity
    masks: Dict[str, torch.Tensor]
    scores: Optional[Dict[str, torch.Tensor]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.granularity not in ("unit", "weight"):
            raise ValueError(f"granularity must be 'unit' or 'weight', got {self.granularity!r}")
        normalized = {k: v.detach().bool() for k, v in self.masks.items()}
        object.__setattr__(self, "masks", normalized)
        if self.scores is not None:
            norm_scores = {k: v.detach().clone() for k, v in self.scores.items()}
            object.__setattr__(self, "scores", norm_scores)

    # ------------------------------------------------------------------
    # Granularity Projection
    # ------------------------------------------------------------------
    def to_weight_mask(self, model: nn.Module) -> "SubnetworkMask":
        """Return a weight-level ``SubnetworkMask`` matching ``model`` parameter shapes."""
        if self.granularity == "weight":
            return self
        modules = dict(model.named_modules())
        weight_masks: Dict[str, torch.Tensor] = {}
        for layer_name, unit_m in self.masks.items():
            if layer_name not in modules:
                raise KeyError(
                    f"Layer {layer_name!r} from SubnetworkMask({self.name!r}) not found in model."
                )
            mod = modules[layer_name]
            weight = getattr(mod, "weight", None)
            if weight is None:
                continue
            weight_masks[layer_name] = _broadcast_unit_mask_to_weight(
                unit_m.to(weight.device), weight, mod
            )
        return SubnetworkMask(
            name=self.name,
            granularity="weight",
            masks=weight_masks,
            scores=self.scores,
            metadata={**self.metadata, "projected_from": "unit"},
        )

    def to_unit_mask(self, min_active_fraction: float = 0.0) -> "SubnetworkMask":
        """Project a weight-level mask to 1D unit-level ``(out_units,)``."""
        if self.granularity == "unit":
            return self
        unit_masks: Dict[str, torch.Tensor] = {}
        for layer_name, w_mask in self.masks.items():
            if w_mask.ndim == 1:
                unit_masks[layer_name] = w_mask.bool()
            else:
                flat = w_mask.reshape(w_mask.shape[0], -1).float()
                if min_active_fraction <= 0.0:
                    unit_masks[layer_name] = flat.any(dim=1)
                else:
                    unit_masks[layer_name] = flat.mean(dim=1) >= min_active_fraction
        return SubnetworkMask(
            name=self.name,
            granularity="unit",
            masks=unit_masks,
            scores=self.scores,
            metadata={**self.metadata, "projected_from": "weight"},
        )

    # ------------------------------------------------------------------
    # Set Algebra
    # ------------------------------------------------------------------
    def _check_compatible(self, other: "SubnetworkMask") -> Set[str]:
        if self.granularity != other.granularity:
            raise ValueError(
                f"Cannot combine masks of different granularities "
                f"({self.granularity!r} vs {other.granularity!r}). "
                f"Project one with .to_weight_mask(model) or .to_unit_mask() first."
            )
        common = set(self.masks.keys()) & set(other.masks.keys())
        if not common:
            raise ValueError("SubnetworkMasks share no common layer keys.")
        return common

    def __and__(self, other: "SubnetworkMask") -> "SubnetworkMask":
        common = self._check_compatible(other)
        out = {k: self.masks[k] & other.masks[k].to(self.masks[k].device) for k in sorted(common)}
        return SubnetworkMask(name=f"({self.name}&{other.name})", granularity=self.granularity, masks=out)

    def __or__(self, other: "SubnetworkMask") -> "SubnetworkMask":
        common = self._check_compatible(other)
        out = {k: self.masks[k] | other.masks[k].to(self.masks[k].device) for k in sorted(common)}
        return SubnetworkMask(name=f"({self.name}|{other.name})", granularity=self.granularity, masks=out)

    def __sub__(self, other: "SubnetworkMask") -> "SubnetworkMask":
        common = self._check_compatible(other)
        out = {k: self.masks[k] & ~other.masks[k].to(self.masks[k].device) for k in sorted(common)}
        return SubnetworkMask(name=f"({self.name}-{other.name})", granularity=self.granularity, masks=out)

    def invert(self) -> "SubnetworkMask":
        out = {k: ~v for k, v in self.masks.items()}
        return SubnetworkMask(
            name=f"~{self.name}",
            granularity=self.granularity,
            masks=out,
            metadata={**self.metadata, "inverted": True},
        )

    # ------------------------------------------------------------------
    # Structural Metrics (Csordas & Lepori)
    # ------------------------------------------------------------------
    def l0(self) -> int:
        return int(sum(int(m.sum().item()) for m in self.masks.values()))

    def max_l0(self) -> int:
        return int(sum(int(m.numel()) for m in self.masks.values()))

    def sparsity(self) -> float:
        total = self.max_l0()
        if total == 0:
            return 0.0
        return 1.0 - (self.l0() / total)

    def density(self) -> float:
        total = self.max_l0()
        if total == 0:
            return 0.0
        return self.l0() / total

    def iou(self, other: "SubnetworkMask") -> float:
        common = self._check_compatible(other)
        inter = 0
        union = 0
        for k in common:
            a = self.masks[k]
            b = other.masks[k].to(a.device)
            inter += int((a & b).sum().item())
            union += int((a | b).sum().item())
        return (inter / union) if union > 0 else 0.0

    def per_layer_iou(self, other: "SubnetworkMask") -> Dict[str, float]:
        common = self._check_compatible(other)
        result: Dict[str, float] = {}
        for k in sorted(common):
            a = self.masks[k]
            b = other.masks[k].to(a.device)
            inter = int((a & b).sum().item())
            union = int((a | b).sum().item())
            result[k] = (inter / union) if union > 0 else 0.0
        return result

    def iomin(self, other: "SubnetworkMask") -> float:
        """Return Intersection-over-Minimum (Csordas et al. subsetness metric): ``|A & B| / min(|A|, |B|)``."""
        common = self._check_compatible(other)
        inter = 0
        count_a = 0
        count_b = 0
        for k in common:
            a = self.masks[k]
            b = other.masks[k].to(a.device)
            inter += int((a & b).sum().item())
            count_a += int(a.sum().item())
            count_b += int(b.sum().item())
        denom = min(count_a, count_b)
        return (inter / denom) if denom > 0 else 0.0

    def per_layer_iomin(self, other: "SubnetworkMask") -> Dict[str, float]:
        """Return per-layer Intersection-over-Minimum (subsetness)."""
        common = self._check_compatible(other)
        result: Dict[str, float] = {}
        for k in sorted(common):
            a = self.masks[k]
            b = other.masks[k].to(a.device)
            inter = int((a & b).sum().item())
            denom = min(int(a.sum().item()), int(b.sum().item()))
            result[k] = (inter / denom) if denom > 0 else 0.0
        return result

    def overlap_fraction(self, other: "SubnetworkMask") -> float:
        """Return ``|self & other| / |self|`` (Csordas mask sharing / remaining fraction)."""
        common = self._check_compatible(other)
        inter = 0
        self_count = 0
        for k in common:
            a = self.masks[k]
            b = other.masks[k].to(a.device)
            inter += int((a & b).sum().item())
            self_count += int(a.sum().item())
        return (inter / self_count) if self_count > 0 else 0.0

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> str:
        resolved = Path(path).resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "granularity": self.granularity,
            "masks": {k: v.cpu() for k, v in self.masks.items()},
            "scores": (
                {k: v.cpu() for k, v in self.scores.items()}
                if self.scores is not None
                else None
            ),
            "metadata": self.metadata,
        }
        torch.save(payload, resolved)
        return str(resolved)

    @classmethod
    def load(cls, path: str | Path) -> "SubnetworkMask":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return cls(
            name=payload["name"],
            granularity=payload["granularity"],
            masks=payload["masks"],
            scores=payload.get("scores"),
            metadata=payload.get("metadata", {}),
        )


def _sample_random_binary_mask(
    shape: Tuple[int, ...],
    n_true: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
    eligible_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sample a random bool tensor of *shape* with exactly *n_true* True entries."""
    out = torch.zeros(shape, dtype=torch.bool, device=device)
    if n_true <= 0:
        return out
    flat_out = out.reshape(-1)
    if eligible_mask is None:
        n_total = flat_out.numel()
        k = min(n_true, n_total)
        idx = torch.randperm(n_total, generator=generator, device=device)[:k]
        flat_out[idx] = True
    else:
        eligible_idx = torch.nonzero(eligible_mask.reshape(-1), as_tuple=False).squeeze(-1)
        if eligible_idx.numel() < n_true:
            raise ValueError(
                f"Cannot sample {n_true} entries from complement with only "
                f"{eligible_idx.numel()} eligible entries."
            )
        perm = torch.randperm(eligible_idx.numel(), generator=generator, device=device)[:n_true]
        flat_out[eligible_idx[perm]] = True
    return out


class MaskApplier:
    """Leak-free context manager applying a :class:`SubnetworkMask` to any ``nn.Module``.

    Supports all 5 NeuroSurgeon / Lepori / Csordas ablation modes:
    - ``"retain"``: keep only weights inside ``mask`` (zero everything outside).
    - ``"zero_ablate"``: zero weights inside ``mask``, keep complement.
    - ``"random_ablate"``: replace weights inside ``mask`` with Kaiming-uniform random values.
    - ``"randomly_sampled"``: zero a random subset of weights with the same cardinality as ``mask``.
    - ``"complement_sampled"``: zero a random subset of weights of the same cardinality drawn
      strictly from ``~mask``.

    Guarantees exact restoration of original weights even if an exception is raised.
    """

    def __init__(
        self,
        model: nn.Module,
        mask: SubnetworkMask,
        mode: AblationMode = "retain",
        exclude_layers: Iterable[str] = (),
        seed: Optional[int] = None,
    ) -> None:
        if mode not in (
            "retain",
            "zero_ablate",
            "random_ablate",
            "randomly_sampled",
            "complement_sampled",
        ):
            raise ValueError(f"Unsupported ablation mode: {mode!r}")
        self.model = model
        self.mask = mask
        self.mode = mode
        self.exclude_layers = set(exclude_layers)
        self.seed = seed
        self._backups: Dict[str, torch.Tensor] = {}

    def __enter__(self) -> nn.Module:
        weight_mask = self.mask.to_weight_mask(self.model)
        modules = dict(self.model.named_modules())

        with torch.no_grad():
            for layer_name, m_bool in weight_mask.masks.items():
                if layer_name in self.exclude_layers or layer_name not in modules:
                    continue
                mod = modules[layer_name]
                weight = getattr(mod, "weight", None)
                if weight is None:
                    continue

                self._backups[layer_name] = weight.data.detach().clone()
                m = m_bool.to(weight.device)
                gen = None
                if self.seed is not None:
                    gen = torch.Generator(device=weight.device)
                    gen.manual_seed(self.seed + (hash(layer_name) & 0xFFFF))

                if self.mode == "retain":
                    weight.data.mul_(m.to(weight.dtype))
                elif self.mode == "zero_ablate":
                    weight.data.mul_((~m).to(weight.dtype))
                elif self.mode == "random_ablate":
                    rand_w = torch.empty_like(weight.data)
                    if rand_w.ndim >= 2:
                        nn.init.kaiming_uniform_(rand_w, a=math.sqrt(5), generator=gen)
                    else:
                        rand_w.normal_(0.0, 0.02, generator=gen)
                    weight.data.copy_(torch.where(m, rand_w, weight.data))
                elif self.mode == "randomly_sampled":
                    n_active = int(m.sum().item())
                    rand_ablate = _sample_random_binary_mask(
                        tuple(weight.shape), n_active, weight.device, generator=gen
                    )
                    weight.data.mul_((~rand_ablate).to(weight.dtype))
                elif self.mode == "complement_sampled":
                    n_active = int(m.sum().item())
                    comp_ablate = _sample_random_binary_mask(
                        tuple(weight.shape),
                        n_active,
                        weight.device,
                        generator=gen,
                        eligible_mask=~m,
                    )
                    weight.data.mul_((~comp_ablate).to(weight.dtype))

        return self.model

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        modules = dict(self.model.named_modules())
        with torch.no_grad():
            for layer_name, orig_w in self._backups.items():
                mod = modules.get(layer_name)
                if mod is not None and hasattr(mod, "weight") and mod.weight is not None:
                    mod.weight.data.copy_(orig_w)
        self._backups.clear()
