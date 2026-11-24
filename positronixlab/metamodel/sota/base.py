"""Base abstractions and shared heads for unified SOTA modularity models.

This module intentionally avoids importing ``positronixlab.pipeline`` so SOTA
models and loaders work in standalone PyTorch environments without requiring
``parallelazy``.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

# Layer types supported by both Atlas observation and NeuroSurgeon / weight masking
MASKABLE_LAYER_TYPES: Tuple[Type[nn.Module], ...] = (nn.Linear, nn.Conv2d, nn.Conv1d)


def _get_maskable_types() -> Tuple[Type[nn.Module], ...]:
    """Return tuple of maskable PyTorch layer classes (including HF Conv1D if available)."""
    try:
        from transformers.pytorch_utils import Conv1D  # type: ignore

        return MASKABLE_LAYER_TYPES + (Conv1D,)
    except Exception:
        return MASKABLE_LAYER_TYPES


def get_maskable_layer_names(
    model: nn.Module,
    include_types: Optional[Tuple[Type[nn.Module], ...]] = None,
    exclude_patterns: Iterable[str] = (),
) -> List[str]:
    """Return canonical dotted names of maskable layers in *model*."""
    types = include_types if include_types is not None else _get_maskable_types()
    patterns = tuple(exclude_patterns)
    names: List[str] = []
    for name, module in model.named_modules():
        if not name:
            continue
        if isinstance(module, types):
            if any(pat in name for pat in patterns):
                continue
            names.append(name)
    return names


@dataclass(frozen=True)
class SOTAModelSpec:
    """Metadata describing a registered SOTA model architecture."""

    arch: str
    source: str  # "csordas" | "lepori" | "neurosurgeon" | "huggingface"
    task_type: str  # "classification" | "odd_one_out" | "seq2seq" | "sequence"
    builder: Callable[..., "SOTAModel"]
    description: str = ""
    default_exclude_patterns: Tuple[str, ...] = field(default_factory=tuple)


class OddOneOutHead(nn.Module):
    """Parameter-free cosine-similarity 4-way Odd-One-Out head (Lepori et al., 2023).

    Given embeddings of shape ``(B, N, D)`` (typically ``N = 4``), computes
    L2-normalized embeddings and returns ``-sum_j cos(x_i, x_j)`` as logits of
    shape ``(B, N)`` so the outlier receives the highest logit.
    """

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 3:
            raise ValueError(
                f"OddOneOutHead expects 3D tensor (B, N, D), got shape {tuple(embeddings.shape)}"
            )
        normed = F.normalize(embeddings, dim=-1)
        # Pairwise cosine similarity matrix: (B, N, N), summed over peers -> (B, N)
        sim = torch.bmm(normed, normed.transpose(1, 2)).sum(dim=-1)
        return -sim


class SOTAModel(nn.Module):
    """Unified base class for SOTA architectures loaded through :class:`ModelLoader`.

    Implements the ``BaseModel`` / ``DeeplearningTrain`` contract (``name``,
    ``device``, ``loss(x, y, y_hat)``, ``save()``, ``load()``) plus modularity
    inspection helpers (``maskable_layers()``, ``to_neurosurgeon()``).
    """

    source: str = "custom"
    arch: str = "custom"
    task_type: str = "classification"
    default_exclude_patterns: Tuple[str, ...] = ()

    def __init__(
        self,
        name: Optional[str] = None,
        device: Optional[torch.device | str] = None,
        criterion: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.name = name or self.__class__.__name__
        self.timestamp = time.time()
        self.basepath: Optional[str] = None
        self.criterion: nn.Module = criterion if criterion is not None else nn.CrossEntropyLoss()
        self._device = torch.device(device) if device is not None else torch.device("cpu")

    @property
    def device(self) -> torch.device:
        for p in self.parameters():
            return p.device
        for b in self.buffers():
            return b.device
        return self._device

    @device.setter
    def device(self, value: torch.device | str) -> None:
        self._device = torch.device(value)

    def to(self, *args: Any, **kwargs: Any) -> "SOTAModel":
        result = super().to(*args, **kwargs)
        for p in result.parameters():
            result._device = p.device
            break
        return result

    def loss(self, x: Any, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        """Compute task loss compatible with ``DeeplearningTrain``."""
        return self.criterion(y_hat, y)

    def maskable_layers(
        self,
        exclude_patterns: Optional[Iterable[str]] = None,
    ) -> List[str]:
        """Return canonical dotted names of maskable/observable layers."""
        patterns = (
            self.default_exclude_patterns
            if exclude_patterns is None
            else tuple(exclude_patterns)
        )
        return get_maskable_layer_names(self, exclude_patterns=patterns)

    def to_neurosurgeon(
        self,
        target_layers: Optional[List[str]] = None,
        mask_method: str = "continuous_sparsification",
        mask_unit: str = "weight",
        mask_hparams: Optional[Dict[str, Any]] = None,
        freeze_base: bool = True,
        l0_lambda: float = 1e-8,
        deepcopy_model: bool = True,
    ) -> Any:
        """Wrap this model in a :class:`NeuroSurgeonWrappedModel`."""
        from positronixlab.metamodel.sota.neurosurgeon_adapter import wrap_with_neurosurgeon

        layers = target_layers if target_layers is not None else self.maskable_layers()
        return wrap_with_neurosurgeon(
            self,
            target_layers=layers,
            mask_method=mask_method,
            mask_unit=mask_unit,
            mask_hparams=mask_hparams,
            freeze_base=freeze_base,
            l0_lambda=l0_lambda,
            deepcopy_model=deepcopy_model,
        )

    def model_path(self, path: Optional[str | Path] = None) -> str:
        if path is None and self.basepath is None:
            resolved = Path("./checkpoint") / f"{self.name}.pth"
        elif path is None:
            resolved = Path(self.basepath) / f"{self.name}.pth"
        else:
            resolved = Path(path)
        return str(resolved.resolve())

    def save(self, path: Optional[str | Path] = None) -> str:
        resolved = self.model_path(path)
        os.makedirs(os.path.dirname(resolved), exist_ok=True)
        torch.save(self.state_dict(), resolved)
        return resolved

    def load(
        self,
        path: Optional[str | Path] = None,
        strict: bool = True,
        map_location: Optional[torch.device | str] = None,
    ) -> Optional[str]:
        resolved = self.model_path(path)
        if not os.path.exists(resolved):
            return None
        loc = map_location if map_location is not None else self.device
        state = torch.load(resolved, map_location=loc, weights_only=True)
        self.load_state_dict(state, strict=strict)
        return resolved
