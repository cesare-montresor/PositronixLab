"""Shared base abstractions for external baseline models and compositional datasets.

Intentionally independent of ``positronixlab.pipeline`` / ``parallelazy`` so
baseline models, datasets, and benchmarks run cleanly in standalone PyTorch
environments.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from positronixlab.dataprovider.dataprovider import (
    CompositionalDataProvider as CompositionalDataProvider,
)
from positronixlab.dataprovider.dataprovider import (
    DataProvider,
)
from positronixlab.dataprovider.dataprovider import (
    SubtaskSplit as SubtaskSplit,
)
from positronixlab.utils.utils import load_data, save_data

# Layer types supported by both Atlas observation and weight/neuron masking
MASKABLE_LAYER_TYPES: tuple[type[nn.Module], ...] = (nn.Linear, nn.Conv2d, nn.Conv1d)


def _get_maskable_types() -> tuple[type[nn.Module], ...]:
    """Return tuple of maskable PyTorch layer classes (including HF Conv1D if available)."""
    try:
        from transformers.pytorch_utils import Conv1D  # type: ignore

        return MASKABLE_LAYER_TYPES + (Conv1D,)
    except Exception:
        return MASKABLE_LAYER_TYPES


def get_maskable_layer_names(
    model: nn.Module,
    include_types: tuple[type[nn.Module], ...] | None = None,
    exclude_patterns: Iterable[str] = (),
) -> list[str]:
    """Return canonical dotted names of maskable layers in *model*."""
    types = include_types if include_types is not None else _get_maskable_types()
    patterns = tuple(exclude_patterns)
    names: list[str] = []
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
    """Metadata describing a registered baseline model architecture."""

    arch: str
    source: str  # "csordas" | "lepori" | "neurosurgeon" | ...
    task_type: str  # "classification" | "odd_one_out" | "seq2seq" | "sequence"
    builder: Callable[..., SOTAModel]
    description: str = ""
    default_exclude_patterns: tuple[str, ...] = field(default_factory=tuple)


class SOTAModel(nn.Module):
    """Unified base class for baseline architectures loaded through :class:`ModelLoader`.

    Implements the ``BaseModel`` / ``DeeplearningTrain`` contract (``name``,
    ``device``, ``loss(x, y, y_hat)``, ``save()``, ``load()``) plus modularity
    inspection helpers (``maskable_layers()``, ``to_neurosurgeon()``).
    """

    source: str = "custom"
    arch: str = "custom"
    task_type: str = "classification"
    default_exclude_patterns: tuple[str, ...] = ()

    def __init__(
        self,
        name: str | None = None,
        device: torch.device | str | None = None,
        criterion: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.name = name or self.__class__.__name__
        self.timestamp = time.time()
        self.basepath: str | None = None
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

    def to(self, *args: Any, **kwargs: Any) -> SOTAModel:
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
        exclude_patterns: Iterable[str] | None = None,
    ) -> list[str]:
        """Return canonical dotted names of maskable/observable layers."""
        patterns = (
            self.default_exclude_patterns
            if exclude_patterns is None
            else tuple(exclude_patterns)
        )
        return get_maskable_layer_names(self, exclude_patterns=patterns)

    def to_neurosurgeon(
        self,
        target_layers: list[str] | None = None,
        mask_method: str = "continuous_sparsification",
        mask_unit: str = "weight",
        mask_hparams: dict[str, Any] | None = None,
        freeze_base: bool = True,
        l0_lambda: float = 1e-8,
        deepcopy_model: bool = True,
    ) -> Any:
        """Wrap this model in a :class:`NeuroSurgeonWrappedModel`."""
        from positronixlab.baselines.neurosurgeon.adapter import wrap_with_neurosurgeon

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

    def model_path(self, path: str | Path | None = None) -> str:
        if path is None and self.basepath is None:
            resolved = Path("./checkpoint") / f"{self.name}.pth"
        elif path is None and self.basepath is not None:
            resolved = Path(self.basepath) / f"{self.name}.pth"
        else:
            assert path is not None
            resolved = Path(path)
        return str(resolved.resolve())

    def save(self, path: str | Path | None = None, **kwargs: Any) -> str:
        resolved = self.model_path(path)
        save_data(resolved, self.state_dict(), **kwargs)
        return resolved

    def load(
        self,
        path: str | Path | None = None,
        strict: bool = True,
        map_location: torch.device | str | None = None,
        **kwargs: Any,
    ) -> str | None:
        resolved = self.model_path(path)
        loc = map_location if map_location is not None else self.device
        state = load_data(resolved, map_location=loc, default=None, **kwargs)
        if state is None:
            return None
        self.load_state_dict(state, strict=strict)
        return resolved


@dataclass(frozen=True)
class SOTADatasetSpec:
    """Registry metadata for a baseline benchmark dataset."""

    name: str
    source: str  # "csordas" | "lepori" | ...
    task_type: str  # "classification" | "odd_one_out" | "arithmetic"
    builder: Callable[..., DataProvider]
    description: str = ""

