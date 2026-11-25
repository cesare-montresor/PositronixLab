"""Model observation and computation graph tracing."""

from __future__ import annotations

from typing import Any

from positronixlab.modularobserver.layerobserver import LayerObserver
from positronixlab.modularobserver.modelmapper import (
    GraphFormatter,
    ModelEdge,
    ModelGraph,
    ModelMapper,
    ModelNode,
    extract_layer_io_metrics,
)

_LAZY_EXPORTS = {
    "ModularObserver": ("positronixlab.modularobserver.modularobserver", "ModularObserver"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_path, attr = _LAZY_EXPORTS[name]
        module = __import__(module_path, fromlist=[attr])
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "GraphFormatter",
    "LayerObserver",
    "ModelEdge",
    "ModelGraph",
    "ModelMapper",
    "ModelNode",
    "ModularObserver",
    "extract_layer_io_metrics",
]
