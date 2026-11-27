"""Model observation and computation graph tracing."""

from __future__ import annotations

from typing import Any

from positronixlab.modularobserver.layerobserver import LayerObserver
from positronixlab.modularobserver.modelmapper import (
    TRACE_ALWAYS,
    GraphFormatter,
    ModelEdge,
    ModelGraph,
    ModelMapper,
    ModelNode,
    extract_layer_io_metrics,
)

_LAZY_EXPORTS = {
    "ActivationCache": ("positronixlab.modularobserver.activation_cache", "ActivationCache"),
    "ModularObserver": ("positronixlab.modularobserver.modularobserver", "ModularObserver"),
    "infer_layer_info": ("positronixlab.modularobserver.activation_cache", "infer_layer_info"),
    "reduce_activation_2d": ("positronixlab.modularobserver.activation_cache", "reduce_activation_2d"),
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
    "ActivationCache",
    "TRACE_ALWAYS",
    "GraphFormatter",
    "LayerObserver",
    "ModelEdge",
    "ModelGraph",
    "ModelMapper",
    "ModelNode",
    "ModularObserver",
    "extract_layer_io_metrics",
    "infer_layer_info",
    "reduce_activation_2d",
]
