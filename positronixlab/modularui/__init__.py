"""Live Dash/Plotly/Cytoscape visualization components (lazy exports for optional ``[ui]`` extra)."""

from __future__ import annotations

from typing import Any

_LAZY_EXPORTS = {
    "ConfigOverlay": ("positronixlab.modularui.components", "ConfigOverlay"),
    "Container": ("positronixlab.modularui.components", "Container"),
    "DashboardLayout": ("positronixlab.modularui.docks", "DashboardLayout"),
    "ExpandableSection": ("positronixlab.modularui.components", "ExpandableSection"),
    "GroupedChartsDisplay": ("positronixlab.modularui.plots", "GroupedChartsDisplay"),
    "HomeLayout": ("positronixlab.modularui.layouts", "HomeLayout"),
    "LayerBoardDisplay": ("positronixlab.modularui.plots", "LayerBoardDisplay"),
    "MetricRow": ("positronixlab.modularui.plots", "MetricRow"),
    "ModelDAGDisplay": ("positronixlab.modularui.components", "ModelDAGDisplay"),
    "ModularUI": ("positronixlab.modularui.modularui", "ModularUI"),
    "Panel": ("positronixlab.modularui.docks", "Panel"),
    "SpaceLossRow": ("positronixlab.modularui.plots", "SpaceLossRow"),
    "Title": ("positronixlab.modularui.components", "Title"),
    "UILocks": ("positronixlab.modularui.modularui", "UILocks"),
    "VarsDisplay": ("positronixlab.modularui.components", "VarsDisplay"),
    "Widget": ("positronixlab.modularui.components", "Widget"),
    "loss_then_accuracy_first": ("positronixlab.modularui.plots", "loss_then_accuracy_first"),
}

__all__ = sorted(_LAZY_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_path, attr = _LAZY_EXPORTS[name]
        module = __import__(module_path, fromlist=[attr])
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
