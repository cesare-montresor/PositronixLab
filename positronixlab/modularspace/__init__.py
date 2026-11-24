from .config import LayerSpaceConfig
from .custertracker import ClusterEvent, ClusterIdentityTracker, ClusterState, ClusterTracker
from .layerspace import LayerSpace
from .presets import PRESETS, get_preset, get_preset_spec, list_presets
from .space_step import SpaceStep, SpaceStepSpec

_LAZY_EXPORTS = {
    "ModularOverSpace": ("positronixlab.modularspace.modularoverspace", "ModularOverSpace"),
    "ModularSpace": ("positronixlab.modularspace.modularspace", "ModularSpace"),
}


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        module_path, attr = _LAZY_EXPORTS[name]
        module = __import__(module_path, fromlist=[attr])
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ClusterEvent",
    "ClusterIdentityTracker",
    "ClusterState",
    "ClusterTracker",
    "LayerSpace",
    "LayerSpaceConfig",
    "ModularOverSpace",
    "ModularSpace",
    "PRESETS",
    "get_preset",
    "get_preset_spec",
    "list_presets",
    "SpaceStep",
    "SpaceStepSpec",
]
