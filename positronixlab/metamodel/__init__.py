"""Metamodel package with lazy exports to avoid eager ``parallelazy`` coupling."""

from __future__ import annotations

from typing import Any

_LAZY_EXPORTS = {
    "MetaModelConfig": ("positronixlab.metamodel.config", "MetaModelConfig"),
    "MetaCriterion": ("positronixlab.metamodel.metacriterion", "MetaCriterion"),
    "MetaMetric": ("positronixlab.metamodel.metacriterion", "MetaMetric"),
    "BaseModel": ("positronixlab.metamodel.metamodel", "BaseModel"),
    "MetaModel": ("positronixlab.metamodel.metamodel", "MetaModel"),
    "ResNet": ("positronixlab.metamodel.metamodel", "ResNet"),
    "MetaTransform": ("positronixlab.metamodel.metatransform", "MetaTransform"),
    "ModelLoader": ("positronixlab.metamodel.sota", "ModelLoader"),
    "SOTAModel": ("positronixlab.metamodel.sota", "SOTAModel"),
    "OddOneOutHead": ("positronixlab.metamodel.sota", "OddOneOutHead"),
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

