"""NAS/HPO package with lazy exports so ``TrialConfig``/``TrialPipeline`` do not eagerly import Ax/BoTorch."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from positronixlab.nas.experiment import Experiment
    from positronixlab.nas.trial_config import TrialConfig
    from positronixlab.nas.trial_pipeline import (
        MetricValue,
        TrialPipeline,
        TrialResult,
        is_oom_error,
    )

_LAZY_EXPORTS = {
    "Experiment": ("positronixlab.nas.experiment", "Experiment"),
    "MetricValue": ("positronixlab.nas.trial_pipeline", "MetricValue"),
    "TrialConfig": ("positronixlab.nas.trial_config", "TrialConfig"),
    "TrialPipeline": ("positronixlab.nas.trial_pipeline", "TrialPipeline"),
    "TrialResult": ("positronixlab.nas.trial_pipeline", "TrialResult"),
    "is_oom_error": ("positronixlab.nas.trial_pipeline", "is_oom_error"),
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
