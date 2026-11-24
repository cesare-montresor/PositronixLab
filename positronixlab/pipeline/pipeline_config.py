from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Dict, Optional

from parallelazy.plugin_data import ManagerOptions

from positronixlab.pipeline.config import BaseConfig


@dataclass
class PipelineOptions(ManagerOptions):
    """
    Runtime execution options for `Pipeline`, extending `parallelazy.ManagerOptions`
    with pipeline-level polling, error-handling, and debug timeout controls.
    """

    terminate_on_error: bool = True
    timeout_sec: Optional[float] = None
    poll_interval: float = 0.25
    status_log_interval: Optional[float] = None
    verbose: bool = False
    parallel_plugins: bool = True

    def to_manager_kwargs(self) -> Dict[str, Any]:
        """Return only the fields accepted by `parallelazy.PluginManager(...)`."""
        manager_field_names = {f.name for f in fields(ManagerOptions)}
        return {name: getattr(self, name) for name in manager_field_names}

    def with_updates(self, **overrides: Any) -> "PipelineOptions":
        valid = {f.name for f in fields(self)}
        filtered = {k: v for k, v in overrides.items() if k in valid}
        return replace(self, **filtered)


@dataclass
class PipelineConfig(BaseConfig):
    """
    Base hyperparameter configuration for a `parallelazy`-backed `Pipeline`.

    Extend this (or compose with domain config mixins) for concrete pipelines.
    """

    seed: Optional[int] = None
    device: Optional[str] = None
