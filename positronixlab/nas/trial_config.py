from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Type, TypeVar

from positronixlab.pipeline.pipeline_config import PipelineConfig

TTrialConfig = TypeVar("TTrialConfig", bound="TrialConfig")


@dataclass
class TrialConfig(PipelineConfig):
    """
    Ax-aware hyperparameter configuration extending `PipelineConfig`.

    Concrete experiments in `AtlasCompose` (or library tests) compose this with
    domain config mixins (e.g. `MetaModelConfig`, `LayerSpaceConfig`,
    `TrainConfig`, `DataConfig`) for full IDE autocomplete while staying flat
    for Ax parameter mapping.
    """

    experiment_name: Optional[str] = None
    trial_index: Optional[int] = None
    trial_dir: Optional[str] = None
    gpu_id: Optional[int] = None

    @classmethod
    def from_trial(
        cls: Type[TTrialConfig],
        params: Mapping[str, Any],
        fixed: Optional[Mapping[str, Any]] = None,
        *,
        experiment_name: Optional[str] = None,
        trial_index: Optional[int] = None,
        trial_dir: Optional[str] = None,
        gpu_id: Optional[int] = None,
        name: Optional[str] = None,
    ) -> TTrialConfig:
        """
        Build a typed `TrialConfig` subclass by merging template fixed constants
        with Ax-generated parameters (Ax params take precedence).
        """
        merged: Dict[str, Any] = {}
        if fixed:
            merged.update(fixed)
        if params:
            merged.update(params)

        if experiment_name is not None:
            merged["experiment_name"] = experiment_name
        if trial_index is not None:
            merged["trial_index"] = trial_index
        if trial_dir is not None:
            merged["trial_dir"] = trial_dir
        if gpu_id is not None:
            merged["gpu_id"] = gpu_id
            merged.setdefault("device", f"cuda:{gpu_id}")

        if name is not None:
            merged["name"] = name
        elif trial_index is not None:
            merged["name"] = f"trial_{trial_index:03d}"

        return cls.from_dict(merged)
