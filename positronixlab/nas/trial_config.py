from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeVar

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

    experiment_name: str | None = None
    trial_index: int | None = None
    trial_dir: str | None = None
    gpu_id: int | None = None

    @classmethod
    def from_trial(
        cls: type[TTrialConfig],
        params: Mapping[str, Any],
        fixed: Mapping[str, Any] | None = None,
        *,
        experiment_name: str | None = None,
        trial_index: int | None = None,
        trial_dir: str | None = None,
        gpu_id: int | None = None,
        name: str | None = None,
    ) -> TTrialConfig:
        """
        Build a typed `TrialConfig` subclass by merging template fixed constants
        with Ax-generated parameters (Ax params take precedence).
        """
        merged: dict[str, Any] = {}
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
            if not merged.get("device"):
                merged["device"] = f"cuda:{gpu_id}"

        if name is not None:
            merged["name"] = name
        elif trial_index is not None:
            merged["name"] = f"trial_{trial_index:03d}"

        return cls.from_dict(merged)
