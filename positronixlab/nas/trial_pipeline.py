from __future__ import annotations

from abc import abstractmethod
from dataclasses import asdict, dataclass, field
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from positronixlab.nas.trial_config import TrialConfig
from positronixlab.pipeline.pipeline import Pipeline
from positronixlab.pipeline.pipeline_config import PipelineOptions

logger = logging.getLogger(__name__)

MetricValue = Union[float, Tuple[float, float]]

_OOM_MARKERS = (
    "out of memory",
    "cudaerrormemoryallocation",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
    "outofmemoryerror",
)


def is_oom_error(message: str) -> bool:
    """Detect CUDA/GPU out-of-memory errors from exception text or plugin logs."""
    low = message.lower()
    return any(marker in low for marker in _OOM_MARKERS)


@dataclass
class TrialResult:
    """
    Standardized outcome of executing a single NAS/HPO trial.

    - `status`: `"completed"`, `"oom"`, `"timeout"`, or `"error"`.
    - `metrics`: Raw Ax metrics (`float` or `(mean, sem)`). Never pre-negated.
    - `metadata`: Observed bookkeeping (e.g. `epochs_trained`, `best_epoch`, `peak_vram_gb`).
    - `effective_overrides`: Optional post-trial config overrides (e.g. `{"train_epochs": 44}`)
      written to `effective_config.yaml` without mutating the immutable Ax input `config.yaml`.
    """

    status: str = "completed"
    metrics: Dict[str, MetricValue] = field(default_factory=dict)
    duration_sec: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    effective_overrides: Dict[str, Any] = field(default_factory=dict)
    errors: Dict[str, List[str]] = field(default_factory=dict)
    reason: Optional[str] = None

    @property
    def is_completed(self) -> bool:
        return self.status == "completed"

    @property
    def is_oom(self) -> bool:
        return self.status == "oom"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True, default=str)


class TrialPipeline(Pipeline):
    """
    Standardized `parallelazy`-backed pipeline for Ax NAS/HPO trials.

    Owns trial-specific concerns on top of generic `Pipeline`:
    - Binds `ManagerOptions.work_dir` to `config.trial_dir` (so `parallelazy`
      places logs under `<trial_dir>/logs/<run_id>/` and artifacts under `<trial_dir>/`).
    - Extracts Ax metrics (`METRICS_ENV_KEY = "metrics"` by default), metadata,
      and optional post-trial `effective_overrides`.
    - Returns a standardized `TrialResult` classifying completion, OOM, timeout,
      and plugin errors.
    """

    METRICS_ENV_KEY: str = "metrics"

    def __init__(
        self,
        config: TrialConfig,
        options: Optional[PipelineOptions] = None,
    ):
        resolved_options = options or PipelineOptions()
        if config.trial_dir and resolved_options.work_dir is None:
            resolved_options = resolved_options.with_updates(work_dir=config.trial_dir)
        super().__init__(config=config, options=resolved_options)
        self.config: TrialConfig = config

    @abstractmethod
    def setup(self) -> None:
        """Load plugins and initialize shared environment before `run()`."""
        pass

    def extract_metrics(self, env: Dict[str, Any]) -> Dict[str, MetricValue]:
        """
        Extract Ax objective/constraint/tracking metrics from the shared plugin
        environment. By default reads `env[self.METRICS_ENV_KEY]`; override for
        custom metric layouts.
        """
        raw = env.get(self.METRICS_ENV_KEY, {})
        if not isinstance(raw, dict):
            return {}
        out: Dict[str, MetricValue] = {}
        for k, v in raw.items():
            if isinstance(v, (tuple, list)) and len(v) == 2:
                out[str(k)] = (float(v[0]), float(v[1]))
            elif isinstance(v, (int, float)):
                out[str(k)] = float(v)
        return out

    def extract_metadata(self, env: Dict[str, Any]) -> Dict[str, Any]:
        """Optional hook to record non-Ax trial metadata in `result.json`."""
        return {}

    def extract_effective_overrides(self, env: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optional hook returning post-trial config overrides (e.g. actual
        `train_epochs` after early stopping) saved to `effective_config.yaml`
        without mutating the immutable input `config.yaml`.
        """
        return {}

    def execute(
        self,
        timeout_sec: Optional[float] = None,
        poll_interval: Optional[float] = None,
        parallel_plugins: Optional[bool] = None,
        terminate_on_error: Optional[bool] = None,
    ) -> TrialResult:
        """
        Run the full trial lifecycle: `setup()` -> `run()` -> `wait()` ->
        extract metrics -> always `shutdown()`.
        """
        t0 = time.monotonic()
        try:
            self.setup()
            self.run(parallel=parallel_plugins)
            self.wait(
                timeout_sec=timeout_sec,
                poll_interval=poll_interval,
                terminate_on_error=terminate_on_error,
            )

            env = self.get_env() or {}
            return TrialResult(
                status="completed",
                metrics=self.extract_metrics(env),
                duration_sec=time.monotonic() - t0,
                metadata=self.extract_metadata(env),
                effective_overrides=self.extract_effective_overrides(env),
            )

        except TimeoutError as exc:
            return TrialResult(
                status="timeout",
                duration_sec=time.monotonic() - t0,
                errors=self.get_errors(),
                reason=str(exc),
            )
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            errors = self.get_errors() if not self._closed else {}
            if not errors:
                errors = {"pipeline": [msg]}
            combined = "\n".join([msg, *[m for msgs in errors.values() for m in msgs]])
            return TrialResult(
                status="oom" if is_oom_error(combined) else "error",
                duration_sec=time.monotonic() - t0,
                errors=errors,
                reason=msg,
            )
        finally:
            self.shutdown()
