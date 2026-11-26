from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import json
import logging
import os
import time
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)
import warnings
import yaml

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=r".*torch\.jit\.script.*deprecated.*", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"^ax(\.|$)")
    from ax.api.client import Client
    from ax.api.configs import StorageConfig
    from ax.core.trial_status import TrialStatus

from positronixlab.nas.trial_config import TrialConfig
from positronixlab.nas.trial_pipeline import MetricValue, TrialPipeline, TrialResult, is_oom_error
from positronixlab.pipeline.pipeline_config import PipelineOptions
from positronixlab.template.template_base import TemplateBase

logger = logging.getLogger(__name__)

TConfig = TypeVar("TConfig", bound=TrialConfig)


def _release_cuda_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _ensure_sqlite_db(url: str) -> None:
    """Bind Ax's SQLAlchemy engine/session factory to `url` and ensure schema tables exist."""
    from ax.storage.sqa_store.db import (
        create_all_tables,
        get_engine,
        init_engine_and_session_factory,
    )

    init_engine_and_session_factory(url=url, force_init=True)
    create_all_tables(get_engine())


class Experiment(Generic[TConfig]):
    """
    Lightweight, SQLite-backed Ax 1.3 experiment runner for NAS and HPO.

    Folder layout (`<output_dir>/<name>/`):
    ```
    <output_dir>/<name>/
    ├── ax.db                 # SQLite source of truth (Ax Client auto-persists)
    ├── template.yaml         # Search space + fixed params + constraints snapshot
    ├── experiment.yaml       # Experiment configuration metadata
    ├── summary.csv           # Human-readable trial table (client.summarize())
    ├── best/                 # Best / Pareto-optimal trial configs + best_summary.json
    └── trials/
        └── trial_000/        # Also parallelazy PluginManager work_dir
            ├── config.yaml   # Immutable Ax input TrialConfig
            ├── effective_config.yaml  # Optional post-trial overrides (e.g. best_epoch)
            ├── result.json   # Metrics, status, duration, metadata, errors
            └── logs/         # parallelazy log_dir (<run_id>/manager.log, ...)
    ```
    """

    def __init__(
        self,
        name: str,
        template: TemplateBase,
        objective: str,
        *,
        trial_pipeline_cls: Optional[Type[TrialPipeline]] = None,
        eval_fn: Optional[Callable[[TConfig], Union[TrialResult, Dict[str, MetricValue]]]] = None,
        config_cls: Type[TConfig] = TrialConfig,  # type: ignore[assignment]
        pipeline_options: Optional[PipelineOptions] = None,
        outcome_constraints: Optional[Sequence[str]] = None,
        tracking_metrics: Optional[Sequence[str]] = None,
        output_dir: str = "output",
        generation_method: str = "fast",
        initialization_budget: Optional[int] = None,
        initialize_with_center: bool = True,
        parallel_init: int = 1,
        parallel_bo: int = 1,
        gpu_ids: Optional[Sequence[int]] = None,
        trial_timeout_sec: Optional[float] = None,
        poll_interval: float = 0.25,
        oom_retries: int = 2,
        oom_backoff_sec: float = 15.0,
        oom_final_status: str = "abandoned",
        reconcile_running: str = "fail",
        random_seed: Optional[int] = None,
    ):
        if trial_pipeline_cls is None and eval_fn is None:
            raise ValueError("Provide either `trial_pipeline_cls` or `eval_fn`.")
        if oom_final_status not in ("abandoned", "failed"):
            raise ValueError("`oom_final_status` must be 'abandoned' or 'failed'.")

        self.name = name
        self.template = template
        self.objective = objective
        self.trial_pipeline_cls = trial_pipeline_cls
        self.eval_fn = eval_fn
        self.config_cls = config_cls
        self.pipeline_options = pipeline_options or PipelineOptions()
        self.outcome_constraints = list(outcome_constraints) if outcome_constraints else None
        self.tracking_metrics = list(tracking_metrics) if tracking_metrics else None

        self.generation_method = generation_method
        self.initialization_budget = initialization_budget
        self.initialize_with_center = initialize_with_center

        self.parallel_init = max(1, int(parallel_init))
        self.parallel_bo = max(1, int(parallel_bo))
        self.gpu_ids = list(gpu_ids) if gpu_ids is not None else None
        self.trial_timeout_sec = trial_timeout_sec
        self.poll_interval = poll_interval

        self.oom_retries = max(0, int(oom_retries))
        self.oom_backoff_sec = float(oom_backoff_sec)
        self.oom_final_status = oom_final_status
        self.reconcile_running = reconcile_running
        self.random_seed = random_seed

        # Directory layout
        self.exp_dir = os.path.abspath(os.path.join(output_dir, self.name))
        self.trials_dir = os.path.join(self.exp_dir, "trials")
        self.best_dir = os.path.join(self.exp_dir, "best")
        self.db_path = os.path.join(self.exp_dir, "ax.db")
        os.makedirs(self.trials_dir, exist_ok=True)
        os.makedirs(self.best_dir, exist_ok=True)

        self._db_url = f"sqlite:///{self.db_path}"
        self.storage_config = StorageConfig(url=self._db_url)
        self.client: Client = self._init_or_resume_client()
        self._save_experiment_metadata()

    @property
    def is_multi_objective(self) -> bool:
        return "," in self.objective

    # -------------------------------------------------------------------------
    # SQLite Lifecycle & Resumption
    # -------------------------------------------------------------------------
    def _init_or_resume_client(self) -> Client:
        existed = os.path.exists(self.db_path) and os.path.getsize(self.db_path) > 0
        _ensure_sqlite_db(self._db_url)

        if existed:
            loaded_client = None
            try:
                loaded_client = Client.load_from_database(
                    experiment_name=self.name,
                    storage_config=self.storage_config,
                )
                logger.info("Resumed Ax experiment '%s' from %s", self.name, self.db_path)
            except Exception as exc:
                logger.warning("Could not load existing DB '%s' (%s); creating new experiment.", self.db_path, exc)
            if loaded_client is not None:
                self._reconcile_interrupted_trials(loaded_client)
                return loaded_client

        client = Client(storage_config=self.storage_config, random_seed=self.random_seed)
        client.configure_experiment(
            name=self.name,
            parameters=self.template.to_parameter_configs(),
            parameter_constraints=self.template.get_constraints() or None,
        )
        client.configure_optimization(
            objective=self.objective,
            outcome_constraints=self.outcome_constraints,
        )
        if self.tracking_metrics:
            client.configure_tracking_metrics(metric_names=self.tracking_metrics)

        init_budget = self.initialization_budget
        if init_budget is None and self.parallel_init > 1:
            init_budget = max(5, self.parallel_init)

        client.configure_generation_strategy(
            method=self.generation_method,
            initialization_budget=init_budget,
            initialize_with_center=self.initialize_with_center,
            allow_exceeding_initialization_budget=(self.parallel_init > 1),
        )
        return client

    def _reconcile_interrupted_trials(self, client: Client) -> None:
        """Handle trials left in RUNNING state if a previous process crashed."""
        running_indices = [
            idx
            for idx, trial in client._experiment.trials.items()
            if trial.status == TrialStatus.RUNNING
        ]
        if not running_indices:
            return
        for idx in running_indices:
            if self.reconcile_running == "abandon":
                client.mark_trial_abandoned(trial_index=idx)
            else:
                client.mark_trial_failed(trial_index=idx, failed_reason="Interrupted prior to completion")
        logger.warning(
            "Reconciled %d leftover RUNNING trials (%s): %s",
            len(running_indices),
            self.reconcile_running,
            running_indices,
        )

    def _save_experiment_metadata(self) -> None:
        self.template.save(os.path.join(self.exp_dir, "template.yaml"))
        meta = {
            "name": self.name,
            "objective": self.objective,
            "is_multi_objective": self.is_multi_objective,
            "outcome_constraints": self.outcome_constraints or [],
            "tracking_metrics": self.tracking_metrics or [],
            "generation_method": self.generation_method,
            "initialization_budget": self.initialization_budget,
            "parallel_init": self.parallel_init,
            "parallel_bo": self.parallel_bo,
            "config_cls": f"{self.config_cls.__module__}.{self.config_cls.__qualname__}",
            "db_path": self.db_path,
        }
        with open(os.path.join(self.exp_dir, "experiment.yaml"), "w", encoding="utf-8") as f:
            yaml.dump(meta, f, sort_keys=False)

    # -------------------------------------------------------------------------
    # Warm-Start from Prior SQLite Experiment (Foundation for Future Variants)
    # -------------------------------------------------------------------------
    def warm_start_from(
        self,
        prior_db_path: str,
        prior_experiment_name: str,
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> int:
        """
        Attach completed trials from a prior SQLite Ax experiment as warm-start
        observations. Returns the number of trials attached.
        """
        prior_url = f"sqlite:///{os.path.abspath(prior_db_path)}"
        _ensure_sqlite_db(prior_url)
        prior_storage = StorageConfig(url=prior_url)
        prior_client = Client.load_from_database(
            experiment_name=prior_experiment_name,
            storage_config=prior_storage,
        )
        prior_exp = prior_client._experiment
        data_df = prior_exp.lookup_data().df

        extracted: List[Tuple[Dict[str, Any], Dict[str, MetricValue]]] = []
        for idx, trial in prior_exp.trials.items():
            if trial.status != TrialStatus.COMPLETED or getattr(trial, "arm", None) is None:
                continue
            params = dict(trial.arm.parameters)
            if filter_fn is not None and not filter_fn(params):
                continue

            trial_rows = data_df[data_df["trial_index"] == idx]
            if trial_rows.empty:
                continue
            raw_data: Dict[str, MetricValue] = {}
            for _, row in trial_rows.iterrows():
                mean = float(row["mean"])
                sem = row.get("sem")
                if sem is not None and sem == sem:  # not NaN
                    raw_data[str(row["metric_name"])] = (mean, float(sem))
                else:
                    raw_data[str(row["metric_name"])] = mean
            extracted.append((params, raw_data))

        # Re-bind Ax SQLAlchemy session factory back to this experiment's DB
        _ensure_sqlite_db(self._db_url)

        attached = 0
        for params, raw_data in extracted:
            new_idx = self.client.attach_trial(parameters=params)
            self.client.complete_trial(trial_index=new_idx, raw_data=raw_data)
            attached += 1

        logger.info(
            "Warm-started '%s' with %d completed trials from '%s'",
            self.name,
            attached,
            prior_experiment_name,
        )
        return attached

    # -------------------------------------------------------------------------
    # Trial Execution Helpers
    # -------------------------------------------------------------------------
    def _trial_dir(self, trial_index: int) -> str:
        tdir = os.path.join(self.trials_dir, f"trial_{trial_index:03d}")
        os.makedirs(tdir, exist_ok=True)
        return tdir

    def _build_trial_config(
        self,
        trial_index: int,
        params: Mapping[str, Any],
        gpu_id: Optional[int] = None,
    ) -> TConfig:
        tdir = self._trial_dir(trial_index)
        config = self.config_cls.from_trial(
            params=params,
            fixed=self.template.get_fixed(),
            experiment_name=self.name,
            trial_index=trial_index,
            trial_dir=tdir,
            gpu_id=gpu_id,
        )
        config.save(os.path.join(tdir, "config.yaml"))
        return config

    def _run_single_trial(self, config: TConfig) -> TrialResult:
        if self.trial_pipeline_cls is not None:
            pipeline = self.trial_pipeline_cls(config=config, options=self.pipeline_options)
            return pipeline.execute(
                timeout_sec=self.trial_timeout_sec,
                poll_interval=self.poll_interval,
            )

        assert self.eval_fn is not None
        t0 = time.monotonic()
        try:
            out = self.eval_fn(config)
            if isinstance(out, TrialResult):
                return out
            return TrialResult(
                status="completed",
                metrics=dict(out),
                duration_sec=time.monotonic() - t0,
            )
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            return TrialResult(
                status="oom" if is_oom_error(msg) else "error",
                duration_sec=time.monotonic() - t0,
                errors={"eval_fn": [msg]},
                reason=msg,
            )

    def _record_trial_result(self, trial_index: int, config: TConfig, result: TrialResult) -> None:
        tdir = self._trial_dir(trial_index)
        result.save(os.path.join(tdir, "result.json"))

        if result.effective_overrides:
            eff_cfg = config.with_updates(**result.effective_overrides)
            eff_cfg.save(os.path.join(tdir, "effective_config.yaml"))

        if result.is_completed:
            self.client.complete_trial(trial_index=trial_index, raw_data=result.metrics)
            logger.info("Trial #%d completed (%.1fs): %s", trial_index, result.duration_sec, result.metrics)
        elif result.is_oom:
            if self.oom_final_status == "failed":
                self.client.mark_trial_failed(trial_index=trial_index, failed_reason=result.reason or "CUDA OOM")
            else:
                self.client.mark_trial_abandoned(trial_index=trial_index)
            logger.warning("Trial #%d marked %s after OOM: %s", trial_index, self.oom_final_status, result.reason)
        elif result.status == "timeout":
            self.client.mark_trial_failed(trial_index=trial_index, failed_reason=result.reason or "Trial timeout")
            logger.warning("Trial #%d timed out: %s", trial_index, result.reason)
        else:
            self.client.mark_trial_failed(trial_index=trial_index, failed_reason=result.reason or "Trial error")
            logger.warning("Trial #%d failed: %s", trial_index, result.reason)

    def _current_concurrency(self, remaining: int) -> int:
        completed = sum(
            1 for t in self.client._experiment.trials.values() if t.status == TrialStatus.COMPLETED
        )
        init_budget = self.initialization_budget if self.initialization_budget is not None else 5
        in_init = completed < init_budget
        workers = self.parallel_init if in_init else self.parallel_bo
        if self.gpu_ids is not None and len(self.gpu_ids) > 0:
            workers = min(workers, len(self.gpu_ids)) if workers > 1 else workers
        return max(1, min(workers, remaining))

    def _retry_oom_solo(self, trial_index: int, params: Mapping[str, Any]) -> None:
        """
        Re-run a trial that hit OOM solo (concurrency=1) with backoff retries
        to handle both parallel-batch contention and transient shared-GPU load.
        """
        gpu_id = self.gpu_ids[0] if self.gpu_ids else None
        config = self._build_trial_config(trial_index, params, gpu_id=gpu_id)

        last_result: Optional[TrialResult] = None
        total_attempts = 1 + self.oom_retries
        for attempt in range(1, total_attempts + 1):
            _release_cuda_cache()
            logger.info(
                "Trial #%d solo OOM retry %d/%d...",
                trial_index,
                attempt,
                total_attempts,
            )
            result = self._run_single_trial(config)
            last_result = result
            if not result.is_oom:
                self._record_trial_result(trial_index, config, result)
                return
            if attempt < total_attempts and self.oom_backoff_sec > 0:
                time.sleep(self.oom_backoff_sec)

        assert last_result is not None
        self._record_trial_result(trial_index, config, last_result)

    # -------------------------------------------------------------------------
    # Main Optimization Loop
    # -------------------------------------------------------------------------
    def run(
        self,
        n_trials: int,
        fixed_parameters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[TConfig, Dict[str, Tuple[float, Optional[float]]], int]]:
        """
        Run (or resume) until `n_trials` trials have reached a terminal state
        (`COMPLETED` / `ABANDONED` / `EARLY_STOPPED`), then export summary and
        best/Pareto configs.
        """
        while True:
            terminal_count = sum(
                1
                for t in self.client._experiment.trials.values()
                if t.status in (TrialStatus.COMPLETED, TrialStatus.ABANDONED, TrialStatus.EARLY_STOPPED)
            )
            remaining = n_trials - terminal_count
            if remaining <= 0:
                break

            batch_size = self._current_concurrency(remaining)
            trials_batch = self.client.get_next_trials(
                max_trials=batch_size,
                fixed_parameters=fixed_parameters,
            )
            if not trials_batch:
                break

            oom_deferred: List[Tuple[int, Mapping[str, Any]]] = []

            if len(trials_batch) == 1:
                t_idx, params = next(iter(trials_batch.items()))
                gpu_id = self.gpu_ids[0] if self.gpu_ids else None
                config = self._build_trial_config(t_idx, params, gpu_id=gpu_id)
                result = self._run_single_trial(config)
                if result.is_oom and self.oom_retries > 0:
                    oom_deferred.append((t_idx, params))
                else:
                    self._record_trial_result(t_idx, config, result)
            else:
                configs: Dict[int, TConfig] = {}
                with ThreadPoolExecutor(max_workers=len(trials_batch)) as executor:
                    futures = {}
                    for slot, (t_idx, params) in enumerate(trials_batch.items()):
                        gpu_id = self.gpu_ids[slot % len(self.gpu_ids)] if self.gpu_ids else None
                        cfg = self._build_trial_config(t_idx, params, gpu_id=gpu_id)
                        configs[t_idx] = cfg
                        futures[executor.submit(self._run_single_trial, cfg)] = (t_idx, params)

                    for future in as_completed(futures):
                        t_idx, params = futures[future]
                        result = future.result()
                        if result.is_oom:
                            logger.warning(
                                "Trial #%d hit OOM during parallel batch (%d workers); deferring to solo retry.",
                                t_idx,
                                len(trials_batch),
                            )
                            oom_deferred.append((t_idx, params))
                        else:
                            self._record_trial_result(t_idx, configs[t_idx], result)

            for t_idx, params in oom_deferred:
                self._retry_oom_solo(t_idx, params)

            self.export_summary()

        return self.export_best()

    # -------------------------------------------------------------------------
    # Reporting & Best Extraction
    # -------------------------------------------------------------------------
    def get_best(
        self,
        use_model_predictions: bool = False,
    ) -> List[Tuple[TConfig, Dict[str, Tuple[float, Optional[float]]], int]]:
        """
        Return best trial(s) as `[(TrialConfig, prediction_dict, trial_index), ...]`.
        Dispatches to `get_pareto_frontier()` for MOO and `get_best_parameterization()`
        for single-objective experiments.
        """
        fixed = self.template.get_fixed()
        items: List[Tuple[Mapping[str, Any], Dict[str, Tuple[float, Optional[float]]], int]] = []

        if self.is_multi_objective:
            frontier = self.client.get_pareto_frontier(use_model_predictions=use_model_predictions)
            for params, pred, t_idx, _arm_name in frontier:
                items.append((params, pred, int(t_idx)))
        else:
            params, pred, t_idx, _arm_name = self.client.get_best_parameterization(
                use_model_predictions=use_model_predictions
            )
            items.append((params, pred, int(t_idx)))

        results: List[Tuple[TConfig, Dict[str, Tuple[float, Optional[float]]], int]] = []
        for params, pred, t_idx in items:
            cfg = self.config_cls.from_trial(
                params=params,
                fixed=fixed,
                experiment_name=self.name,
                trial_index=t_idx,
                trial_dir=self._trial_dir(t_idx),
            )
            results.append((cfg, pred, t_idx))
        return results

    def export_summary(self) -> str:
        """Write human-readable `summary.csv` from `client.summarize()`."""
        csv_path = os.path.join(self.exp_dir, "summary.csv")
        try:
            df = self.client.summarize()
            df.to_csv(csv_path, index=False)
        except Exception as exc:
            logger.debug("Skipping summary.csv export: %s", exc)
        return csv_path

    def export_best(
        self,
        use_model_predictions: bool = False,
    ) -> List[Tuple[TConfig, Dict[str, Tuple[float, Optional[float]]], int]]:
        """Export best / Pareto frontier configs to `best/` and write `best_summary.json`."""
        self.export_summary()
        try:
            best_items = self.get_best(use_model_predictions=use_model_predictions)
        except Exception as exc:
            logger.warning("Could not extract best parameterization: %s", exc)
            return []

        summary_entries = []
        for cfg, pred, t_idx in best_items:
            cfg.save(os.path.join(self.best_dir, f"trial_{t_idx:03d}.yaml"))
            summary_entries.append(
                {
                    "trial_index": t_idx,
                    "config_file": f"trial_{t_idx:03d}.yaml",
                    "predictions": {
                        k: [float(v[0]), None if v[1] is None else float(v[1])]
                        if isinstance(v, tuple)
                        else float(v)
                        for k, v in pred.items()
                    },
                }
            )

        with open(os.path.join(self.best_dir, "best_summary.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "experiment_name": self.name,
                    "objective": self.objective,
                    "best_trials": summary_entries,
                },
                f,
                indent=2,
            )
        return best_items
