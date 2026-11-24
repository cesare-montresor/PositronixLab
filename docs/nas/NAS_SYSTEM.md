# PositronixLab NAS System

SQLite-backed Neural Architecture Search (NAS) and Hyperparameter Optimization (HPO) layer built on **Ax 1.3+** (`ax.api.client.Client`), `TemplateBase`, and `parallelazy`-backed `TrialPipeline`.

## Architecture

```
TemplateBase (search space + fixed constants + constraints)
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│  Experiment[TConfig]  (Ax 1.3 Client + SQLite ax.db)              │
│                                                                   │
│  ask:  client.get_next_trials(max_trials=concurrency)             │
│           │                                                       │
│           ▼  TrialConfig.from_trial(params, fixed, trial_dir...)  │
│  run:  ThreadPoolExecutor ──► TrialPipeline.execute() / eval_fn   │
│           │                      └─ parallelazy PluginManager     │
│           ▼  (OOM? defer to solo retry with backoff)              │
│  tell: result.json + effective_config.yaml + client.complete/mark │
│           │                                                       │
│  export: summary.csv + best/trial_NNN.yaml + best_summary.json    │
└───────────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Flat, composable `TrialConfig`**: Subclasses `PipelineConfig` (`BaseConfig`) so concrete experiments compose domain config mixins (`MetaModelConfig`, `LayerSpaceConfig`, `TrainConfig`, `DataConfig`) for full IDE typing while remaining a flat key-value mapping for Ax.
2. **Immutable input vs. effective config**: `trials/trial_NNN/config.yaml` stores the exact Ax-requested parameters. Post-trial adjustments (such as `train_epochs` discovered via early stopping) are written separately to `effective_config.yaml` via `TrialResult.effective_overrides`.
3. **SQLite as single source of truth**: Every Ax mutation auto-persists to `<output_dir>/<name>/ax.db`. Re-instantiating `Experiment` with the same `name` and `output_dir` resumes seamlessly.
4. **Raw Ax metrics**: Metrics reported in `TrialResult.metrics` are never pre-negated; minimization is expressed in Ax's objective string (e.g. `objective="val_acc, -model_size"`).
5. **OOM resilience**: CUDA/GPU OOM errors detected via `is_oom_error()` are deferred out of parallel batches and retried solo (`concurrency=1`) with CUDA cache clearing and configurable backoff.

## Module Layout

| File | Contents |
| :--- | :--- |
| `positronixlab/nas/__init__.py` | Public re-exports (`Experiment`, `MetricValue`, `TrialConfig`, `TrialPipeline`, `TrialResult`, `is_oom_error`) |
| `positronixlab/nas/trial_config.py` | `TrialConfig(PipelineConfig)` and `from_trial(...)` factory |
| `positronixlab/nas/trial_pipeline.py` | `TrialPipeline(Pipeline)`, `TrialResult`, `MetricValue`, `is_oom_error` |
| `positronixlab/nas/experiment.py` | `Experiment[TConfig]` orchestration loop, SQLite resume, warm-start, exports |
| `positronixlab/nas/ax_best_practices.md` | Ax 1.3 guardrails and capability decision tree |
| `positronixlab/nas/ax_reference_guide.md` | Detailed Ax 1.3 API reference (`§1`–`§12`) |
| `positronixlab/nas/legacy/` | Archived pre-refactor 2-phase NAS code (do not use for new work) |

## On-Disk Experiment Layout

```text
<output_dir>/<name>/
├── ax.db                          # SQLite database (Ax source of truth)
├── template.yaml                  # Snapshot of TemplateBase search space & fixed params
├── experiment.yaml                # Experiment metadata (objective, concurrency, config_cls)
├── summary.csv                    # Tabular trial summary from client.summarize()
├── best/
│   ├── best_summary.json          # Best / Pareto-optimal trial indices & predictions
│   └── trial_NNN.yaml             # Serialized TrialConfig for each best/Pareto trial
└── trials/
    └── trial_NNN/                 # Also bound to parallelazy ManagerOptions.work_dir
        ├── config.yaml            # Immutable Ax input TrialConfig
        ├── effective_config.yaml  # Optional post-trial config with effective_overrides
        ├── result.json            # TrialResult (status, metrics, duration, metadata, errors)
        └── logs/                  # parallelazy logs (<run_id>/manager.log, ...)
```

---

## API Reference

### `trial_config.py`

#### `TrialConfig(PipelineConfig)`

Dataclass extending `PipelineConfig` (`BaseConfig` + `seed`, `device`) with NAS bookkeeping fields:

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `experiment_name` | `Optional[str]` | `None` | Owning `Experiment.name` |
| `trial_index` | `Optional[int]` | `None` | Ax integer trial index |
| `trial_dir` | `Optional[str]` | `None` | Per-trial directory (also `parallelazy` `work_dir`) |
| `gpu_id` | `Optional[int]` | `None` | Assigned GPU slot index |

#### `TrialConfig.from_trial(params, fixed=None, *, experiment_name=None, trial_index=None, trial_dir=None, gpu_id=None, name=None) -> TTrialConfig`

Merges `fixed` constants with Ax-generated `params` (`params` take precedence), attaches trial metadata, sets `device = f"cuda:{gpu_id}"` if `"device"` is absent, defaults `name` to `f"trial_{trial_index:03d}"`, and constructs the target subclass via `cls.from_dict(merged)` (silently dropping keys not declared on `cls`).

---

### `trial_pipeline.py`

#### `MetricValue`

`Union[float, Tuple[float, float]]` — either a scalar `mean` (Ax infers noise) or `(mean, sem)`.

#### `is_oom_error(message: str) -> bool`

Case-insensitive detector matching CUDA/cuBLAS/cuDNN OOM markers (`"out of memory"`, `"cudaerrormemoryallocation"`, `"cublas_status_alloc_failed"`, `"cudnn_status_alloc_failed"`, `"outofmemoryerror"`).

#### `TrialResult`

Dataclass representing the outcome of one trial execution:

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `status` | `str` | `"completed"` | `"completed"`, `"oom"`, `"timeout"`, or `"error"` |
| `metrics` | `Dict[str, MetricValue]` | `{}` | Raw Ax metrics (never pre-negated) |
| `duration_sec` | `float` | `0.0` | Wall-clock trial duration in seconds |
| `metadata` | `Dict[str, Any]` | `{}` | Non-Ax bookkeeping (e.g., `best_epoch`, `peak_vram_gb`) |
| `effective_overrides` | `Dict[str, Any]` | `{}` | Post-trial config overrides saved to `effective_config.yaml` |
| `errors` | `Dict[str, List[str]]` | `{}` | Per-plugin or pipeline error messages |
| `reason` | `Optional[str]` | `None` | Human-readable failure/timeout summary |

Properties & methods: `is_completed`, `is_oom`, `to_dict()`, `save(filepath)`.

#### `TrialPipeline(Pipeline)`

Abstract `parallelazy`-backed pipeline specialized for NAS trials:

- **`__init__(config: TrialConfig, options: Optional[PipelineOptions] = None)`**: Automatically sets `options.work_dir = config.trial_dir` when `config.trial_dir` is set and `options.work_dir is None`.
- **`setup() -> None`** *(abstract)*: Load plugins and populate shared plugin environment.
- **`is_complete() -> bool`** *(abstract, inherited from `Pipeline`)*: Poll condition for completion.
- **`extract_metrics(env: Dict[str, Any]) -> Dict[str, MetricValue]`**: Defaults to parsing `env[self.METRICS_ENV_KEY]` (`"metrics"`). Override for custom layouts.
- **`extract_metadata(env: Dict[str, Any]) -> Dict[str, Any]`**: Optional hook for `result.json` metadata.
- **`extract_effective_overrides(env: Dict[str, Any]) -> Dict[str, Any]`**: Optional hook for `effective_config.yaml`.
- **`execute(timeout_sec=None, poll_interval=None, parallel_plugins=None, terminate_on_error=None) -> TrialResult`**: Runs `setup() -> run() -> wait() -> extract_*() -> shutdown()` (guaranteed via `finally`), returning a classified `TrialResult`.

---

### `experiment.py`

#### `Experiment(Generic[TConfig])`

```python
Experiment(
    name: str,
    template: TemplateBase,
    objective: str,
    *,
    trial_pipeline_cls: Optional[Type[TrialPipeline]] = None,
    eval_fn: Optional[Callable[[TConfig], Union[TrialResult, Dict[str, MetricValue]]]] = None,
    config_cls: Type[TConfig] = TrialConfig,
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
)
```

Key methods:
- **`run(n_trials: int, fixed_parameters: Optional[Dict[str, Any]] = None)`**: Runs (or resumes) until `n_trials` trials reach `COMPLETED`, `ABANDONED`, or `EARLY_STOPPED`, exporting `summary.csv` after each batch and `best/` on finish.
- **`warm_start_from(prior_db_path: str, prior_experiment_name: str, filter_fn=None) -> int`**: Attaches completed trials from another SQLite Ax experiment as warm-start observations.
- **`get_best(use_model_predictions: bool = False)`**: Dispatches to `get_pareto_frontier()` for multi-objective (`"," in objective`) or `get_best_parameterization()` for single-objective.
- **`export_summary() -> str`**: Writes `<exp_dir>/summary.csv`.
- **`export_best(use_model_predictions: bool = False)`**: Writes `<exp_dir>/best/trial_NNN.yaml` and `best_summary.json`.

## Known Constraints & Caveats

- **SQLAlchemy global singleton**: Ax binds `SESSION_FACTORY` process-globally. Avoid interleaving writes across two live `Experiment` objects pointing to different SQLite files in the same process.
- **Failure-path bug (current HEAD)**: `experiment.py` passes `reason=` instead of `failed_reason=` to `Client.mark_trial_failed()` (see `analysis.md` §2). Until fixed, use `oom_final_status="abandoned"` and `reconcile_running="abandon"`, and catch non-OOM exceptions inside `eval_fn` if needed.
