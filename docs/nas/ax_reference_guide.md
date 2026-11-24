# Ax 1.3 Comprehensive Reference Guide (`ax-platform`)

Exhaustive, code-verified technical reference for `ax-platform` (>= 1.3, Python >= 3.11, `botorch>=0.18.1`).
Indexed by `ax_best_practices.md`. Each section is wrapped in `<!-- AX_REF:START §N -->` ... `<!-- AX_REF:END §N -->` markers for selective chunk retrieval.

---

<!-- AX_REF:START §1 -->
## §1. Legacy / Deprecated API Ban & 1.3 Migration Table

Never emit 0.x APIs in new code; migrate any encountered legacy usage using this code-verified mapping:

| DO NOT USE (Legacy 0.x) | 1.3 Replacement | Notes |
| :--- | :--- | :--- |
| `from ax.service.ax_client import AxClient` | `from ax import Client` (`ax.api.client.Client`) | `AxClient` is deprecated (removed in Ax 1.4.0). |
| `from ax.service.managed_loop import optimize` | `Client` ask-tell loop | Deprecated wrapper around `Client`. |
| `from ax.service.orchestrator import Orchestrator` | `from ax.orchestration.orchestrator import Orchestrator` | Or `Client.run_trials(...)` with `IRunner`/`IMetric`. |
| `Scheduler` / `SchedulerOptions` | `Orchestrator` / `OrchestratorOptions` | Moved to `ax.orchestration`. |
| `ax_client.create_experiment(parameters=[{...}])` | `client.configure_experiment(parameters=[RangeParameterConfig(...), ...])` | Typed dataclasses replace raw dicts. |
| `ObjectiveProperties` | `client.configure_optimization(objective="...", outcome_constraints=[...])` | SymPy expression strings. |
| `MultiObjective(...)`, `ScalarizedObjective(...)` | `Objective(expression="acc, -loss")` or `Objective(expression="2*m1 + m2")` | Unified into `Objective(expression=..., metric_name_to_signature=...)`. |
| `ObjectiveThreshold(...)`, `ScalarizedOutcomeConstraint(...)` | `OutcomeConstraint(expression=..., metric_name_to_signature=...)` | In `Client`, pass strings to `outcome_constraints=[...]`. |
| `HierarchicalSearchSpace(...)` | `SearchSpace(...)` or `ChoiceParameterConfig(..., dependent_parameters=...)` | Upstreamed into `SearchSpace`. |
| `GenerationStep(...)` | `GenerationNode(...)` + `TransitionCriterion` / `PausingCriterion` | `GenerationStep` is a legacy factory returning `GenerationNode`. |
| `GeneratorSpec(model_kwargs=..., model_gen_kwargs=..., model_cv_kwargs=...)` | `GeneratorSpec(generator_kwargs=..., generator_gen_kwargs=..., cv_kwargs=...)` | `model_*` kwargs removed in Ax 1.4. |
| `choose_generation_strategy_legacy(...)` | `client.configure_generation_strategy(...)` or `ax.api.utils.generation_strategy_dispatch.choose_generation_strategy(...)` | Legacy dispatcher builds `GenerationStep`s. |
| `ax_client.get_next_trial()` -> `(params, idx)` | `client.get_next_trials(max_trials=1)` -> `dict[int, TParameterization]` | Returns `{trial_index: parameters}` dict. |
| `ax_client.update_running_trial_with_intermediate_data(...)` | `client.attach_data(trial_index, raw_data, progression=step)` | Separates data attachment from completion. |
| `ax_client.stop_trial_early(idx)` | `client.mark_trial_early_stopped(idx)` | After `client.should_stop_trial_early(idx)` returns `True`. |
| `ax_client.log_trial_failure(idx)` / `abandon_trial(idx)` | `client.mark_trial_failed(idx, failed_reason=...)` / `client.mark_trial_abandoned(idx)` | `FAILED` = retryable; `ABANDONED` = non-retryable. |
| `ax_client.get_trials_data_frame()` | `client.summarize()` | Returns a `pd.DataFrame`. |
| `ax_client.get_contour_plot()` / `ax.plot.*` | `client.compute_analyses(analyses=[...])` | Modern `Analysis` / `PlotlyAnalysisCard` framework. |
| `Metric.fetch_trial_data_multi` / `fetch_experiment_data_multi` | `IMetric.fetch(...)` (API) or `Metric.bulk_fetch_trial_data` / `bulk_fetch_experiment_data` (core) | Legacy classmethods deprecated. |
<!-- AX_REF:END §1 -->

---

<!-- AX_REF:START §2 -->
## §2. Core Types, Config Dataclasses & `Client` Signatures

### Public Exports & Types (`ax/__init__.py`, `ax/api/types.py`, `ax/api/configs.py`)
```python
from ax import Client, ChoiceParameterConfig, RangeParameterConfig, StorageConfig, TOutcome, TParameterization
from ax.api.configs import DerivedParameterConfig
from ax.api.protocols import IMetric, IRunner
from ax.core.trial_status import TrialStatus

TParameterValue = int | float | str | bool
TParameterization = Mapping[str, TParameterValue]
TOutcome = Mapping[str, float | tuple[float, float]]  # metric_name -> mean | (mean, sem)
```

### Config Dataclasses
```python
@dataclass
class RangeParameterConfig:
    name: str
    bounds: tuple[float, float]                  # Inclusive (lower, upper)
    parameter_type: Literal["float", "int"]
    step_size: float | None = None               # Must evenly divide (upper - lower); linear only
    scaling: Literal["linear", "log"] | None = None  # "log" requires lower > 0

@dataclass
class ChoiceParameterConfig:
    name: str
    values: list[float] | list[int] | list[str] | list[bool]  # len == 1 -> FixedParameter
    parameter_type: Literal["float", "int", "str", "bool"]
    is_ordered: bool | None = None               # Default: False for str (>2 vals), True otherwise
    log_scale: bool | None = None                # Auto-detected for >=3 positive ordered numeric vals
    dependent_parameters: Mapping[TParameterValue, Sequence[str]] | None = None

@dataclass
class DerivedParameterConfig:
    name: str
    expression_str: str                          # Linear SymPy expr (float/int) or single param name (bool/str)
    parameter_type: Literal["float", "int", "str", "bool"]

@dataclass
class StorageConfig:
    creator: Callable[..., Any] | None = None
    url: str | None = None                       # e.g. "sqlite:///experiments.db"
    registry_bundle: RegistryBundleBase | None = None
```

### `Client` Complete Method Reference (`ax/api/client.py`)
```python
class Client(WithDBSettingsBase):
    def __init__(self, storage_config: StorageConfig | None = None, random_seed: int | None = None) -> None: ...

    # 1. Configure
    def configure_experiment(
        self,
        parameters: Sequence[RangeParameterConfig | ChoiceParameterConfig | DerivedParameterConfig],
        parameter_constraints: Sequence[str] | None = None,
        name: str | None = None,
        description: str | None = None,
        experiment_type: str | None = None,
        owner: str | None = None,
    ) -> None: ...
    def configure_optimization(
        self,
        objective: str,
        outcome_constraints: Sequence[str] | None = None,
        pruning_target_parameterization: TParameterization | None = None,
    ) -> None: ...
    def configure_generation_strategy(
        self,
        method: Literal["quality", "fast", "random_search"] = "fast",
        initialization_budget: int | None = None,
        initialization_random_seed: int | None = None,
        initialize_with_center: bool = True,
        use_existing_trials_for_initialization: bool = True,
        min_observed_initialization_trials: int | None = None,
        allow_exceeding_initialization_budget: bool = False,
        torch_device: str | None = None,
        simplify_parameter_changes: bool = False,
    ) -> None: ...
    def configure_tracking_metrics(self, metric_names: Sequence[str]) -> None: ...
    def configure_runner(self, runner: IRunner) -> None: ...
    def configure_metrics(self, metrics: Sequence[IMetric]) -> None: ...

    # 2. Conduct Experiment
    def get_next_trials(self, max_trials: int, fixed_parameters: TParameterization | None = None) -> dict[int, TParameterization]: ...
    def attach_data(self, trial_index: int, raw_data: TOutcome, progression: int | None = None) -> None: ...
    def complete_trial(self, trial_index: int, raw_data: TOutcome | None = None, progression: int | None = None) -> TrialStatus: ...
    def attach_trial(self, parameters: TParameterization, arm_name: str | None = None) -> int: ...
    def attach_baseline(self, parameters: TParameterization, arm_name: str | None = None) -> int: ...
    def should_stop_trial_early(self, trial_index: int) -> bool: ...
    def mark_trial_failed(self, trial_index: int, failed_reason: str | None = None) -> None: ...
    def mark_trial_abandoned(self, trial_index: int) -> None: ...
    def mark_trial_early_stopped(self, trial_index: int) -> None: ...
    def run_trials(
        self,
        max_trials: int,
        parallelism: int = 1,
        tolerated_trial_failure_rate: float = 0.5,
        initial_seconds_between_polls: int = 1,
    ) -> None: ...

    # 3. Analyze & Predict
    def get_best_parameterization(self, use_model_predictions: bool = True) -> tuple[TParameterization, TOutcome, int, str]: ...
    def get_pareto_frontier(self, use_model_predictions: bool = True) -> list[tuple[TParameterization, TOutcome, int, str]]: ...
    def predict(self, points: Sequence[TParameterization]) -> list[dict[str, tuple[float, float]]]: ...
    def summarize(
        self,
        trial_indices: Iterable[int] | None = None,
        trial_statuses: Sequence[Literal["candidate", "running", "failed", "completed", "abandoned", "early_stopped", "staged"]] | None = None,
    ) -> pd.DataFrame: ...
    def compute_analyses(self, analyses: Sequence[Analysis] | None = None, display: bool = True) -> list[AnalysisCardBase]: ...

    # 4. Save / Load
    def save_to_json_file(self, filepath: str = "ax_client_snapshot.json") -> None: ...
    @classmethod
    def load_from_json_file(cls, filepath: str = "ax_client_snapshot.json", storage_config: StorageConfig | None = None) -> Self: ...
    @classmethod
    def load_from_database(cls, experiment_name: str, storage_config: StorageConfig | None = None) -> Self: ...

    # 5. Power-User Setters (Non-API Escape Hatches — signatures may change across minor versions)
    def set_experiment(self, experiment: Experiment) -> None: ...
    def set_optimization_config(self, optimization_config: OptimizationConfig) -> None: ...
    def set_generation_strategy(self, generation_strategy: GenerationStrategy) -> None: ...
    def set_early_stopping_strategy(self, early_stopping_strategy: BaseEarlyStoppingStrategy) -> None: ...
    # Internal accessors: client._experiment, client._generation_strategy, client._early_stopping_strategy
```
<!-- AX_REF:END §2 -->

---

<!-- AX_REF:START §3 -->
## §3. Persistence: SQLite & JSON Storage

### Prerequisites
SQLAlchemy is an optional dependency in `ax-platform`:
```shell
pip install sqlalchemy   # or: pip install "ax-platform[mysql]"
```

### SQLite Auto-Persistence
When `Client(storage_config=StorageConfig(url="sqlite:///..."))` is used, **every mutating `Client` method automatically saves** to the database (`Experiment`, `Trial`s, `Data`, `GenerationStrategy`, and `AnalysisCard`s).

```python
from ax import Client, StorageConfig
from ax.storage.sqa_store.db import create_all_tables, get_engine, init_engine_and_session_factory

url = "sqlite:///path/to/experiments.db"
# Required for new SQLite files (creates experiment_v2 etc.) and when switching
# DB URLs in the same Python process (SESSION_FACTORY is a process-global singleton).
init_engine_and_session_factory(url=url, force_init=True)
create_all_tables(get_engine())

storage_config = StorageConfig(url=url)

# Create & run
client = Client(storage_config=storage_config, random_seed=42)
client.configure_experiment(name="exp_nas_001", parameters=[...])  # Unique `name` is REQUIRED
client.configure_optimization(objective="acc, -latency")

# Resume in a new process (loads Experiment + GenerationStrategy)
resumed = Client.load_from_database(experiment_name="exp_nas_001", storage_config=storage_config)
```

### Persistence Caveats (Code-Verified)
1. **Schema creation & global `SESSION_FACTORY`:** `Client(storage_config=StorageConfig(url=...))` does **not** auto-create SQLite tables on a brand-new DB file (raises `sqlite3.OperationalError: no such table: experiment_v2`), and `ax.storage.sqa_store.db.SESSION_FACTORY` is cached process-wide. Always call `init_engine_and_session_factory(url=url, force_init=True)` + `create_all_tables(get_engine())` before opening a new or different SQLite file in the same process.
2. **Non-persisted state:** `client.set_early_stopping_strategy(...)` and `client._random_seed` are **not** stored in SQLite. Re-apply `set_early_stopping_strategy(...)` after `load_from_database(...)`. (Sobol's internal sequence state *is* persisted on `GeneratorRun`s.)
3. **Custom `IRunner` / `IMetric` serialization:** Register custom classes via `RegistryBundle`:
   ```python
   from ax.storage.registry_bundle import RegistryBundle

   storage_config = StorageConfig(
       url="sqlite:///experiments.db",
       registry_bundle=RegistryBundle(runner_clss={MyRunner: None}, metric_clss={MyMetric: None}),
   )
   ```
4. **Concurrency:** For SQLite, use a single coordinator process owning `Client` while worker processes/GPUs evaluate trials and return results. Avoid concurrent writes to the same SQLite file from multiple independent `Client` processes.

### JSON Snapshots (Zero DB Dependencies)
```python
client.save_to_json_file("snapshot.json")
restored = Client.load_from_json_file("snapshot.json")  # Optional: storage_config=...
```
<!-- AX_REF:END §3 -->

---

<!-- AX_REF:START §4 -->
## §4. Search Spaces: Ranges, Choices, Derived, Constraints & Hierarchical (NAS)

### Parameter Details
- **`RangeParameterConfig`**:
  - `scaling="log"` requires `bounds[0] > 0`; incompatible with `step_size` and `parameter_constraints`.
  - `step_size` must evenly divide `upper - lower`; internally constructs an ordered `ChoiceParameter`.
- **`ChoiceParameterConfig`**:
  - Always set `is_ordered` explicitly to avoid `AxParameterWarning`. 2-value choices are always treated as ordered.
  - `log_scale=None` auto-detects log scale for >=3 positive ordered numeric choices with geometric spacing or spanning >=2 orders of magnitude.
  - `len(values) == 1` creates a `FixedParameter`.
- **`DerivedParameterConfig`**:
  - Linear SymPy expressions for `float`/`int` (e.g., `"2 * d_in + 16"`); simple single-parameter copy only for `str`/`bool`.
  - Cannot derive from another `DerivedParameter` or a `FixedParameter`; cannot appear in `parameter_constraints`.

### Dependent / Hierarchical Parameters (NAS)
Define multi-level conditional spaces with `dependent_parameters: Mapping[TParameterValue, Sequence[str]]`:

```python
parameters = [
    ChoiceParameterConfig(
        name="block_type",
        parameter_type="str",
        values=["conv", "transformer", "mlp"],
        is_ordered=False,
        dependent_parameters={
            "conv": ["conv_type", "channels"],
            "transformer": ["num_heads", "head_dim"],
            # "mlp" has no dependents (valid)
        },
    ),
    # Level 2 hierarchy nested under "conv":
    ChoiceParameterConfig(
        name="conv_type",
        parameter_type="str",
        values=["standard", "depthwise"],
        is_ordered=False,
        dependent_parameters={"standard": ["kernel_size"]},
    ),
    ChoiceParameterConfig(name="kernel_size", parameter_type="int", values=[3, 5, 7], is_ordered=True),
    RangeParameterConfig(name="channels", bounds=(32, 512), parameter_type="int"),
    ChoiceParameterConfig(name="num_heads", parameter_type="int", values=[2, 4, 8], is_ordered=True),
    RangeParameterConfig(name="head_dim", bounds=(16, 128), parameter_type="int"),
]
```

**Strict Hierarchical Rules (Verified in `ax/core/search_space.py`):**
- **Strict tree (no cycles / no shared child across branches):** Every dependent parameter must have a single unique path from a top-level parameter. If two branches need `dropout`, name them `conv_dropout` and `vit_dropout`, or make `dropout` top-level.
- **Active-branch-only parameterizations:** `get_next_trials` returns only active parameters for the chosen branch. Calls to `attach_trial`, `attach_baseline`, and `predict` **must** omit inactive branch parameters or validation will fail.
- **Inspection helper:** `client._experiment.search_space.hierarchical_structure_str(parameter_names_only=True)`.

### Linear Parameter Constraints
Passed via `configure_experiment(parameter_constraints=[...])`:
- Linear inequalities (`<=`, `>=`) only: e.g., `["x1 <= x2", "w1 + w2 + w3 <= 1.0", "2 * d_in - d_out <= 0"]`.
- Allowed **only** on non-log `RangeParameterConfig` and numeric ordered non-log `ChoiceParameterConfig`.
- **Equality workaround (`x1 + x2 + x3 == 1`):** Optimize `x1`, `x2` with `"x1 + x2 <= 1"` and define `DerivedParameterConfig(name="x3", expression_str="1 - x1 - x2", parameter_type="float")`.
<!-- AX_REF:END §4 -->

---

<!-- AX_REF:START §5 -->
## §5. Fixed Parameters, Fixed Features & Dynamic Search Space Updates

1. **Experiment-Wide Fixed Parameter:** Pass a 1-element `ChoiceParameterConfig(name="version", values=["v2"], parameter_type="str")`. Excluded from GP dims via `RemoveFixed`, kept in all arm dicts.
2. **Per-Call Fixed Features (`Client.get_next_trials`):**
   ```python
   trials = client.get_next_trials(max_trials=2, fixed_parameters={"block_type": "transformer", "lr": 1e-3})
   ```
   Pins specified tunable parameters for that generation call only (works with Sobol and BoTorch). *Note:* `fixed_parameters` overrides algorithmically determined `fixed_features` (like disabled params), so include all needed fixed keys if combining both.
3. **Node-Level Fixed Features (Custom `GenerationStrategy`):**
   `spec.fixed_features = ObservationFeatures(parameters={"fidelity": 1.0})` on a `GeneratorSpec`.
4. **Dynamic Search Space Updates Mid-Experiment (Core Escape Hatch):**
   - **Disable parameters** (freeze for future trials while keeping historical trials valid):
     ```python
     client._experiment.search_space.disable_parameters({"dropout": 0.1})
     ```
   - **Add parameters** after trials have run (must provide `backfill_value` on the `ax.core.parameter.*` object for existing trials):
     ```python
     from ax.core.parameter import RangeParameter, ParameterType
     client._experiment.search_space.add_parameters([
         RangeParameter("weight_decay", ParameterType.FLOAT, 0.0, 0.1, backfill_value=0.0)
     ])
     ```
<!-- AX_REF:END §5 -->

---

<!-- AX_REF:START §6 -->
## §6. Objectives, Multi-Objective Optimization (MOO), Constraints & Tracking Metrics

### SymPy Objective Expressions (`configure_optimization`)
Expressions are **maximized**. Prefix with `-` to minimize. Always report raw (un-negated) values to `complete_trial`/`attach_data`.

| Mode | `objective` String | Optimal Extraction |
| :--- | :--- | :--- |
| **Single** | `"val_acc"` or `"-val_loss"` | `client.get_best_parameterization()` |
| **Scalarized** | `"2.0 * val_acc - 0.1 * latency_ms"` | `client.get_best_parameterization()` |
| **Multi-Objective** | `"val_acc, -latency_ms, -model_mb"` | `client.get_pareto_frontier()` |

### MOO Objective Thresholds vs. Outcome Constraints
- Keep MOO to **2–4 objectives**.
- In MOO, any single-metric constraint in `outcome_constraints` on an objective metric aligned with its optimization direction (`>=` for maximized, `<=` for minimized) is **automatically converted to an `objective_threshold` (reference point)** for hypervolume (`qLogNEHVI`). Set thresholds ~10% worse than minimum acceptable performance.
- Constraints on non-objective metrics (or bounding an objective *against* its optimization direction, e.g., `"flops >= 1e6"` while minimizing `flops`) remain true outcome constraints.
- In **single-objective** mode, constraining the objective metric raises `UserInputError`.

```python
client.configure_optimization(
    objective="val_acc, -latency_ms",
    outcome_constraints=[
        "val_acc >= 0.75",      # -> ObjectiveThreshold
        "latency_ms <= 50.0",   # -> ObjectiveThreshold
        "vram_gb <= 24.0",      # -> OutcomeConstraint
    ],
)
```

### Relative Outcome Constraints (`* baseline`)
```python
base_idx = client.attach_baseline(parameters=prod_params)
client.complete_trial(base_idx, raw_data={"val_acc": 0.85, "qps": 1200.0})
client.configure_optimization(objective="val_acc", outcome_constraints=["qps >= 0.95 * baseline"])
```
Scalarized outcome constraints (e.g., `"0.5*m1 + 0.5*m2 >= ..."`) **cannot** be relative.

### Tracking Metrics
- Explicit: `client.configure_tracking_metrics(["train_loss", "gpu_mem"])`.
- Implicit: Unexpected metric keys in `attach_data`/`complete_trial` are auto-registered as tracking metrics.
- Note: in `ax-platform 1.3.1`, `Client.configure_generation_strategy(...)` does **not** accept a `fit_tracking_metrics` keyword argument (only lower-level generator/node specs expose it).
<!-- AX_REF:END §6 -->

---

<!-- AX_REF:START §7 -->
## §7. Bayesian Optimization, Generation Strategies & High-Dim (BONSAI + MAP-SAAS)

### Built-In Dispatch (`configure_generation_strategy`)
If not called, defaults to `method="fast"` on first `get_next_trials`/`run_trials`:
- **`method="fast"`**: `CenterGenerationNode` → `Sobol` → `BOTORCH_MODULAR` (auto-selects `SingleTaskGP`/`MixedSingleTaskGP`, `qLogNEI` or `qLogNEHVI`, and `optimize_acqf_mixed_alternating`). Best default for <= ~20 parameters.
- **`method="quality"`**: `CenterGenerationNode` → `Sobol` → `SaasFullyBayesianSingleTaskGP(use_input_warping=True)` (Warped SAASBO). Requires `pip install "ax-platform[fully_bayesian]"` (JAX/NumPyro). Slower fit, higher sample efficiency on noisy/high-dim tasks.
- **`method="random_search"`**: `CenterGenerationNode` → `Sobol` only.

### Initialization Budget & `DataRequiredError`
- Default `initialization_budget=5`; `min_observed_initialization_trials=max(1, budget // 2)`.
- Sobol transitions to `MBM` when BOTH: (1) non-`FAILED`/non-`ABANDONED` trials >= `initialization_budget`, AND (2) `COMPLETED` trials with data >= `min_observed_initialization_trials`.
- If `allow_exceeding_initialization_budget=False` (default), requesting more trials after hitting `initialization_budget` before `min_observed_initialization_trials` complete raises `DataRequiredError`. In async parallel pools, set `initialization_budget >= parallelism` or `allow_exceeding_initialization_budget=True`.
- `initialize_with_center=True` evaluates the search-space midpoint first (falling back to the LP Chebyshev center when linear `parameter_constraints` make the naive midpoint infeasible).

### High-Dimensional BO: BONSAI + MAP-SAAS (20–100+ Params, Fast, Sparse Changes)
Uses `EnsembleMapSaasSingleTaskGP` (MAP-estimated SAAS ensemble, no JAX/NumPyro required) + BONSAI pruning (`simplify_parameter_changes=True`), snapping irrelevant parameters back to `pruning_target_parameterization` (or `status_quo`):

```python
from ax.api.utils.generation_strategy_dispatch import choose_generation_strategy
from ax.api.utils.structs import GenerationStrategyDispatchStruct
from ax.generators.torch.botorch_modular.surrogate import ModelConfig
from botorch.models.map_saas import EnsembleMapSaasSingleTaskGP

client.configure_optimization(objective="-loss", pruning_target_parameterization=default_params)
gs = choose_generation_strategy(
    struct=GenerationStrategyDispatchStruct(
        method="custom",
        initialization_budget=10,
        initialize_with_center=True,
        simplify_parameter_changes=True,
        torch_device="cuda",  # optional
    ),
    model_config=ModelConfig(botorch_model_class=EnsembleMapSaasSingleTaskGP, name="BONSAI"),
)
client.set_generation_strategy(gs)
```
<!-- AX_REF:END §7 -->

---

<!-- AX_REF:START §8 -->
## §8. Ask-Tell Execution, Parallelism, Warm-Starting & Trial Lifecycle

### Warm-Starting from Historical Data / DataFrame
```python
for _, row in historical_df.iterrows():
    idx = client.attach_trial(parameters={"x1": float(row["x1"]), "x2": float(row["x2"])}, arm_name=str(row["arm"]))
    client.complete_trial(trial_index=idx, raw_data={"loss": float(row["loss"])})
```
With `use_existing_trials_for_initialization=True` (default), historical trials satisfy the init budget immediately.

### Noise / SEM Reporting
- `{"m": 0.85}` → `sem=NaN` (GP infers noise level from data).
- `{"m": (0.85, 0.02)}` → known SEM (e.g., cross-validation standard error).
- `{"m": (0.85, 0.0)}` → noiseless deterministic evaluation.

### Trial Lifecycle & Statuses (`ax.core.trial_status.TrialStatus`)
`CANDIDATE` → `STAGED` → `RUNNING` → `COMPLETED` | `FAILED` | `EARLY_STOPPED` | `ABANDONED` | `STALE`.

| Client Method | Status | Re-suggested? | Modeled? | Use Case |
| :--- | :--- | :--- | :--- | :--- |
| `complete_trial(idx, raw_data)` | `COMPLETED` | No | Yes | Normal completion (partial metrics allowed & modeled). |
| `mark_trial_failed(idx, reason)` | `FAILED` | **Yes** | No | Transient failure (preemption, network/node crash). |
| `mark_trial_abandoned(idx)` | `ABANDONED` | **No** | No | Deterministic failure (OOM, NaN divergence, invalid config) with no data. |
| `mark_trial_early_stopped(idx)` | `EARLY_STOPPED` | **No** | Partial | Stopped mid-run after attaching intermediate progression data. |

*Note on `BatchTrial`:* Only use `BatchTrial` when arms must be evaluated *jointly* in the same environment (e.g., non-stationary online A/B test). For independent parallel evaluations, always use multiple 1-arm `Trial`s.
<!-- AX_REF:END §8 -->

---

<!-- AX_REF:START §9 -->
## §9. Early Stopping & Global Stopping

### Trial-Level Early Stopping Strategies (`ax.early_stopping.strategies`)
- **`PercentileEarlyStoppingStrategy`** *(default if `should_stop_trial_early` is called without `set_early_stopping_strategy`)*:
  ```python
  PercentileEarlyStoppingStrategy(
      metric_signatures: Iterable[str] | None = None,  # Length-1; None = objective metric
      percentile_threshold: float = 50.0,              # Stop bottom X% at same step
      min_progression: float | None = 10,              # Must be >= patience
      max_progression: float | None = None,            # Don't stop trials past this step
      min_curves: int | None = 5,                      # Min trials reaching min_progression
      normalize_progressions: bool = False,            # Map progressions to [0, 1]
      n_best_trials_to_complete: int | None = None,    # Protect top-N trials from stopping
      interval: float | None = None,                   # Throttle checks to min_prog + k * interval
      patience: float = 0.0,                           # Require underperformance across [t - patience, t]
      check_safe: bool = False,                        # Gate stopping if simulation shows global best would stop
  )
  ```
- **`ThresholdEarlyStoppingStrategy`**:
  ```python
  ThresholdEarlyStoppingStrategy(
      metric_signatures=None, metric_threshold=0.2, min_progression=10,
      max_progression=None, min_curves=5, normalize_progressions=False, check_safe=False,
  )
  ```
- **`StabilityGatedEarlyStoppingStrategy`** (also in `ax.early_stopping.strategies`).
- **Multi-Metric Composition:**
  `AndEarlyStoppingStrategy(left, right)` / `OrEarlyStoppingStrategy(left, right)` / `OrEarlyStoppingStrategy.from_early_stopping_strategies([s1, s2, ...])`.

### Ask-Tell Progression Loop
```python
client.set_early_stopping_strategy(
    PercentileEarlyStoppingStrategy(percentile_threshold=50.0, min_progression=5, min_curves=5, patience=2.0, n_best_trials_to_complete=2)
)
for epoch in range(1, max_epochs + 1):
    metrics = train_and_eval_epoch(params, epoch)
    client.attach_data(trial_index=trial_idx, raw_data=metrics, progression=epoch)
    if client.should_stop_trial_early(trial_index=trial_idx):
        client.mark_trial_early_stopped(trial_index=trial_idx)
        break
else:
    client.complete_trial(trial_index=trial_idx)
```

### Global Stopping (`ax.global_stopping.strategies.improvement`)
```python
from ax.global_stopping.strategies.improvement import ImprovementGlobalStoppingStrategy

gss = ImprovementGlobalStoppingStrategy(
    min_trials=20,                      # Min completed trials before checking
    window_size=5,                      # Recent trials window
    improvement_bar=0.05,               # Single-obj: fraction of IQR; MOO: fraction of hypervolume
    inactive_when_pending_trials=True,
)
should_stop, reason = gss.should_stop_optimization(experiment=client._experiment)
```
*Limitation:* Does not support scalarized objectives or scalarized outcome constraints.
<!-- AX_REF:END §9 -->

---

<!-- AX_REF:START §10 -->
## §10. Closed-Loop Orchestration (`IRunner`, `IMetric`, `Orchestrator`)

### Protocol Subclasses (`ax.api.protocols`)
```python
from typing import Any, Mapping
from ax.api.protocols import IMetric, IRunner
from ax.api.types import TParameterization
from ax.core.trial_status import TrialStatus

class JobRunner(IRunner):
    def run_trial(self, trial_index: int, parameterization: TParameterization) -> dict[str, Any]:
        return {"job_id": submit_job(parameterization)}  # MUST be JSON-serializable

    def poll_trial(self, trial_index: int, trial_metadata: Mapping[str, Any]) -> TrialStatus:
        return poll_job(trial_metadata["job_id"])

    def stop_trial(self, trial_index: int, trial_metadata: Mapping[str, Any]) -> dict[str, Any]:
        cancel_job(trial_metadata["job_id"])
        return {}

class JobMetric(IMetric):
    def fetch(self, trial_index: int, trial_metadata: Mapping[str, Any]) -> tuple[int, float | tuple[float, float]]:
        step, mean, sem = read_job_metric(trial_metadata["job_id"])
        return (step, (mean, sem))  # or (step, mean)

client.configure_runner(JobRunner())
client.configure_metrics([JobMetric(name="val_loss")])  # Name must match objective/constraint
client.run_trials(max_trials=40, parallelism=4, tolerated_trial_failure_rate=0.2, initial_seconds_between_polls=5)
```

### Direct `Orchestrator` (Full `OrchestratorOptions`)
Use `ax.orchestration.orchestrator.Orchestrator` directly when you need TTLs, early/global stopping, or batch dispatch in closed-loop mode:
```python
from ax.orchestration.orchestrator import Orchestrator
from ax.orchestration.orchestrator_options import OrchestratorOptions, TrialType

orch = Orchestrator(
    experiment=client._experiment,
    generation_strategy=client._generation_strategy_or_choose(),
    options=OrchestratorOptions(
        max_pending_trials=4,
        trial_type=TrialType.TRIAL,
        total_trials=50,
        tolerated_trial_failure_rate=0.2,
        min_failed_trials_for_failure_rate_check=5,
        ttl_seconds_for_trials=3600,
        init_seconds_between_polls=5,
        min_seconds_before_poll=1.0,
        seconds_between_polls_backoff_factor=1.5,  # Auto-set to 1 when early_stopping_strategy is set
        run_trials_in_batches=False,
        early_stopping_strategy=my_ess,
        global_stopping_strategy=my_gss,
        wait_for_running_trials=True,
        terminate_if_status_quo_infeasible=False,
    ),
    db_settings=client.db_settings,
)
orch.run_n_trials(max_trials=50)
```
<!-- AX_REF:END §10 -->

---

<!-- AX_REF:START §11 -->
## §11. Advanced Customization: Modular BoTorch, Custom Nodes & External Generators

### `Generators` Registry Enum (`ax.adapter.registry.Generators`)
`SOBOL`, `BOTORCH_MODULAR`, `BO_MIXED`, `SAASBO`, `SAAS_MTGP`, `ST_MTGP`, `FACTORIAL`, `THOMPSON`, `EMPIRICAL_BAYES_THOMPSON`, `EB_ASHR`, `UNIFORM`, `IN_SAMPLE_UNIFORM`, `BOTL`.

### `GenerationNode`, `TransitionCriterion` & `PausingCriterion`
- **`GenerationNode(name, generator_specs, transition_criteria=None, pausing_criteria=None, best_model_selector=None, should_deduplicate=False, fallback_specs=None)`**
- **`CenterGenerationNode(next_node_name, use_existing_trials_for_initialization=False)`**
- **`TransitionCriterion` (`ax.generation_strategy.transition_criterion`):**
  - `MinTrials(threshold, transition_to, only_in_statuses=None, not_in_statuses=None, use_all_trials_in_exp=False, count_only_trials_with_data=False)`
  - `AutoTransitionAfterGen(transition_to, continue_trial_generation=True)`
  - `IsSingleObjective(transition_to)`
  - `AuxiliaryExperimentCheck(transition_to, auxiliary_experiment_purposes_to_include=None, auxiliary_experiment_purposes_to_exclude=None)`
- **`PausingCriterion`:**
  - `MaxTrialsAwaitingData(threshold, only_in_statuses=None, not_in_statuses=[FAILED, ABANDONED], use_all_trials_in_exp=False)` -> raises `DataRequiredError`.
  - `MaxGenerationParallelism(threshold, only_in_statuses=[TrialStatus.RUNNING])` -> raises `MaxParallelismReachedException`.

### Modular BoTorch (`SurrogateSpec`, `ModelConfig` & CV Model Selection)
```python
from ax.adapter.registry import Generators
from ax.generation_strategy.center_generation_node import CenterGenerationNode
from ax.generation_strategy.generation_node import GenerationNode
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generation_strategy.generator_spec import GeneratorSpec
from ax.generation_strategy.transition_criterion import MaxTrialsAwaitingData, MinTrials
from ax.generators.torch.botorch_modular.surrogate import ModelConfig, SurrogateSpec
from ax.utils.stats.model_fit_stats import MSE  # or RANK_CORRELATION; or AIC/BIC/MLL from botorch.utils.evaluation
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.models.map_saas import AdditiveMapSaasSingleTaskGP
from botorch.models.transforms.input import Warp
from gpytorch.kernels import MaternKernel

surrogate_spec = SurrogateSpec(
    model_configs=[
        ModelConfig(botorch_model_class=AdditiveMapSaasSingleTaskGP, input_transform_classes=[Warp]),
        ModelConfig(botorch_model_class=SingleTaskGP, covar_module_class=MaternKernel, covar_module_options={"nu": 2.5}),
    ],
    eval_criterion=MSE,
    allow_batched_models=False,  # Required for per-metric CV model selection
    num_folds=10,                # None = leave-one-out CV
)
bo_node = GenerationNode(
    name="CustomMBM",
    generator_specs=[
        GeneratorSpec(
            generator_enum=Generators.BOTORCH_MODULAR,
            generator_kwargs={"surrogate_spec": surrogate_spec, "botorch_acqf_class": qLogNoisyExpectedImprovement},
            generator_gen_kwargs={
                "model_gen_options": {"optimizer_kwargs": {"num_restarts": 20, "options": {"batch_limit": 5, "maxiter": 200}}}
            },
        )
    ],
    should_deduplicate=True,
)
sobol_node = GenerationNode(
    name="Sobol",
    generator_specs=[GeneratorSpec(Generators.SOBOL, generator_kwargs={"seed": 0})],
    transition_criteria=[MinTrials(threshold=5, transition_to="CustomMBM", use_all_trials_in_exp=True)],
    pausing_criteria=[MaxTrialsAwaitingData(threshold=5)],
)
client.set_generation_strategy(
    GenerationStrategy(name="Center+Sobol+CustomMBM", nodes=[CenterGenerationNode(next_node_name="Sobol"), sobol_node, bo_node])
)
```

### Registering Custom BoTorch Models & Acquisition Functions
- **Custom Model:** Subclass `ExactGP, GPyTorchModel` with `_num_outputs = 1` and `@classmethod construct_inputs(cls, training_data: SupervisedDataset) -> dict[str, Tensor]`. Optionally register `@submodel_input_constructor.register(MyGP)` (`ax.generators.torch.botorch_modular.surrogate`) and `@fit_botorch_model.register(MyGP)` (`ax.generators.torch.botorch_modular.utils`).
- **Custom Acquisition Function:** Subclass `MCAcquisitionFunction` and register `@acqf_input_constructor(MyAcqf)` (`botorch.acquisition.input_constructors`).

### External Non-BoTorch Generator (`ExternalGenerationNode`)
```python
from ax.generation_strategy.external_generation_node import ExternalGenerationNode

class CustomEvolutionNode(ExternalGenerationNode):
    def __init__(self) -> None:
        super().__init__(name="CustomEvolution", should_deduplicate=True)

    def update_generator_state(self, experiment, data) -> None:
        # Update internal optimizer state from experiment.trials and data.df
        ...

    def get_next_candidate(self, pending_parameters) -> dict:
        # Return next valid TParameterization (active branch only if hierarchical)
        ...
```
<!-- AX_REF:END §11 -->

---

<!-- AX_REF:START §12 -->
## §12. Predictions, Summaries, Analyses & Custom Analysis Cards

### Extraction & Prediction
```python
# Single-objective best feasible in-sample arm:
best_params, pred_dict, trial_idx, arm_name = client.get_best_parameterization(use_model_predictions=True)

# Multi-objective feasible in-sample Pareto frontier:
for params, pred_dict, trial_idx, arm_name in client.get_pareto_frontier(use_model_predictions=True):
    # pred_dict: {metric_name: (mean, sem)} when model-predicted
    ...

# Out-of-sample predictions at arbitrary valid points -> list[dict[metric_name, (mean, sem)]]:
preds = client.predict(points=[cand_1, cand_2])

# Tabular summary DataFrame:
df = client.summarize(trial_statuses=["completed", "early_stopped"])
```

### Built-In Analysis Catalog
- **Tabular (`ax.analysis`):** `OverviewAnalysis` *(default for `compute_analyses()`)*, `Summary`, `BestTrials`, `MetricSummary`, `SearchSpaceSummary`.
- **Plotly (`ax.analysis.plotly`):** `ArmEffectsPlot`, `ScatterPlot`, `SlicePlot`, `ContourPlot`, `TopSurfacesAnalysis`, `SensitivityAnalysisPlot`, `CrossValidationPlot`, `ParallelCoordinatesPlot`, `ProgressionPlot`, `UtilityProgressionAnalysis`, `ConstraintFeasibilityPlot`, `PFeasiblePlot`, `MarginalEffectsPlot`, `BanditRollout`.
- **Healthchecks (`ax.analysis.healthcheck`):** `BaselineImprovementAnalysis`, `CanGenerateCandidatesAnalysis`, `ComplexityRatingAnalysis`, `ConstraintsFeasibilityAnalysis`, `EarlyStoppingAnalysis`, `PredictableMetricsAnalysis`, `RegressionAnalysis`, `SearchSpaceAnalysis`, `TransferLearningAnalysis`.

```python
from ax.analysis.plotly import CrossValidationPlot, UtilityProgressionAnalysis, ScatterPlot, SensitivityAnalysisPlot
from ax.analysis.healthcheck import ConstraintsFeasibilityAnalysis
from ax.analysis.summary import Summary

# Set display=False in scripts/CI; access card.df or card.get_figure()
cards = client.compute_analyses(
    analyses=[
        Summary(),
        CrossValidationPlot(),
        UtilityProgressionAnalysis(),
        ScatterPlot(x_metric_name="latency_ms", y_metric_name="val_acc"),
        SensitivityAnalysisPlot(metric_name="val_acc"),
        ConstraintsFeasibilityAnalysis(),
    ],
    display=False,
)
```

### Writing a Custom `Analysis` / `PlotlyAnalysisCard`
```python
import pandas as pd
from plotly import express as px
from ax.analysis.analysis import Analysis
from ax.analysis.plotly.plotly_analysis import PlotlyAnalysisCard, create_plotly_analysis_card

class TrialStatusBarChart(Analysis):
    def compute(self, experiment=None, generation_strategy=None, adapter=None) -> PlotlyAnalysisCard:
        records = [{"status": s.name, "count": len(ts)} for s, ts in experiment.trials_by_status.items() if ts]
        df = pd.DataFrame.from_records(records)
        return create_plotly_analysis_card(
            name=self.__class__.__name__,
            title="Trials by Status",
            subtitle="Count of trials per status",
            df=df,
            fig=px.bar(df, x="status", y="count"),
        )
```
<!-- AX_REF:END §12 -->