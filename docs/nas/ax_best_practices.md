# Ax 1.3 Agent Context & Lookup Index (`ax-platform`)

Always-load context file for developing on `ax-platform` (>= 1.3, Python >= 3.11, `botorch>=0.18.1`).

---

## 0. AGENT USAGE PROTOCOL (READ FIRST)

1. **Always start here:** Use this file for 1.3 guardrails, architecture decisions, and section routing.
2. **Never guess from pretraining & avoid scanning the Ax repo:** Pretraining weights are heavily biased toward deprecated Ax 0.x (`AxClient`, `Scheduler`, `GenerationStep`). Do **not** guess signatures or browse the `ax-platform` source tree unless a specific detail is absent from `ax_reference_guide.md`.
3. **Read `ax_reference_guide.md` selectively by chunk markers:** Every section `§1`–`§12` in `ax_reference_guide.md` is delimited by exact markers:
   - `<!-- AX_REF:START §N -->` ... `<!-- AX_REF:END §N -->`
   - Before writing code for any capability beyond the minimal skeleton in Section 2, identify the relevant `§N` tag(s) in Sections 3–4, locate their line bounds, and read **only those line ranges**:
     ```bash
     # Example: find exact line ranges for §3 (SQLite) and §4 (Hierarchical/NAS)
     rg -n "AX_REF:(START|END) §(3|4)\b" ax_reference_guide.md
     ```
   - Do **not** load the entire `ax_reference_guide.md` into context unless you are designing across almost all subsystems at once.

---

## 1. HARD GUARDRAILS — NEVER USE

* **NEVER** use legacy 0.x APIs: `AxClient` (`ax.service.ax_client`), `optimize` (`ax.service.managed_loop`), `Scheduler`, `ObjectiveProperties`, `MultiObjective`, `ScalarizedObjective`, `ObjectiveThreshold`, `ScalarizedOutcomeConstraint`, `HierarchicalSearchSpace`, or `GenerationStep`. (Full migration table in `§1`.)
* **ALWAYS** use the 1.3 unified `Client` API:
  ```python
  from ax import Client, RangeParameterConfig, ChoiceParameterConfig, StorageConfig
  from ax.api.configs import DerivedParameterConfig
  from ax.api.protocols import IMetric, IRunner
  from ax.core.trial_status import TrialStatus
  ```
* **NEVER** pass `objectives=` (plural) to `configure_optimization` — the argument is `objective: str` (singular).
* **NEVER** pass `url=` to `Client(...)` — pass `Client(storage_config=StorageConfig(url="sqlite:///..."))`.
* **NEVER** call `get_next_trials(...)()` — it returns `dict[int, TParameterization]` directly (`{trial_index: params}`).
* **NEVER** negate metric values in `complete_trial`/`attach_data` when minimizing. Use `objective="-loss"` and report raw `loss`.
* **NEVER** use `BatchTrial` for independent parallel workers (GPU/CPU pools). Use multiple 1-arm `Trial`s via `get_next_trials(max_trials=N)` or `run_trials(parallelism=N)`.

---

## 2. Minimal Golden-Path Skeleton (SQLite + MOO + Dependent Params + Fixed Features + BO)

```python
from ax import Client, RangeParameterConfig, ChoiceParameterConfig, StorageConfig

storage = StorageConfig(url="sqlite:///ax_experiments.db")
client = Client(storage_config=storage, random_seed=42)

client.configure_experiment(
    name="hpo_nas_v1",  # Required for SQLite persistence
    parameters=[
        RangeParameterConfig(name="lr", bounds=(1e-5, 1e-1), parameter_type="float", scaling="log"),
        ChoiceParameterConfig(
            name="arch",
            parameter_type="str",
            values=["cnn", "vit"],
            is_ordered=False,
            dependent_parameters={"cnn": ["kernel_size"], "vit": ["num_heads"]},
        ),
        ChoiceParameterConfig(name="kernel_size", parameter_type="int", values=[3, 5, 7], is_ordered=True),
        ChoiceParameterConfig(name="num_heads", parameter_type="int", values=[2, 4, 8], is_ordered=True),
    ],
)
client.configure_optimization(
    objective="val_acc, -latency_ms",
    outcome_constraints=["val_acc >= 0.70", "latency_ms <= 50.0", "vram_gb <= 24.0"],
)
client.configure_generation_strategy(method="fast", initialization_budget=8)

# Ask-tell loop (with optional per-call fixed_parameters):
trials = client.get_next_trials(max_trials=2, fixed_parameters={"arch": "vit"})
for trial_idx, params in trials.items():
    # `params` contains ONLY active parameters for the chosen branch (e.g., num_heads, NOT kernel_size)
    metrics = evaluate(params)  # {"val_acc": (0.88, 0.01), "latency_ms": 22.5, "vram_gb": 11.2}
    client.complete_trial(trial_index=trial_idx, raw_data=metrics)

# Retrieve Pareto frontier & resume later from SQLite:
frontier = client.get_pareto_frontier(use_model_predictions=True)  # [(params, pred_dict, idx, arm), ...]
resumed = Client.load_from_database(experiment_name="hpo_nas_v1", storage_config=storage)
```

---

## 3. Full Capability Decision Tree

| Capability / Problem | Recommended 1.3 Approach | Chunk Tag |
| :--- | :--- | :--- |
| **Persist & resume state** | `StorageConfig(url="sqlite:///...")` (auto-saves every call; needs `sqlalchemy`) or `save_to_json_file`. | `§3` |
| **Continuous / integer / discrete** | `RangeParameterConfig` (`scaling="log"`, `step_size`) & `ChoiceParameterConfig` (`is_ordered`, `log_scale`). | `§4` |
| **Conditional / NAS parameters** | `ChoiceParameterConfig(..., dependent_parameters={val: [child_names]})` (strict tree, no cycles). | `§4` |
| **Derived / equality constraints** | `DerivedParameterConfig(expression_str=...)` + linear inequality `parameter_constraints`. | `§4` |
| **Fixed parameters / features** | Always fixed → 1-value `ChoiceParameterConfig`. Per-call → `get_next_trials(fixed_parameters=...)`. Mid-run → `disable_parameters`. | `§5` |
| **Single / scalarized / MOO** | `objective="acc"`, `"-loss"`, `"2*acc - lat"`, or `"acc, -lat"` (+ MOO thresholds & constraints). | `§6` |
| **Relative constraints** | `attach_baseline(params)` + `complete_trial(...)` + `outcome_constraints=["qps >= 0.95 * baseline"]`. | `§6` |
| **BO surrogate selection** | <=20 dims: `method="fast"`. High-dim noisy: `method="quality"` (SAASBO). High-dim + fast + sparse: BONSAI + `EnsembleMapSaasSingleTaskGP`. | `§7` |
| **Execution paradigm** | Custom worker/training loop → Ask-Tell (`get_next_trials`/`complete_trial`). Ax-managed async → `IRunner`/`IMetric` + `run_trials` or `Orchestrator`. | `§8`, `§10` |
| **Warm-start from prior data** | Loop `attach_trial` + `complete_trial` (auto-counted toward init budget). | `§8` |
| **Stop bad runs / convergence** | Trial-level: `attach_data(..., progression=step)` + `should_stop_trial_early`. Global: `ImprovementGlobalStoppingStrategy`. | `§9` |
| **Custom BoTorch / non-BO** | `SurrogateSpec` + `ModelConfig` (with CV model selection) in `GenerationNode`, or `ExternalGenerationNode`. | `§11` |
| **Diagnostics & plotting** | `client.predict`, `client.summarize`, `client.compute_analyses(analyses=[...], display=False)`. | `§12` |

---

## 4. Lookup Index → `ax_reference_guide.md`

Grab chunks using `<!-- AX_REF:START §N -->` to `<!-- AX_REF:END §N -->`:

| Tag | Section Title | Key Contents |
| :--- | :--- | :--- |
| **`§1`** | **Legacy / Deprecated API Ban & Migration** | Full 0.x → 1.3 symbol mapping (`AxClient`, `optimize`, `Scheduler`, `GenerationStep`, `ObjectiveProperties`, etc.). |
| **`§2`** | **Core Types, Configs & `Client` Signatures** | Exact signatures for `Client`, all `*Config` dataclasses, `TOutcome`/`TParameterization`, and power-user setters. |
| **`§3`** | **Persistence: SQLite & JSON** | `StorageConfig`, `sqlalchemy` extra, auto-save lifecycle, `load_from_database`, non-persisted state, `RegistryBundle`, JSON. |
| **`§4`** | **Search Spaces & Dependent Params (NAS)** | Range/Choice/Derived configs, linear `parameter_constraints`, equality workaround, multi-level hierarchical tree rules. |
| **`§5`** | **Fixed Parameters & Fixed Features** | 1-value `FixedParameter`, `get_next_trials(fixed_parameters=...)`, `GeneratorSpec.fixed_features`, `disable_parameters`/`add_parameters`. |
| **`§6`** | **Objectives, MOO, Constraints & Tracking** | SymPy syntax, MOO objective thresholds vs. outcome constraints, relative `* baseline`, tracking metrics, result extraction. |
| **`§7`** | **BO, Generation Strategies & BONSAI** | `configure_generation_strategy`, `"fast"`/`"quality"`/`"random_search"`, init budget & `DataRequiredError`, BONSAI + MAP-SAAS. |
| **`§8`** | **Ask-Tell, Parallelism & Trial Lifecycle** | Async loops, warm-starting from DataFrame, `(mean, sem)` noise semantics, `FAILED`/`ABANDONED`/`EARLY_STOPPED`, `BatchTrial` rule. |
| **`§9`** | **Early Stopping & Global Stopping** | `Percentile`, `Threshold`, `StabilityGated`, `And`/`Or` composition, progression curves, `ImprovementGlobalStoppingStrategy`. |
| **`§10`** | **Closed-Loop Orchestration** | `IRunner`, `IMetric`, `Client.run_trials`, direct `Orchestrator` & full `OrchestratorOptions` reference. |
| **`§11`** | **Custom Models, Nodes & External Gen** | `Generators` enum, `GenerationNode`, `TransitionCriterion`/`PausingCriterion`, `SurrogateSpec`/`ModelConfig` CV selection, custom BoTorch & `ExternalGenerationNode`. |
| **`§12`** | **Predictions, Summaries & Analyses** | `predict`, `summarize`, `compute_analyses`, full Plotly/Healthcheck/Tabular catalog, custom `Analysis` & `PlotlyAnalysisCard`. |

---

## 5. Top 7 Code-Verified Gotchas

1. **SQLite needs `sqlalchemy`, `name=`, and table creation:** `sqlalchemy` is optional (`pip install sqlalchemy`). Every DB-backed experiment must set a unique `name` in `configure_experiment`. On a new SQLite file (or when switching DB URLs in the same Python process, because `SESSION_FACTORY` is a process-global singleton), call `init_engine_and_session_factory(url=url, force_init=True)` and `create_all_tables(get_engine())` from `ax.storage.sqa_store.db` before `Client(...)` (`§3`), otherwise SQLite raises `no such table: experiment_v2`.
2. **Early stopping is NOT saved to SQLite:** After `Client.load_from_database(...)`, always re-call `client.set_early_stopping_strategy(...)`.
3. **Hierarchical branches omit inactive params:** Generated `params` contain only active branch parameters; passing inactive child params to `attach_trial` / `attach_baseline` / `predict` raises a validation error.
4. **`DataRequiredError` during initialization:** Default `initialization_budget=5`, `min_observed_initialization_trials=max(1, budget // 2)`. Requesting more trials than `initialization_budget` before enough trials complete raises `DataRequiredError` unless `allow_exceeding_initialization_budget=True`.
5. **SEM / Noise semantics:** `{"m": 0.9}` → `sem=NaN` (GP infers noise). `{"m": (0.9, 0.0)}` → noiseless. `get_best_parameterization()` / `get_pareto_frontier()` return `(mean, sem)`, **not** variance.
6. **Constraint restrictions:** `parameter_constraints` are linear inequalities only—forbidden on `scaling="log"`, `is_ordered=False` choices, and `DerivedParameterConfig`. In single-objective mode, outcome constraints cannot reference the objective metric.
7. **`method="quality"` (SAASBO) extra:** Requires `pip install "ax-platform[fully_bayesian]"` (JAX/NumPyro). For fast high-dim BO without JAX/NumPyro, use BONSAI + `EnsembleMapSaasSingleTaskGP` (`§7`).