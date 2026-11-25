# NAS Usage Guide

Practical guide to running Hyperparameter Optimization (HPO) and Neural Architecture Search (NAS) experiments with `positronixlab.nas`.

## 1. Compose a Typed `TrialConfig`

Combine `TrialConfig` with any domain config mixins your experiment needs. Ax parameter names must match dataclass field names; undeclared keys are silently ignored by `from_dict()`.

```python
from dataclasses import dataclass
from positronixlab.dataprovider import DataConfig
from positronixlab.deeplearning import TrainConfig
from positronixlab.metamodel import MetaModelConfig
from positronixlab.modularspace import LayerSpaceConfig
from positronixlab.nas import TrialConfig


@dataclass
class AtlasTrialConfig(MetaModelConfig, LayerSpaceConfig, TrainConfig, DataConfig, TrialConfig):
    exp_metric_modularity: bool = True
```

## 2. Define the Search Space with `TemplateBase`

Use `TemplateBase` to separate searchable parameters (`add_range`, `add_choice`, `add_derived`, `add_constraint`) from non-searchable constants (`add_fixed`).

```python
from positronixlab.template import TemplateBase

tpl = TemplateBase("mnist_nas")
tpl.add_choice(
    "model_type",
    ["fc", "cnn"],
    "str",
    is_ordered=False,
    dependent_parameters={"cnn": ["cnn_kernel_size"]},
)
tpl.add_choice("cnn_kernel_size", [3, 5], "int", is_ordered=True)
tpl.add_range("learning_rate", [1e-4, 1e-2], "float", scaling="log")
tpl.add_range("modular_space_size", [4, 12], "int")
tpl.add_fixed("dataset_name", "mnist")
tpl.add_fixed("train_epochs", 50)
```

## 3. Quick Start with `eval_fn` (Lightweight / In-Process)

For unit tests, dry runs, or simple in-process training functions, pass `eval_fn`:

```python
from positronixlab.nas import Experiment, TrialResult


def evaluate(cfg: AtlasTrialConfig) -> TrialResult:
    # Train / evaluate using cfg.learning_rate, cfg.model_type, etc.
    val_acc = 0.85
    best_epoch = 32
    return TrialResult(
        status="completed",
        metrics={"val_acc": val_acc},          # raw metric (never pre-negated)
        metadata={"best_epoch": best_epoch},   # stored in result.json
        effective_overrides={"train_epochs": best_epoch},  # saved to effective_config.yaml
    )


exp = Experiment(
    name="mnist_hpo",
    template=tpl,
    objective="val_acc",
    eval_fn=evaluate,
    config_cls=AtlasTrialConfig,
    output_dir="output",
    initialization_budget=4,
    random_seed=42,
)
best = exp.run(n_trials=12)
best_cfg, predictions, trial_idx = best[0]
```

> `eval_fn` may also return a plain `Dict[str, MetricValue]` (e.g. `{"val_acc": 0.85}` or `{"val_acc": (0.85, 0.01)}`), which `Experiment` wraps in a completed `TrialResult`.

## 4. Production Runs with `TrialPipeline` (`parallelazy`)

Subclass `TrialPipeline` to launch multi-process `parallelazy` plugins for each trial. `TrialPipeline` automatically binds `PluginManager.work_dir` to `config.trial_dir` so plugin logs land under `trials/trial_NNN/logs/`.

```python
from typing import Any, Dict
from positronixlab.nas import Experiment, MetricValue, TrialPipeline
from positronixlab.pipeline import PipelineOptions


class AtlasTrialPipeline(TrialPipeline):
    def setup(self) -> None:
        # Load plugins into self.plugin_manager and seed shared env
        self.plugin_manager.env_set("metrics", {"val_acc": 0.0})

    def is_complete(self) -> bool:
        return bool(self.get_env("training_done"))

    def extract_metrics(self, env: Dict[str, Any]) -> Dict[str, MetricValue]:
        return {"val_acc": float(env["val_acc"])}

    def extract_effective_overrides(self, env: Dict[str, Any]) -> Dict[str, Any]:
        return {"train_epochs": int(env["best_epoch"])}


exp = Experiment(
    name="atlas_nas",
    template=tpl,
    objective="val_acc",
    trial_pipeline_cls=AtlasTrialPipeline,
    config_cls=AtlasTrialConfig,
    pipeline_options=PipelineOptions(display_logs=False, write_logs=True),
    output_dir="output",
    initialization_budget=6,
    parallel_init=2,
    parallel_bo=1,
    gpu_ids=[0, 1],
    trial_timeout_sec=3600.0,
)
exp.run(n_trials=30)
```

## 5. Multi-Objective Optimization (MOO) & Constraints

Use comma-separated Ax 1.3 expressions in `objective` (prefix `-` to minimize) and raw metric names in `outcome_constraints`:

```python
exp = Experiment(
    name="mnist_moo",
    template=tpl,
    objective="val_acc, -model_size",
    outcome_constraints=["val_acc >= 0.70", "model_size <= 500000"],
    tracking_metrics=["train_loss"],
    eval_fn=evaluate_moo,
    config_cls=AtlasTrialConfig,
    parallel_init=2,
    parallel_bo=2,
)
pareto_frontier = exp.run(n_trials=20)
for cfg, pred, idx in pareto_frontier:
    print(f"Trial {idx}: {pred}")
```

## 6. Resuming & Warm-Starting

- **Resume an interrupted experiment**: Instantiate `Experiment` with the same `name` and `output_dir`, then call `run(n_trials=...)` with the target total trial count. Ax loads state from `<output_dir>/<name>/ax.db`.
- **Warm-start from a prior experiment**: Transfer completed trials from an existing SQLite DB before calling `run()`:

```python
new_exp = Experiment(name="phase2", template=tpl, objective="val_acc", eval_fn=evaluate, config_cls=AtlasTrialConfig)
attached = new_exp.warm_start_from(
    prior_db_path="output/phase1/ax.db",
    prior_experiment_name="phase1",
    filter_fn=lambda params: params.get("model_type") == "cnn",
)
new_exp.run(n_trials=attached + 10)
```

## 7. Parallelism & OOM Handling

- `parallel_init` controls batch concurrency during Sobol initialization; `parallel_bo` controls concurrency during Bayesian optimization.
- `gpu_ids=[0, 1]` caps parallel concurrency to `len(gpu_ids)` and assigns `cfg.gpu_id` (and `cfg.device = f"cuda:{gpu_id}"` if `device` is unset).
- If a trial raises a CUDA OOM error (matched by `is_oom_error()`), it is deferred out of the parallel batch and retried solo (`concurrency=1`) with `torch.cuda.empty_cache()` and `oom_backoff_sec` sleep between retries. If retries are exhausted, the trial is marked `ABANDONED` (`oom_final_status="abandoned"`).

## 8. Running the Example

```bash
.venv/bin/python examples/nas/example_nas_trial.py
```
