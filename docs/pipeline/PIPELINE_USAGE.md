# PositronixLab Pipeline Usage Guide

Practical recipes for building composable configurations with `BaseConfig` and orchestrating `parallelazy`-backed pipelines with `Pipeline` and `PipelineOptions`.

---

## 1. Composing Hyperparameter Configs with `BaseConfig`

Domain modules define flat `@dataclass` mixins inheriting from `BaseConfig`. Compose them via Python multiple inheritance so your concrete config has full IDE autocomplete while remaining a flat dictionary for Ax sweeps and YAML/JSON serialization:

```python
from __future__ import annotations

from dataclasses import dataclass
from positronixlab.pipeline import BaseConfig


@dataclass
class DataConfig(BaseConfig):
    dataset_name: str = "fashion_mnist"
    batch_size: int = 32


@dataclass
class TrainConfig(BaseConfig):
    learning_rate: float = 1e-3
    train_epochs: int = 10
    use_amp: bool = False


@dataclass
class ExperimentConfig(TrainConfig, DataConfig):
    seed: int = 42
```

### Constructing from Dicts (`from_dict`)

`from_dict()` silently drops unknown keys (useful when passing mixed Ax parameters or legacy YAML files) and coerces primitive types:

```python
cfg = ExperimentConfig.from_dict({
    "name": "run_01",
    "learning_rate": "0.002",   # coerced to float
    "batch_size": "64",         # coerced to int
    "use_amp": "true",          # coerced to bool
    "ignored_extra_key": 999,   # dropped silently
})
```

> **Tip:** Always include `from __future__ import annotations` in modules defining `BaseConfig` subclasses so `Optional[T]` annotations are stringified consistently for `from_dict()` coercion.

### Immutable Updates & Deterministic Signatures

Use `with_updates()` to derive modified configs (e.g., post-trial effective overrides) without mutating the original:

```python
sig_before = cfg.signature()
effective = cfg.with_updates(train_epochs=7, unknown_key="ignored")

assert cfg.train_epochs == 10
assert effective.train_epochs == 7
assert cfg.signature() == sig_before
assert effective.signature() != sig_before
```

### Saving and Loading (YAML & JSON)

```python
cfg.save("output/run_01/config.yaml")
loaded_yaml = ExperimentConfig.load("output/run_01/config.yaml")

cfg.save("output/run_01/config.json")
loaded_json = ExperimentConfig.load("output/run_01/config.json")

assert loaded_yaml.signature() == cfg.signature()
assert loaded_json.signature() == cfg.signature()
```

---

## 2. Configuring Runtime Execution with `PipelineOptions`

Keep runtime execution settings separate from model hyperparameters using `PipelineOptions`:

```python
from positronixlab.pipeline import PipelineOptions

options = PipelineOptions(
    work_dir="output/exp_01",
    display_logs=False,
    terminate_on_error=True,
    timeout_sec=300.0,
    poll_interval=0.5,
    status_log_interval=30.0,
    verbose=False,
    parallel_plugins=True,
)

# Extract only the kwargs accepted by parallelazy.PluginManager
manager_kwargs = options.to_manager_kwargs()

# Create an updated copy without mutating the original
debug_options = options.with_updates(verbose=True, timeout_sec=60.0)
```

---

## 3. Implementing a Concrete `Pipeline`

Subclass `Pipeline` and implement `is_complete()`. Load plugins in `__init__` or a `setup()` method, then use the context manager so `shutdown()` is guaranteed on exit or exception:

```python
from typing import Optional
from positronixlab.pipeline import Pipeline, PipelineConfig, PipelineOptions


class MyPipeline(Pipeline):
    def __init__(
        self,
        config: PipelineConfig,
        options: Optional[PipelineOptions] = None,
    ):
        super().__init__(config=config, options=options)
        # Load parallelazy plugins here, e.g.:
        # self.plugin_manager.load_plugins([...], parallel=self.options.parallel_plugins)

    def is_complete(self) -> bool:
        # Check a shared env completion flag or plugin termination
        return bool(self.get_env("done")) or self.all_plugins_terminated()


cfg = PipelineConfig(name="demo", seed=42, device="cuda:0")
opts = PipelineOptions(timeout_sec=120.0, status_log_interval=10.0)

with MyPipeline(cfg, opts) as pipe:
    pipe.enable_monitoring(hotkey="m")
    pipe.run()
    pipe.wait()
    results = pipe.get_env("metrics")
```

### Handling Errors and Timeouts in `wait()`

- **`terminate_on_error=True` (default):** as soon as `is_error()` returns `True`, `wait()` calls `shutdown()` and raises `RuntimeError` containing the collected plugin error messages.
- **`terminate_on_error=False`:** `wait()` logs the error and continues polling until `is_complete()` or `timeout_sec`. (Note: currently logs on every poll iteration — use a larger `poll_interval` if running with `terminate_on_error=False`.)
- **`timeout_sec`:** when elapsed time reaches `timeout_sec`, `wait()` calls `shutdown()` and raises `TimeoutError`. Pass `timeout_sec=None` for unbounded runs.

---

## 4. Choosing Between `PipelineConfigV1`, `PipelineConfigV2`, and Composable Mixins

| Config Class | When to Use |
|---|---|
| `BaseConfig` + domain mixins (`DataConfig`, `TrainConfig`, `MetaModelConfig`, `LayerSpaceConfig`, `TrialConfig`) | **Preferred for new code & NAS experiments.** Modular, zero unused fields, full autocomplete. |
| `PipelineConfigV1` | Legacy `ModularSpace` (V1) runs that expect `modular_space_loss_centers` / `modular_space_loss_alignment`. |
| `PipelineConfigV2` | Flat Ax sweeps targeting `LayerSpace` + composable `SpaceStep` stages (`modular_space_scoring`, `modular_space_contrast`, `modular_space_loss_sigreg`, etc.). |

See [`examples/pipeline/example_pipeline_config.py`](file:///home/cesare/Projects/PositronixLab/examples/pipeline/example_pipeline_config.py) for a runnable end-to-end configuration example.
