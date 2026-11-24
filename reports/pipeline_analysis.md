# `positronixlab/pipeline` — Module Analysis

## 1. Overview & Refactor Status

`positronixlab/pipeline` provides the configuration foundation and generic multiprocess pipeline orchestration layer for `positronixlab`. It is currently **mid-refactor** from a legacy monolithic `PipelineConfig` (originally `TrialConfig` / `TrialManager`, renamed in `beffb47` and `a5b9b30`) into a clean, layered architecture:

| File | Role | Status |
|---|---|---|
| [`__init__.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/pipeline/__init__.py) | Eagerly exports `BaseConfig`; lazily exports `Pipeline`, `PipelineConfig`, `PipelineOptions`, `PipelineConfigV1`, `PipelineConfigV2` via PEP 562 `__getattr__`. | Active |
| [`config.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/pipeline/config.py) | `BaseConfig`: framework-independent dataclass base with deterministic MD5 `signature()`, `to_dict()`, `to_yaml()`, immutable `with_updates()`, JSON/YAML `save()`/`load()`, and forgiving `from_dict()`. | Active (new foundation) |
| [`pipeline_config.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/pipeline/pipeline_config.py) | `PipelineOptions(ManagerOptions)` (runtime execution/polling/timeout controls) and `PipelineConfig(BaseConfig)` (`name`, `seed`, `device`). | Active (coupled to `parallelazy`) |
| [`pipeline.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/pipeline/pipeline.py) | `Pipeline(ABC)`: generic `parallelazy.PluginManager` wrapper with lifecycle management (`run`, `wait`, `shutdown`, `close`, context manager), error collection, monitoring, and shared environment access. | Active |
| [`pipeline_config_v1.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/pipeline/pipeline_config_v1.py) | `PipelineConfigV1(PipelineConfig)`: flat monolithic config for `ModularSpace` (V1). | Compatibility / Transitional |
| [`pipeline_config_v2.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/pipeline/pipeline_config_v2.py) | `PipelineConfigV2(PipelineConfig)`: flat monolithic config for `LayerSpace` / `SpaceStep` (V2). | Compatibility / Transitional |
| `legacy/pipeline_config.py` | Archived pre-refactor config (do not touch). | Legacy |

### Composable Mixin Adoption
Downstream modules have already begun adopting `BaseConfig` as composable mixins:
- `positronixlab.dataprovider.config.DataConfig(BaseConfig)`
- `positronixlab.deeplearning.config.TrainConfig(BaseConfig)`
- `positronixlab.metamodel.config.MetaModelConfig(BaseConfig)`
- `positronixlab.modularspace.config.LayerSpaceConfig(BaseConfig)`
- `positronixlab.nas.trial_config.TrialConfig(PipelineConfig)` (and `TrialPipeline(Pipeline)`)

Meanwhile, older modules (`ModularSpace`, `ModularObserver`, `MetaCriterion`, `DataProviderFactory`, `DeeplearningBase`/`Train`/`Eval`, `tasks/compositionality`) still type-hint `PipelineConfig` (importing from `positronixlab.pipeline`).

### `parallelazy` Dependency Status
- In `.venv`, `parallelazy` has been updated from `0.0.6` (mentioned in `AGENTS.md`) to editable `0.2.1` pointing to `/home/cesare/Projects/parallelazy`.
- `pipeline.py:9` imports `from parallelazy import PluginManager, is_debugpy_attached`, and `pipeline_config.py:6` imports `from parallelazy.plugin_data import ManagerOptions`.
- In environments where `parallelazy` is not installed, is pinned to `0.0.6` (missing `is_debugpy_attached`), or whose editable target is inaccessible (e.g., restricted sandboxes), importing **not only** `Pipeline` but **also** `PipelineConfig`, `PipelineConfigV1`, `PipelineConfigV2`, and `TrialConfig` fails with `ModuleNotFoundError` / `ImportError`.

---

## 2. Bugs & Defects

### B1. `PipelineConfig` is transitively coupled to `parallelazy` (High)
- **Location:** [`pipeline_config.py:6`](file:///home/cesare/Projects/PositronixLab/positronixlab/pipeline/pipeline_config.py#L6)
- **Root Cause:** `from parallelazy.plugin_data import ManagerOptions` sits at the top of `pipeline_config.py`, which defines both `PipelineOptions(ManagerOptions)` and `PipelineConfig(BaseConfig)`.
- **Impact:** `PipelineConfig` itself is a pure dataclass (`BaseConfig` + `seed` + `device`) with zero runtime dependency on `parallelazy`. Placing `PipelineOptions` in the same module defeats the lazy-export isolation in [`__init__.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/pipeline/__init__.py#L3-L17): importing `PipelineConfig`, `PipelineConfigV1`, `PipelineConfigV2`, or `TrialConfig` immediately pulls in `parallelazy` (and `torch.multiprocessing`).

### B2. `BaseConfig.from_dict()` skips type coercion for `Optional[T]` on subclasses without `from __future__ import annotations` and for PEP 604 unions (Medium)
- **Location:** [`config.py:57-83`](file:///home/cesare/Projects/PositronixLab/positronixlab/pipeline/config.py#L57-L83)
- **Root Cause:** `valid_fields = {f.name: f.type for f in fields(cls)}` checks `expected_type in (str, "str", "Optional[str]")` (and similarly for `bool`, `int`, `float`).
  - When a subclass omits `from __future__ import annotations` (as **both** `pipeline_config_v1.py` and `pipeline_config_v2.py` do), `Optional[str]` evaluates to the runtime `typing.Union[str, None]` object, which matches neither `str`, `"str"`, nor `"Optional[str]"`. Consequently, `PipelineConfigV2.modular_space_reducer` bypasses coercion completely.
  - When a subclass uses PEP 604 unions (`int | None`, `str | None`), the stringified or runtime type also fails to match `"Optional[int]"`.
  - Additionally, `except (ValueError, TypeError): filtered[k] = v` (`config.py:82-83`) silently stores unparseable strings (e.g. `int("3.5")` or `float("bad")`) into typed numeric fields instead of raising a clear error, and `int(True)` silently coerces booleans into integers.
- **Impact:** Silent type corruption when deserializing YAML/JSON or Ax parameter dicts into configs.

### B3. `Pipeline.wait()` logs error spam on every poll when `terminate_on_error=False` and checks `is_error()` before `is_complete()` (Medium)
- **Location:** [`pipeline.py:134-150`](file:///home/cesare/Projects/PositronixLab/positronixlab/pipeline/pipeline.py#L134-L150)
- **Root Cause:**
  1. When `terminate_on_error=False` and a plugin enters `"error"`, `wait()` logs `logger.error(...)` on every single loop iteration (4×/sec at the default `poll_interval=0.25`) with no deduplication.
  2. `is_error()` (`line 134`) is evaluated before `is_complete()` (`line 143`). If a pipeline completes right as a secondary/teardown plugin errors, `wait()` shuts down and raises `RuntimeError` instead of returning cleanly.
  3. `time.sleep(interval)` (`line 150`) does not clamp the sleep duration to the remaining timeout (`timeout - (time.monotonic() - t0)`), so `wait()` can overshoot `timeout_sec` by up to `poll_interval`.
- **Impact:** Log flooding during non-fatal error runs, race-condition false failures at completion, and imprecise timeouts.

### B4. `Pipeline.is_error()` misses abrupt worker crashes (SIGKILL / segfault / OOM kill) (Medium)
- **Location:** [`pipeline.py:77-83`](file:///home/cesare/Projects/PositronixLab/positronixlab/pipeline/pipeline.py#L77-L83)
- **Root Cause:** `is_error()` only checks `status.worker_status.state == "error"`. If a spawned worker dies abruptly without updating `worker_status` (e.g., Linux OOM killer or segfault), `status.process_alive` becomes `False` while `worker_status` is `None` or still `"running"`.
- **Impact:** `is_error()` returns `False`. If `timeout_sec=None` and `is_complete()` waits on an environment flag rather than checking `all_plugins_terminated()`, `wait()` hangs indefinitely.

### B5. `Pipeline.get_errors()` does not guard against worker-exit race in `plugin_errors()` (Low)
- **Location:** [`pipeline.py:99-100`](file:///home/cesare/Projects/PositronixLab/positronixlab/pipeline/pipeline.py#L99-L100)
- **Root Cause:** Between `plugin_status_all()` (`line 94`) and `self.plugin_manager.plugin_errors(plugin_id)` (`line 100`), a worker reported as `process_alive=True` may exit or close its IPC pipe, causing `plugin_errors()` to raise an unhandled exception inside `wait()`.
- **Impact:** Unexpected crash during error collection instead of returning partial/available errors.

### B6. `Pipeline.__init__` initializes `self._closed` after `PluginManager` setup; `shutdown()` swallows exceptions silently (Low)
- **Location:** [`pipeline.py:34-38`](file:///home/cesare/Projects/PositronixLab/positronixlab/pipeline/pipeline.py#L34-L38), [`pipeline.py:173-186`](file:///home/cesare/Projects/PositronixLab/positronixlab/pipeline/pipeline.py#L173-L186)
- **Root Cause:**
  - `self._closed: bool = False` is set on line 38 *after* `PluginManager(...)` and `env_set(...)` (lines 34–36). If lines 35–36 raise, any cleanup calling `shutdown()`/`close()` hits `AttributeError: ... '_closed'`.
  - `shutdown()` catches `except Exception: pass` (`lines 178-179`) with no logging, hiding worker stop failures.
  - `close()` sets `self._closed = True` before calling `self.plugin_manager.close()` (`lines 185-186`); if `plugin_manager.close()` fails (e.g. `KeyboardInterrupt`), retries are permanently skipped.

### B7. `BaseConfig.load()` crashes on non-dict YAML/JSON roots; `save()` has case-sensitive `.json` check and lacks `default=str` (Low)
- **Location:** [`config.py:38-53`](file:///home/cesare/Projects/PositronixLab/positronixlab/pipeline/config.py#L38-L53)
- **Root Cause:**
  - `load()` passes `data or {}` directly to `cls.from_dict()` without checking `isinstance(data, dict)`. A valid YAML/JSON list or scalar raises `AttributeError: '...' object has no attribute 'items'`.
  - `save()` checks `filepath.endswith(".json")` (case-sensitive; `.JSON` writes YAML) and omits `default=str` in `json.dump()` (unlike `signature()` at line 23 which uses `default=str`), causing `TypeError` for `Path`/`Enum` fields.

### B8. `_log_status_snapshot()` logs plugin errors at `INFO` level (Low)
- **Location:** [`pipeline.py:159-171`](file:///home/cesare/Projects/PositronixLab/positronixlab/pipeline/pipeline.py#L159-L171)
- **Root Cause:** Even when `has_problem = True` (`ERRORS:` or `ALIVE (status unknown)`), line 171 logs via `logger.info(msg)` rather than `logger.warning`/`logger.error`.
- **Impact:** Runs configured at `WARNING` log level silently miss periodic error snapshots.

---

## 3. Inconsistencies & Design Issues

1. **Duplicate epoch fields in V1/V2 (`epochs` vs `train_epochs`)**: Both [`pipeline_config_v1.py:19,49`](file:///home/cesare/Projects/PositronixLab/positronixlab/pipeline/pipeline_config_v1.py#L19) and [`pipeline_config_v2.py:22,52`](file:///home/cesare/Projects/PositronixLab/positronixlab/pipeline/pipeline_config_v2.py#L22) define both `train_epochs: int = 10` and `epochs: int = 10` with identical defaults.
2. **Stale docstring in `PipelineConfigV1`**: [`pipeline_config_v1.py:7-8`](file:///home/cesare/Projects/PositronixLab/positronixlab/pipeline/pipeline_config_v1.py#L7-L8) claims *"Can be initialized directly with both params and fixed dictionaries"*, which is false — `from_trial(params, fixed)` was moved to `positronixlab.nas.trial_config.TrialConfig`.
3. **Inconsistent sentinel defaults**: `input_reduce_every: int = 0` vs `output_reduce_every: int = -1` in both V1 (`lines 29, 38`) and V2 (`lines 32, 41`).
4. **`PipelineOptions` does not share `BaseConfig` serialization API**: [`pipeline_config.py:12-33`](file:///home/cesare/Projects/PositronixLab/positronixlab/pipeline/pipeline_config.py#L12-L33) re-implements `with_updates()` (without a generic `TypeVar` return type) and lacks `to_dict()`, `from_dict()`, `save()`, `load()`, and `signature()`.
5. **`__init__.py` lazy exports are not cached & lack `TYPE_CHECKING` stubs**: [`__init__.py:12-17`](file:///home/cesare/Projects/PositronixLab/positronixlab/pipeline/__init__.py#L12-L17) re-runs `__getattr__` on every attribute access instead of caching in `globals()`, and static type checkers cannot resolve `from positronixlab.pipeline import PipelineConfig` without an `if TYPE_CHECKING:` block.
6. **Legacy caller drift**: `positronixlab/nas/legacy/trial_runner.py:5` still imports `TrialManager` from `positronixlab.pipeline`, which no longer exists.

---

## 4. Opportunities & Recommendations

1. **Split `PipelineOptions` out of `pipeline_config.py`** (e.g. into `pipeline_options.py` or `pipeline.py`) so `PipelineConfig`, `PipelineConfigV1`, `PipelineConfigV2`, and `TrialConfig` have **zero** runtime dependency on `parallelazy`.
2. **Use `typing.get_type_hints(cls)` (or unwrap `get_origin`/`get_args`) in `BaseConfig.from_dict()`** so `Optional[T]`, `T | None`, and stringified annotations coerce consistently regardless of `from __future__ import annotations`. Consider validating or warning on coercion failure rather than silently storing raw strings in numeric fields, and validating `isinstance(data, dict)` in `load()`.
3. **Harden `Pipeline.wait()`**:
   - Check `is_complete()` before `is_error()` (or make error precedence explicit).
   - Deduplicate `logger.error(...)` messages when `terminate_on_error=False`.
   - Clamp `time.sleep(min(interval, remaining))` when `timeout is not None`.
   - Log `has_problem` snapshots at `logger.warning` in `_log_status_snapshot()`.
   - Guard `plugin_errors(plugin_id)` with `try/except` in `get_errors()`.
4. **Initialize `self._closed = False` before `PluginManager(...)`** in `Pipeline.__init__`, and log (at `debug`/`warning`) any exception caught in `shutdown()`.
5. **Deprecate `PipelineConfigV1` / `PipelineConfigV2`** once `ModularSpace`, `ModularObserver`, and `MetaCriterion` accept composable `BaseConfig` mixins (`MetaModelConfig`, `LayerSpaceConfig`, `TrainConfig`, `DataConfig`), and add `from __future__ import annotations` to both in the meantime.
6. **Add `if TYPE_CHECKING:` imports and cache `globals()[name]`** in `positronixlab/pipeline/__init__.py`.

---

## 5. Docs / Tests / Examples Summary

- **Documentation (`positronixlab/pipeline/docs/`)**:
  - Created [`PIPELINE_SYSTEM.md`](file:///home/cesare/Projects/PositronixLab/positronixlab/pipeline/docs/PIPELINE_SYSTEM.md) — architecture, module layout, class hierarchy, API reference, and `parallelazy` integration notes.
  - Created [`PIPELINE_USAGE.md`](file:///home/cesare/Projects/PositronixLab/positronixlab/pipeline/docs/PIPELINE_USAGE.md) — practical recipes for `BaseConfig` composition, serialization, `PipelineOptions`, subclassing `Pipeline`, and migrating from `PipelineConfigV1`/`V2`.
- **Tests (`positronixlab/pipeline/tests/`)**:
  - Created [`test_pipeline_config.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/pipeline/tests/test_pipeline_config.py) — unit tests for `BaseConfig`, composable mixins, serialization/hashing, `from_dict()` coercion (including `xfail` tests documenting known bugs B2/B7), `PipelineOptions`, `PipelineConfig`, `PipelineConfigV1`, and `PipelineConfigV2`.
  - Created [`test_pipeline.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/pipeline/tests/test_pipeline.py) — unit tests for `Pipeline` ABC lifecycle, monitoring, error collection, `wait()` loop semantics (completion, error, non-fatal error, timeout, status snapshots), idempotent `shutdown()`/`close()`, and context manager using a lightweight fake `PluginManager`.
- **Examples (`positronixlab/pipeline/examples/`)**:
  - Created [`example_pipeline_config.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/pipeline/examples/example_pipeline_config.py) — runnable demonstration of `BaseConfig` mixin composition, `from_dict()` coercion, immutable `with_updates()`, deterministic `signature()`, and YAML/JSON round-trip persistence.
