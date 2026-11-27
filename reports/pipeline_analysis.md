# `positronixlab/pipeline` — Open Issues Analysis

> **Status (2026-11-27 — `v0.1.3` / `5740a0c` + working tree):** **29 passed, 3 xfailed, 0 failed** (`tests/pipeline/`).
> **Solved issues archive:** Resolved B5, v0.1.3 A9d/C6, `get_errors()` empty-list regression, and lazy-export caching moved to [solved/pipeline_solved.md](file:///home/cesare/Projects/PositronixLab/reports/solved/pipeline_solved.md).

---

## 1. Open Bugs & Defects

### High / Medium Severity

2. **`PipelineConfig` is transitively coupled to `parallelazy` (B1)**
   - **Location:** `positronixlab/pipeline/pipeline_config.py:6`
   - **Details:** `from parallelazy.plugin_data import ManagerOptions` sits at top level alongside pure dataclass `PipelineConfig`, defeating lazy-export isolation for `PipelineConfig`, `PipelineConfigV1`, `PipelineConfigV2`, and `TrialConfig`.

3. **`BaseConfig.from_dict()` skips `Optional[T]` & PEP 604 union coercion (B2)**
   - **Location:** `positronixlab/pipeline/config.py:56-84` (xfails `test_pipeline_config.py:184, 267`)
   - **Details:** Fails on `int | None` and on `Optional[T]` in subclasses omitting `from __future__ import annotations` (`pipeline_config_v1.py`, `pipeline_config_v2.py`); silently swallows `ValueError`/`TypeError` on invalid numeric strings.

4. **`Pipeline.wait()` error spam, completion-vs-error ordering, and unclamped sleep (B3)**
   - **Location:** `positronixlab/pipeline/pipeline.py:142-158`
   - **Details:** Logs `logger.error` every poll when `terminate_on_error=False`, checks `is_error()` before `is_complete()`, and does not clamp `time.sleep(interval)` to remaining timeout.

5. **`Pipeline.is_error()` misses abrupt worker crashes (SIGKILL / OOM) (B4)**
   - **Location:** `positronixlab/pipeline/pipeline.py:77-83`
   - **Details:** Only checks `worker_status.state == "error"`, missing workers whose process died abruptly while `worker_status` is `None` or `"running"`.

### Low Severity

6. **`Pipeline.__init__` `_closed` init order & silent exception swallowing in `shutdown()`/`close()` (B6 & v0.1.3 C6):** `pipeline.py:34-38, 105-106, 184-194`.
7. **`BaseConfig.load()` crashes on non-dict YAML/JSON root; `save()` case-sensitive `.json` & `yaml.dump` vs `safe_load` tuple mismatch (B7):** `config.py:30, 41-53` (xfail `test_pipeline_config.py:197`).
8. **`_log_status_snapshot()` logs errors at `INFO` level (B8):** `pipeline.py:178-179`.

---

## 2. Open Design Issues

- Duplicate `epochs` vs `train_epochs` and inconsistent `input_reduce_every=0` vs `output_reduce_every=-1` in `PipelineConfigV1`/`V2`; stale `PipelineConfigV1` docstring.
- `PipelineOptions` does not share `BaseConfig` serialization API.
- `pipeline/__init__.py` lazy exports are not cached in `globals()` and lack an `if TYPE_CHECKING:` block.
