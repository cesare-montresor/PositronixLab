# `positronixlab/pipeline` — Solved Issues Archive

> **Archived:** 2026-11-27 (`v0.1.3` / `5740a0c` + working tree).
> **Active open issues:** See [pipeline_analysis.md](file:///home/cesare/Projects/PositronixLab/reports/pipeline_analysis.md).

---

## Resolved Bugs

| ID | Summary | Resolution Evidence |
| :--- | :--- | :--- |
| **B5** | `Pipeline.get_errors()` did not guard against worker-exit race in `plugin_errors()` | Wrapped `plugin_errors(plugin_id)` in `try/except Exception` (`pipeline.py:100-106`). *(Note: introduced empty-list fallback regression tracked in active `pipeline_analysis.md`.)* |
| **v0.1.3 A9d** | `BaseConfig.signature()` MD5 failed under FIPS | Passes `usedforsecurity=False` to `hashlib.md5` (`config.py:24`). |
| **Regression (2026-11-27)** | `Pipeline.get_errors()` skipped `plugin_errors(plugin_id)` fallback when `worker_status.errors == []` | Fixed `pipeline.py:98` (`if isinstance(raw_errs, (list, tuple)) and raw_errs:`); `test_pipeline_is_error_and_get_errors` passes. |
| **v0.1.3 C6 / `S110` (2026-11-27)** | Silent `except Exception: pass` in `get_errors()` and `shutdown()` | Replaced with `logger.debug(..., exc_info=True)` (`pipeline.py:106, 187`). |
| **Cleanup (2026-11-27)** | `pipeline/__init__.py` did not cache lazy exports in `globals()` | Added `globals()[name] = value` caching (`pipeline/__init__.py:16`). |
| **B2 (`b414936`)** | `BaseConfig.from_dict()` skipped `Optional[T]` & PEP 604 `T \| None` union coercion and tuple coercion | Upgraded `pipeline/config.py` with `get_type_hints` / `Optional` unwrapping and tuple coercion; both B2 xfails removed and passing. |
| **B7 (`b414936`)** | `BaseConfig.load()` crashed on non-dict YAML/JSON root; `save()` used `yaml.dump` vs `safe_load` & case-sensitive `.json` | Switched to `yaml.safe_dump` + tuple/dict validation in `pipeline/config.py`; B7 xfail removed and passing. |
