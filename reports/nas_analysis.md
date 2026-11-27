# `positronixlab/nas` — Open Issues Analysis

> **Status (2026-11-27 — `v0.1.3` / `5740a0c` + working tree):** **15 passed, 5 xfailed, 2 failed** *(both failures are sandbox-only `parallelazy` stub errors when `../parallelazy` is outside workspace)*.
> **Solved issues archive:** Critical Ax 1.3 `failed_reason=` crash (#1) & lazy `__init__.py` moved to [solved/nas_solved.md](file:///home/cesare/Projects/PositronixLab/reports/solved/nas_solved.md).

---

## 1. Open Bugs & Defects

### High Severity

1. **Unbounded retry loop in `Experiment.run()` on persistent trial failures (#2)**
   - **Location:** `positronixlab/nas/experiment.py:439-455`
   - **Details:** `terminal_count` excludes `FAILED` trials and there is no `max_consecutive_failures` circuit breaker, so any deterministic failure loops forever.

2. **`warm_start_from()` leaks Ax's global SQLAlchemy `SESSION_FACTORY` on error (#3)**
   - **Location:** `positronixlab/nas/experiment.py:260-292` (xfail `test_nas.py:761`)
   - **Details:** `_ensure_sqlite_db(self._db_url)` is not wrapped in `try/finally`, leaving subsequent DB calls bound to `prior_url` if loading fails.

### Medium Severity

3. **Solo OOM retry off-by-one & missing initial backoff (#4):** `experiment.py:406-421, 459-467` (xfail `test_nas.py:679`) — executes `2 + oom_retries` attempts on solo trials without initial cooldown sleep.
4. **`_current_concurrency()` init-budget mismatch (#5):** `experiment.py:196-198` vs `:390` — resolved `max(5, parallel_init)` is not stored on `self.initialization_budget`, prematurely dropping concurrency when `parallel_init > 5`.
5. **`TrialPipeline.execute()` loses per-plugin errors on timeout or `wait()` failure (#6):** `trial_pipeline.py:161-179` — `Pipeline.wait()` calls `shutdown()` before raising, so `get_errors()` returns `{}`.
6. **`Experiment.export_best()` leaves stale `best/trial_*.yaml` files (#7):** `experiment.py:557-571` (xfail `test_nas.py:734`).
7. **`TrialConfig.from_trial()` fails to set `device` from `gpu_id` when `fixed` has `device=None` (#8):** `trial_config.py:55-57` (xfail `test_nas.py:711`).

### Low Severity & Robustness

8. **`TrialPipeline.extract_metrics()` converts `bool` to `float`, crashes on `(mean, None)`, drops numpy/torch scalars (#9):** `trial_pipeline.py:112-117` (xfail `test_nas.py:720`).
9. **`eval_fn` returning `TrialResult` leaves `duration_sec=0.0` (#10):** `experiment.py:346-347`.
10. **Validation & side-effect gaps (#11):** Unvalidated `reconcile_running`, silent shadowing of `eval_fn` and `PipelineOptions.poll_interval`, and `get_best()` creating empty `trials/trial_NNN/` dirs (`experiment.py:114-117, 142, 339, 530`).
11. **v0.1.3 review items:** `except Exception: pass` (`C6`, `experiment.py:50-51`), runtime `assert` (`C7`, `experiment.py:342, 423`), and top-level `ax.*` imports without `[nas]` extra hint (`E5`, `experiment.py:29-31`).

---

## 2. Open Design Notes

- Direct access to Ax's private `client._experiment` in 4 places; `warm_start_from()` is not idempotent (duplicates trials if called twice) and doesn't sort progression steps; `TrialResult` lacks `.load()`.
- Ax internal SQLAlchemy decoder emits deprecation warnings during tests.
