# `positronixlab/nas` — Open Issues Analysis

> **Status (2026-11-27):** **0 open `xfail`s** (`tests/nas/`; 3 sandbox-only `parallelazy` stub errors when `../parallelazy` is outside workspace).
> **Solved issues archive:** Resolved #1 (Ax 1.3 `failed_reason=`), #3 (`warm_start_from` DB restore), #4 (solo OOM retry), #7 (`export_best` stale YAML), #8 (`TrialConfig` `device=None`), #9 (`extract_metrics`), lazy `__init__.py`, and C6/C7 moved to [solved/nas_solved.md](file:///home/cesare/Projects/PositronixLab/reports/solved/nas_solved.md).

---

## 1. Open Bugs & Defects

### High Severity

1. **Unbounded retry loop in `Experiment.run()` on persistent trial failures (#2)**
   - **Location:** `positronixlab/nas/experiment.py:445-465`
   - **Details:** `terminal_count` excludes `FAILED` trials and there is no `max_consecutive_failures` circuit breaker, so any deterministic failure loops forever.

### Medium Severity

2. **`_current_concurrency()` init-budget mismatch (#5):** `experiment.py:196-198` vs `:390` — resolved `max(5, parallel_init)` is not stored on `self.initialization_budget`, prematurely dropping concurrency when `parallel_init > 5`.
3. **`TrialPipeline.execute()` loses per-plugin errors on timeout or `wait()` failure (#6):** `trial_pipeline.py:161-179` — `Pipeline.wait()` calls `shutdown()` before raising, so `get_errors()` returns `{}`.

### Low Severity & Robustness

4. **`eval_fn` returning `TrialResult` leaves `duration_sec=0.0` (#10):** `experiment.py:346-347`.
5. **Validation & side-effect gaps (#11):** Unvalidated `reconcile_running`, silent shadowing of `eval_fn` and `PipelineOptions.poll_interval`, and `get_best()` creating empty `trials/trial_NNN/` dirs (`experiment.py:114-117, 142, 339, 530`).
6. **v0.1.3 E5:** Top-level `ax.*` imports without `[nas]` extra hint (`experiment.py:29-31`).

---

## 2. Open Design Notes

- Direct access to Ax's private `client._experiment` in 4 places; `warm_start_from()` is not idempotent (duplicates trials if called twice) and doesn't sort progression steps; `TrialResult` lacks `.load()`.
- Ax internal SQLAlchemy decoder emits deprecation warnings during tests.
