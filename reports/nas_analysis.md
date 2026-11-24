# `positronixlab/nas` Module Analysis

## 1. Overview & Refactor Status

The `positronixlab/nas` module was recently refactored from the legacy 2-phase `NasExperiment` runner (archived in `positronixlab/nas/legacy/`) into a clean, lightweight, SQLite-backed Ax 1.3+ experiment runner split across three active files:

| File | Responsibility |
| :--- | :--- |
| `positronixlab/nas/__init__.py` | Public exports: `Experiment`, `MetricValue`, `TrialConfig`, `TrialPipeline`, `TrialResult`, `is_oom_error`. |
| `positronixlab/nas/trial_config.py` | `TrialConfig(PipelineConfig)` — flat, Ax-aware dataclass for composing domain config mixins (`MetaModelConfig`, `LayerSpaceConfig`, `TrainConfig`, `DataConfig`) via `TrialConfig.from_trial(...)`. |
| `positronixlab/nas/trial_pipeline.py` | `TrialPipeline(Pipeline)`, `TrialResult`, `MetricValue`, `is_oom_error` — standardized `parallelazy`-backed trial execution lifecycle (`setup -> run -> wait -> extract -> shutdown`) with `work_dir` binding and OOM/timeout/error classification. |
| `positronixlab/nas/experiment.py` | `Experiment[TConfig]` — Ax 1.3 `Client` wrapper with SQLite persistence/resumption, parallel initialization & BO via `ThreadPoolExecutor`, solo OOM retry with backoff, cross-experiment `warm_start_from()`, and artifact export (`template.yaml`, `experiment.yaml`, `summary.csv`, `best/`). |

**Refactor status:** The core happy-path architecture is clean, modern (strictly Ax 1.3 `Client` API, no deprecated `AxClient`), and well-structured. All 6 existing tests in `tests/test_nas.py` pass when `parallelazy` is importable. However, **all trial-failure paths (`mark_trial_failed`) are currently broken** due to an Ax 1.3 keyword argument mismatch (`reason=` vs `failed_reason=`) that was never exercised by the existing test suite.

*(Note: Repo-level `AGENTS.md` still references the pre-refactor `NasExperiment` 2-phase architecture and `output/<exp>/databases/` layout.)*

---

## 2. Bugs & Defects

### High Severity

1. **Wrong keyword argument `reason=` in all `Client.mark_trial_failed()` calls**
   - **Location:** `positronixlab/nas/experiment.py:215`, `experiment.py:369`, `experiment.py:374`, `experiment.py:377`
   - **Root cause:** Ax 1.3 `Client.mark_trial_failed(self, trial_index: int, failed_reason: str | None = None)` expects `failed_reason=`, not `reason=`.
   - **Impact:** Raises `TypeError: Client.mark_trial_failed() got an unexpected keyword argument 'reason'` on **every** non-OOM trial error, trial timeout, OOM with `oom_final_status="failed"`, and crash-recovery reconciliation with default `reconcile_running="fail"`.
   - **Compounding defect (`experiment.py:165-175`):** In `_init_or_resume_client()`, `self._reconcile_interrupted_trials(client)` is called *inside* the `try:` block intended to catch missing/uninitialized DBs. When `mark_trial_failed(..., reason=...)` raises `TypeError`, `_init_or_resume_client` catches it, logs `"Could not load existing DB ...; creating new experiment."`, and attempts to recreate an experiment with the same name in the same SQLite DB (which fails with a duplicate-experiment error).

2. **Potential infinite loop in `Experiment.run()` on persistent trial failures**
   - **Location:** `positronixlab/nas/experiment.py:433-449`
   - **Root cause:** `terminal_count` only counts `COMPLETED`, `ABANDONED`, and `EARLY_STOPPED` (excluding `FAILED`), and there is no `max_failures` / consecutive-failure circuit breaker.
   - **Impact:** Once Bug #1 is fixed, any deterministic failure in `eval_fn` or `TrialPipeline` (e.g., bad config, missing data, broken plugin) causes `Experiment.run(n_trials=N)` to loop indefinitely, spawning unbounded `FAILED` trials in SQLite and on disk.

3. **`warm_start_from()` leaks Ax's process-global SQLAlchemy `SESSION_FACTORY` on error**
   - **Location:** `positronixlab/nas/experiment.py:254-286`
   - **Root cause:** `_ensure_sqlite_db(prior_url)` mutates Ax's global `SESSION_FACTORY` singleton, and `_ensure_sqlite_db(self._db_url)` at line 286 is not wrapped in `try ... finally`.
   - **Impact:** If `Client.load_from_database(...)` or extraction fails (e.g., wrong `prior_experiment_name`), the global session factory remains bound to `prior_url`, silently directing subsequent DB operations on `self.client` to the wrong SQLite file.

### Medium Severity

4. **Solo OOM retry off-by-one and missing initial backoff**
   - **Location:** `positronixlab/nas/experiment.py:391-418`, `experiment.py:453-461`
   - **Root cause:** When `len(trials_batch) == 1`, `run()` executes attempt 1, then appends to `oom_deferred` if `oom_retries > 0`, and `_retry_oom_solo()` executes `total_attempts = 1 + self.oom_retries` *additional* attempts (`2 + oom_retries` total instead of `1 + oom_retries`), with no `oom_backoff_sec` sleep before the first retry.
   - **Impact:** Extra wasted OOM attempt on solo runs and immediate retry without GPU cooldown.

5. **`_current_concurrency()` init-budget mismatch when `initialization_budget=None` and `parallel_init > 1`**
   - **Location:** `positronixlab/nas/experiment.py:190-192` vs `experiment.py:384`
   - **Root cause:** `_init_or_resume_client` passes `init_budget = max(5, self.parallel_init)` to Ax without storing the resolved value on `self.initialization_budget`, while `_current_concurrency` hardcodes `5` when `self.initialization_budget is None`.
   - **Impact:** With e.g. `parallel_init=8, parallel_bo=1, initialization_budget=None`, concurrency prematurely drops from 8 to 1 after 5 completed trials even though Ax's Sobol initialization budget is 8.

6. **`TrialPipeline.execute()` loses per-plugin errors on timeout or `wait()` failure**
   - **Location:** `positronixlab/nas/trial_pipeline.py:161-179` (interacting with `positronixlab/pipeline/pipeline.py:134-149`)
   - **Root cause:** `Pipeline.wait()` calls `self.shutdown()` (closing `plugin_manager` and setting `self._closed = True`) *before* raising `TimeoutError` or `RuntimeError`. `TrialPipeline.execute()` then calls `self.get_errors()` in `except TimeoutError` (without checking `self._closed`) and `self.get_errors() if not self._closed else {}` in `except Exception`, so `errors` is always empty on timeout and always falls back to `{"pipeline": [msg]}` on plugin error.
   - **Impact:** Per-plugin error attribution (`{plugin_id: [...]}`) is lost in `result.json`.

7. **`Experiment.export_best()` leaves stale `best/trial_*.yaml` files**
   - **Location:** `positronixlab/nas/experiment.py:551-566`
   - **Root cause:** `export_best()` writes new `best/trial_{t_idx:03d}.yaml` files without clearing previous `trial_*.yaml` exports.
   - **Impact:** After resuming an experiment or updating the Pareto frontier, `best/` contains stale YAML configs no longer listed in `best_summary.json`.

8. **`TrialConfig.from_trial()` fails to set `device` from `gpu_id` if `fixed` contains `device=None`**
   - **Location:** `positronixlab/nas/trial_config.py:55-57`
   - **Root cause:** `merged.setdefault("device", f"cuda:{gpu_id}")` does not overwrite an existing `"device"` key whose value is `None` (or an inherited default like `"cuda:0"` when distributing across multiple `gpu_ids`).
   - **Impact:** `cfg.gpu_id` and `cfg.device` can disagree (`gpu_id=1`, `device=None` or `"cuda:0"`).

### Low Severity

9. **`TrialPipeline.extract_metrics()` converts booleans to floats, crashes on `(mean, None)`, and drops numpy/torch scalars**
   - **Location:** `positronixlab/nas/trial_pipeline.py:112-116`
   - **Root cause:** `isinstance(True, int)` is `True` in Python; `float(v[1])` raises `TypeError` if SEM is `None`; `np.float32` / 0-d `torch.Tensor` are not instances of `(int, float)` and are silently skipped.

10. **`eval_fn` returning `TrialResult` leaves `duration_sec=0.0`**
    - **Location:** `positronixlab/nas/experiment.py:340-341`
    - **Root cause:** When `eval_fn` returns a `TrialResult` instance directly without setting `duration_sec`, `_run_single_trial` returns it verbatim instead of filling in `time.monotonic() - t0`.

11. **Missing input validation & silent shadowing in `Experiment.__init__`**
    - **Location:** `positronixlab/nas/experiment.py:107, 110-138, 331-334`
    - **Details:**
      - `reconcile_running` is not validated (e.g., typo `"abandoned"` silently falls through to `"fail"`).
      - Passing both `trial_pipeline_cls` and `eval_fn` silently ignores `eval_fn`.
      - `self.poll_interval` (default `0.25`) is unconditionally passed to `pipeline.execute(...)`, silently overriding `PipelineOptions.poll_interval`.
      - `Experiment.get_best()` calls `self._trial_dir(t_idx)` (`experiment.py:524`), creating empty `trials/trial_NNN/` directories as a side effect for warm-started trials.

---

## 3. Inconsistencies & Design Issues

- **Documentation placement:** `ax_best_practices.md` and `ax_reference_guide.md` live in the module root `positronixlab/nas/` instead of `positronixlab/nas/docs/` (unlike `dataprovider/docs/`, `logger/docs/`, `modularcluster/docs/`, `stats/docs/`, `template/docs/`). They are kept in place per review constraints, but should be moved into `docs/` (updating internal cross-references) once convenient.
- **Ax 1.3 API & deprecation noise:**
  - `experiment.py` directly accesses Ax's private `client._experiment` in 4 places (`lines 206, 261, 382, 436`). While pragmatic for trial status inspection, it couples tightly to Ax internals.
  - Running the test suite produces 341 warnings (mostly Ax's internal SQLAlchemy decoder emitting `Objective`/`MultiObjective`/`ObjectiveThreshold` deprecation warnings and `AxParameterWarning: sort_values` on string choices).
  - `Experiment.is_multi_objective` checks `"," in self.objective` (`experiment.py:156`), which works for standard Ax MOO expressions (`"acc, -size"`) but would misclassify a future function call expression containing a comma.
- **OOM retry asymmetry:** For `len(trials_batch) == 1`, `oom_retries=0` skips `_retry_oom_solo`; for `len(trials_batch) > 1`, `oom_retries=0` still defers to `_retry_oom_solo` for 1 solo attempt (`experiment.py:458` vs `475`).
- **`warm_start_from()` progression & idempotency:** Iterating `data_df[data_df["trial_index"] == idx]` overwrites `raw_data[metric_name]` without sorting by progression step, and calling `warm_start_from()` twice on the same experiment attaches duplicate trials.
- **Process-global SQLAlchemy singleton:** Creating two `Experiment` instances with different SQLite paths in the same process causes the second instance's `_ensure_sqlite_db()` to silently rebind the global `SESSION_FACTORY` away from the first instance.
- **Missing `TrialResult.load()`:** `BaseConfig` and `TemplateBase` provide `.save()` / `.load()` round-trips, whereas `TrialResult` only provides `.save()`.

---

## 4. Opportunities & Recommendations

1. **Fix `mark_trial_failed` immediately:** Replace `reason=` with `failed_reason=` at `experiment.py:215, 369, 374, 377`, and move `_reconcile_interrupted_trials(client)` *outside* the `Client.load_from_database` `try/except` block.
2. **Add a failure circuit breaker to `Experiment.run()`:** Add `max_consecutive_failures: int = 10` (or `max_failures`) so persistent trial errors abort cleanly instead of looping forever.
3. **Wrap `warm_start_from()` DB switch in `try ... finally`:** Guarantee `_ensure_sqlite_db(self._db_url)` always restores the global session factory; optionally deduplicate against already-attached arms.
4. **Fix solo OOM retry count & backoff:** Avoid running `1 + oom_retries` extra attempts when the first attempt was already solo, and apply `oom_backoff_sec` before retrying.
5. **Resolve `initialization_budget` once in `__init__`:** Store the resolved integer on `self._resolved_init_budget` so Ax, `experiment.yaml`, and `_current_concurrency()` agree.
6. **Harden `TrialPipeline` error capture & `extract_metrics()`:** Capture `self.get_errors()` before `shutdown()` in `Pipeline.wait()` (or store last errors on `Pipeline`), reject `bool` in `extract_metrics`, handle `(mean, None)` tuples, and support numpy/torch scalar values.
7. **Clean stale files in `export_best()` and avoid side-effect `makedirs` in `get_best()`.**
8. **Respect `PipelineOptions.poll_interval`:** Default `Experiment(..., poll_interval: Optional[float] = None)` so `PipelineOptions.poll_interval` is only overridden when explicitly passed.
9. **Move `ax_best_practices.md` and `ax_reference_guide.md` into `positronixlab/nas/docs/`** and update `AGENTS.md` to reflect the new `Experiment` / `TrialConfig` / `TrialPipeline` API.

---

## 5. Docs / Tests / Examples Summary

- **Docs created:**
  - `positronixlab/nas/docs/NAS_SYSTEM.md` — architecture, folder layout, lifecycle, SQLite & OOM semantics, and full API reference.
  - `positronixlab/nas/docs/NAS_USAGE.md` — practical guide covering `TrialConfig` composition, `TemplateBase` search spaces, single-objective/MOO `Experiment` usage, `TrialPipeline` integration, resumption, and warm-starting.
- **Tests expanded (`positronixlab/nas/tests/test_nas.py`):**
  - Preserved all 6 existing integration tests.
  - Added unit tests covering `TrialConfig.from_trial`, `is_oom_error`, `TrialResult`, `TrialPipeline.extract_metrics` & `execute` (timeout/error/OOM/`work_dir`), `Experiment` validation, `reconcile_running="abandon"`, `warm_start_from` with `filter_fn`, `fixed_parameters`, and `oom_retries=0`.
  - Added `@pytest.mark.xfail(strict=True)` regression tests documenting the confirmed bugs (`mark_trial_failed` `reason=` kwarg across error/timeout/OOM-failed/reconcile-fail, solo OOM off-by-one, `TrialConfig` `device=None` with `gpu_id`, `extract_metrics` bool/None-SEM handling, stale `best/` exports, and `warm_start_from` global DB leak).
- **Example added (`positronixlab/nas/examples/example_nas_trial.py`):**
  - Self-contained runnable script demonstrating a composed `TrialConfig`, hierarchical `TemplateBase` search space, and a fast mock `Experiment` run inspecting generated artifacts.
