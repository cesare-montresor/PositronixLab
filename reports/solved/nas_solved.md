# `positronixlab/nas` — Solved Issues Archive

> **Archived:** 2026-11-27 (`v0.1.3` / `5740a0c` + working tree).
> **Active open issues:** See [nas_analysis.md](file:///home/cesare/Projects/PositronixLab/reports/nas_analysis.md).

---

## Resolved Bugs & Improvements

| ID | Summary | Resolution Evidence |
| :--- | :--- | :--- |
| **#1 (Critical)** | Wrong `reason=` kwarg in all 4 `Client.mark_trial_failed()` calls + `_reconcile_interrupted_trials` trapped inside `try/except` | All 4 calls updated to `failed_reason=` (`experiment.py:221, 375, 380, 383`); `_reconcile_interrupted_trials(loaded_client)` moved outside `try/except` (`experiment.py:179-181`). Regression tests at `tests/nas/test_nas.py:601-676` pass without xfail. |
| **Exports** | Eager `__init__.py` imports | Upgraded `nas/__init__.py:1-31` to PEP 562 lazy exports with `if TYPE_CHECKING:` and `globals()` caching. |
| **v0.1.3 C6 & C7 (2026-11-27)** | Silent `except Exception: pass` (`experiment.py:50`) & runtime `assert`s (`experiment.py:342, 423`) | Replaced with `logger.debug(..., exc_info=True)` and explicit `RuntimeError` guards. |
| **#3 (P1, 2026-11-27)** | `warm_start_from()` leaked global `SESSION_FACTORY` on error | Wrapped prior DB read in `try/finally: _ensure_sqlite_db(self._db_url)` (`experiment.py:259-293`); xfail removed and passing. |
| **#4 (P1, 2026-11-27)** | Solo OOM retry off-by-one (`2 + oom_retries`) & missing initial backoff | Added `already_solo` flag to `_retry_oom_solo()` (`experiment.py:399-434, 468-503`); xfail removed and passing. |
| **#7 (P1, 2026-11-27)** | `export_best()` left stale `best/trial_*.yaml` files | Clears old `trial_*.yaml` files before exporting new bests (`experiment.py:568-571`); xfail removed and passing. |
| **#8 (P1, 2026-11-27)** | `TrialConfig.from_trial()` failed to set `device` from `gpu_id` when `fixed` had `device=None` | Changed `setdefault` to `if not merged.get("device"):` (`trial_config.py:58-59`); xfail removed and passing. |
| **#9 (P1, 2026-11-27)** | `TrialPipeline.extract_metrics()` converted `bool` to `float`, crashed on `(mean, None)`, dropped numpy/torch scalars | Skips `bool`, handles `(mean, None)` → `float(mean)`, and unwraps `.item()` scalars (`trial_pipeline.py:112-130`); xfail removed. |
