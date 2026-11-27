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
