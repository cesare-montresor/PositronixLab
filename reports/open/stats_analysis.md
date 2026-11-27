# `positronixlab/stats` — Open Issues Analysis

> **Status (2026-11-27):** **37 passed, 0 xfailed, 0 failed** (`tests/stats/`; requires `CUPY_CACHE_DIR=/tmp/cupy_cache` in read-only-home sandboxes when `cuml`/`cupy` is installed).
> **Solved issues archive:** Resolved import-time `cuml` crash (#1), all 14 P1 xfail bugs (#2–#11), UMAP `None` guard, and `__init__.py` population moved to [solved/stats_solved.md](file:///home/cesare/Projects/PositronixLab/reports/solved/stats_solved.md).

---

## 1. Open Bugs & Defects

### Low Severity

1. **`DisplacementFilter.check_and_update()` retains autograd graphs via `.clone()` without `.detach()` (#12):** `filter.py:19, 33`.
2. **`OverallProcessor.process()` mismatched shapes/devices across keys (#3 remaining):** `processor.py:222` — assumes all keys share identical shape and device.
3. **`UmapProcessor.process()` CUDA tensors (#4 remaining):** `processor.py:114` — calls `t.numpy()` without `.detach().cpu()`.

---

## 2. Open Design & Hygiene Issues

- **Duplication:** `DisplacementFilter` (`filter.py`) is orphaned and duplicates `DisplacementReducer` (`reducer.py`).
- **Inconsistent locking:** `MeanReducer`/`DisplacementReducer`/`UmapProcessor` hold internal locks while `Last`/`Ema`/`Window`/`OverallProcessor`/`DisplacementFilter` do not. `OverallProcessor.process()` stores references without `.clone()`.
- **Exports & hygiene (v0.1.3 F):** `stats/__init__.py` eagerly imports `processor.py` (pulling in `cuml`/`cupy` if installed) and omits `REDUCERS`/`PROCESSORS`; empty non-abstract `_after_drain` in ABC (`B027`, `reducer.py:25`); unused `logger` in `processor.py:11` and `stream.py:16`.
