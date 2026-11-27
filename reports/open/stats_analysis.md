# `positronixlab/stats` — Open Issues Analysis

> **Status (2026-11-27 — `v0.1.3` / `5740a0c` + working tree):** **23 passed, 14 xfailed, 0 failed** (`tests/stats/`; requires `CUPY_CACHE_DIR=/tmp/cupy_cache` in read-only-home sandboxes when `cuml`/`cupy` is installed).
> **Solved issues archive:** Resolved import-time `cuml` crash (#1) & `__init__.py` population moved to [solved/stats_solved.md](file:///home/cesare/Projects/PositronixLab/reports/solved/stats_solved.md).

---

## 1. Open Bugs & Defects

### High Severity

1. **`WindowReducer.drain()` crashes on tensor shape change & `maxlen=0` (#2)**
   - **Location:** `positronixlab/stats/reducer.py:105-115` (xfails `test_stats.py:329, 340`)

3. **`OverallProcessor.process()` crashes on empty input & mismatched shapes/devices (#3)**
   - **Location:** `positronixlab/stats/processor.py:200-205` (xfail `test_stats.py:523`)

4. **`UmapProcessor.process()` crashes on CUDA tensors & post-fit neuron count changes (#4)**
   - **Location:** `positronixlab/stats/processor.py:102, 106-114` (xfail `test_stats.py:485`)

5. **`UmapProcessor._create_mapper()` sets `n_neighbors >= num_neurons` for small layers (#5)**
   - **Location:** `positronixlab/stats/processor.py:82-96` (xfail `test_stats.py:475`)

### Medium Severity

6. **`_as_tensor()` crashes on list/tuple of non-scalar tensors (#6):** `common.py:16-17` (xfail `test_stats.py:152`).
7. **`DisplacementFilter` lacks `method` validation & crashes on empty tensors; `DisplacementReducer` crashes on empty tensors (#7):** `filter.py:9-11, 25-26`, `reducer.py:152-153` (xfails `test_stats.py:226, 234, 380`).
8. **`Stream.ingest()` counts empty generators & gated-out `DisplacementReducer` updates (#8):** `stream.py:49-55` (xfails `test_stats.py:642, 661`).
9. **`Stream.merge_streams()` crashes when `reducer_args`/`processor_args` is `None` (#9):** `stream.py:88-89` (xfail `test_stats.py:685`).

### Low Severity

10. **`UmapProcessor.last_index()` ambiguous between uninitialized and index `0` (#10):** `processor.py:158-160` (xfail `test_stats.py:466`).
11. **`UmapProcessor.centroids_2d()` crashes on empty `pts`/`labels` (#11):** `processor.py:179-180` (xfail `test_stats.py:497`).
12. **`DisplacementFilter.check_and_update()` retains autograd graphs via `.clone()` without `.detach()` (#12):** `filter.py:19, 33`.

---

## 2. Open Design & Hygiene Issues

- **Duplication:** `DisplacementFilter` (`filter.py`) is orphaned and duplicates `DisplacementReducer` (`reducer.py`).
- **Inconsistent locking:** `MeanReducer`/`DisplacementReducer`/`UmapProcessor` hold internal locks while `Last`/`Ema`/`Window`/`OverallProcessor`/`DisplacementFilter` do not. `OverallProcessor.process()` stores references without `.clone()`.
- **Exports & hygiene (v0.1.3 F):** `stats/__init__.py` eagerly imports `processor.py` (pulling in `cuml`/`cupy` if installed) and omits `REDUCERS`/`PROCESSORS`; empty non-abstract `_after_drain` in ABC (`B027`, `reducer.py:25`); unused `logger` in `processor.py:11` and `stream.py:16`.
