# `positronixlab/modularcluster` — Open Issues Analysis

> **Status (2026-11-27):** **32 passed, 0 xfailed, 0 failed** (`tests/modularcluster/`).
> **Solved issues archive:** Resolved B1–B7 (`ModularClusterKmeans` + warm-start B6), A6a–d, and cleanup moved to [solved/modularcluster_solved.md](file:///home/cesare/Projects/PositronixLab/reports/solved/modularcluster_solved.md).

---

## 1. Open Bugs & Defects

### Medium Severity (Stream Fix Drift)

1. **B5 & B7 fix drift in `ModularClusterKmeansStream`**
   - **Location:** `modularcluster_kmeans_stream.py:77-81, 99-110, 265-268`
   - **Details:** Both the `k_min > k_max` small-`n` guard (A6a/B5) and the `< 2` unique-labels silhouette guard (A6b/B7) were fixed in `ModularClusterKmeans` only — `ModularClusterKmeansStream` remains vulnerable to both crashes. Also `# gpu=True` is still commented out at `modularcluster_kmeans_stream.py:124`.

### Low–Medium Severity

2. **`StreamingClusterTracker` debounce publishes stale assignments, leaks `_pending_k`, and rejects `torch.Tensor` (B8)**
   - **Location:** `streaming_cluster_tracker.py:113-117, 182-183, 222-229`
   - **Details:** When debounce holds `k`, `update()` republishes previous frame's `last_stats` instead of current clustering at `last_published_k`; `_pending_k` is not reset upon acceptance; `update()` assumes `np.ndarray` and fails on `torch.Tensor`.

5. **Unused `tol`/`EPS` & exception-unsafe `_effective_max_iter` (B9 remaining)**
   - **Location:** `modularcluster_kmeans_stream.py:251, 272`
   - **Details:** `_effective_max_iter` is reset outside `try/finally` (and `max_iter_override or self.max_iter` treats `0` as falsy); `tol` and `EPS` are unused across all three classes.

---

## 2. Open Design, Performance & Hygiene Issues

- **Code duplication (~250 lines):** `ModularClusterKmeansStream` is a copy-paste fork of `ModularClusterKmeans` (root cause of the B5/B7 fix drift); consolidate warm-start support into `ModularClusterKmeans`.
- **O(n²) CPU silhouette cache (v0.1.3 D3) & extra `assign()` call (D4):** `modularcluster_kmeans.py:92-105, 129, 233-235` — consider optional `sample_size` for large `n`.
- **`faiss-cpu` hard dependency (v0.1.3 E3):** `pyproject.toml:42` has no optional `gpu` extra.
- **Hygiene (v0.1.3 F):** 10 `print()` calls in `modularcluster_kmeans.py` and 11 in `modularcluster_kmeans_stream.py`; stale `positronixlab/modularcluster/README.md` still references pre-split `PluginModularCluster`.
