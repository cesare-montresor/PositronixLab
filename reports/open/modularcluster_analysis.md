# `positronixlab/modularcluster` — Open Issues Analysis

> **Status (2026-11-27 — `v0.1.3` / `5740a0c` + working tree):** **29 passed, 3 xfailed, 0 failed** (`tests/modularcluster/`).
> **Solved issues archive:** Resolved B1–B5/B7 (`ModularClusterKmeans`), A6a–d, and cleanup moved to [solved/modularcluster_solved.md](file:///home/cesare/Projects/PositronixLab/reports/solved/modularcluster_solved.md).

---

## 1. Open Bugs & Defects

### Medium Severity (Stream Fix Drift & Warm-Start Bounds)

1. **B5 & B7 fix drift in `ModularClusterKmeansStream`**
   - **Location:** `modularcluster_kmeans_stream.py:77-81, 99-110, 265-268`
   - **Details:** Both the `k_min > k_max` small-`n` guard (A6a/B5) and the `< 2` unique-labels silhouette guard (A6b/B7) were fixed in `ModularClusterKmeans` only — `ModularClusterKmeansStream` remains vulnerable to both crashes. Also `# gpu=True` is still commented out at `modularcluster_kmeans_stream.py:124`.

2. **`ModularClusterKmeansStream.fit()` warm-start ignores `self.k_min` / `self.k_max` (B6)**
   - **Location:** `modularcluster_kmeans_stream.py:257-259` (xfail `test_streaming_kmeans.py:245`)
   - **Details:** Clamps the warm-start search window to `[2, n - 1]` instead of `[self.k_min, self.k_max]`.

3. **`StreamingClusterTracker._relabel_to_match_previous()` mismatched return annotation & stale test**
   - **Location:** `streaming_cluster_tracker.py:123-125`; `tests/modularcluster/test_streaming_kmeans.py:190-237`
   - **Details:** `_relabel_to_match_previous()` still annotates `-> torch.Tensor` despite returning `Tuple[torch.Tensor, torch.Tensor]`. As a result, `test_b3_tracker_reorders_centroids_on_relabel` fails on tuple comparison instead of verifying the B3 fix. Similarly, the two B2 xfail tests (`:190, :203`) fail due to FAISS K=3 cluster splitting on their random seeds rather than diagonal corruption.

### Low–Medium Severity

4. **`StreamingClusterTracker` debounce publishes stale assignments, leaks `_pending_k`, and rejects `torch.Tensor` (B8)**
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
