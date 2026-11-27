# `positronixlab/modularcluster` — Solved Issues Archive

> **Archived:** 2026-11-27 (`v0.1.3` / `5740a0c` + working tree).
> **Active open issues:** See [modularcluster_analysis.md](file:///home/cesare/Projects/PositronixLab/reports/modularcluster_analysis.md).

---

## Resolved Bugs & Cleanup

| ID | Summary | Resolution Evidence |
| :--- | :--- | :--- |
| **B1** | `ModularClusterKmeans._kmeans()` returned `float32` labels (breaking `bincount_cpu`) | Fixed to `torch.as_tensor(labels, dtype=torch.long, device=self.device)` (`modularcluster_kmeans.py:130`); `test_b1_*` passes. |
| **B2** | `_distance()` unconditionally called `fill_diagonal_(0.0)` on `(n, k)` matrices | Guarded with `if x is y: dist.fill_diagonal_(0.0)` in both `modularcluster_kmeans.py:87-88` and `modularcluster_kmeans_stream.py:94-95`. *(Note: the 2 B2 xfail tests still fail due to FAISS K=3 cluster splitting on test seeds, not diagonal corruption.)* |
| **B3** | `StreamingClusterTracker._relabel_to_match_previous()` remapped labels without reordering centroids | Now returns `(centroids[inv_remap_t], remap_t[labels])` (`streaming_cluster_tracker.py:138, 148-152, 231-242`). *(Return type annotation & test unpacking still need cleanup.)* |
| **B4 / v0.1.3 A6c** | CUDA `device` crashed FAISS `train()`/`assign()` and `gpu=True` failed with `faiss-cpu` | Both classes convert inputs to contiguous CPU `float32` NumPy arrays; `modularcluster_kmeans.py:115` checks `hasattr(faiss, 'StandardGpuResources')`. |
| **B5 / v0.1.3 A6a (`ModularClusterKmeans`)** | `_compute_k_range(n)` returned `(2, 1)` for `n in {2, 3}` leaving `best_k=None` | Added 1-cluster fallback when `k_min > k_max` (`modularcluster_kmeans.py:225-231`); stale xfail removed. *(Still open in `ModularClusterKmeansStream`.)* |
| **B7 / v0.1.3 A6b (`ModularClusterKmeans`)** | Degenerate `< 2` unique labels crashed `_silhouette_score()` | Added `if len(np.unique(labels_np)) < 2: return -1.0` (`modularcluster_kmeans.py:96-98`). *(Still open in `ModularClusterKmeansStream`.)* |
| **v0.1.3 A6d** | Unused `dists` variable from `kmeans.assign(x_np)` | Changed to `_, labels = kmeans.assign(x_np)` (`modularcluster_kmeans.py:129`). |
| **B9 (imports)** | Unused imports (`time`, `override`, `os`, `contextlib`) | Removed from both KMeans modules. |
| **Cleanup** | Empty `__init__.py` & stale `tests/test_kmeans_plugin.py` | `__init__.py` populated (`__init__.py:1-9`); stale `test_kmeans_plugin.py` deleted. |
| **Cleanup (2026-11-27)** | Mismatched return annotation on `_relabel_to_match_previous` & stale B3 xfail test | Fixed return annotation to `-> tuple[torch.Tensor, torch.Tensor]` (`streaming_cluster_tracker.py:125`), updated `test_b3_tracker_reorders_centroids_on_relabel` to unpack tuple and removed stale `xfail`. |
| **B6 (P1, 2026-11-27)** | `ModularClusterKmeansStream.fit()` warm-start ignored `self.k_min` / `self.k_max` | Clamped warm-start `[k_min, k_max]` against `self._compute_k_range(n)` (`modularcluster_kmeans_stream.py:257-260`); xfail removed and passing. |
