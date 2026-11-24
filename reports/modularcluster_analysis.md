# `positronixlab/modularcluster` Module Analysis

## 1. Overview & Refactor Status

`positronixlab/modularcluster` provides FAISS-backed K-Means clustering with adaptive `K` selection (silhouette-score optimization) and streaming cluster tracking for continuously evolving computation-unit (CU) embeddings.

### Active Components

| File | Class | Role | Status |
| :--- | :--- | :--- | :--- |
| `__init__.py` | *(empty)* | Package init (no re-exports) | Minimal |
| `modularcluster_kmeans.py` | `ModularClusterKmeans` | Synchronous adaptive K-Means (Euclidean & cosine/spherical) via `faiss.Kmeans` + 3-phase silhouette search | Active, has bugs (3 failing tests) |
| `modularcluster_kmeans_stream.py` | `ModularClusterKmeansStream` | Fork of `ModularClusterKmeans` adding per-`k` centroid caching, warm-start narrow `K` window, and per-call `max_iter_override` | Active, has bugs (no tests prior to this review) |
| `streaming_cluster_tracker.py` | `StreamingClusterTracker` | Per-layer streaming wrapper over `ModularClusterKmeansStream`: EMA drift gating, periodic full search, Hungarian relabeling, ARI diagnostics, hysteresis + debounce on `K` changes | Active, has bugs (no tests prior to this review) |

### Post-Split Refactor Status
- Concrete plugin wrappers (`PluginModularCluster` / `ModularClusterPlugin`) were moved out of `positronixlab` into `../AtlasCompose` (on top of `../parallelazy`) in commit `4e70649`.
- **Stale artifacts remaining in this module:**
  - `tests/test_kmeans_plugin.py` still imports `positronixlab.modularcluster.modularcluster_plugin` and `positronixlab.plugin`, breaking full-repo pytest collection (`ModuleNotFoundError`).
  - `README.md` (not modified in this review) still documents `PluginModularCluster` and `modularcluster_plugin.py` and omits `ModularClusterKmeansStream` and `StreamingClusterTracker`.
  - `docs/MODULARCLUSTER_{API,SYSTEM,USAGE}.md` were updated in this review to remove pre-split plugin references and document all three active classes.

---

## 2. Bugs & Defects

### B1. `ModularClusterKmeans._kmeans()` returns `float32` labels instead of `int64` (High)
- **Location:** `positronixlab/modularcluster/modularcluster_kmeans.py:128`
- **Code:** `labels = torch.Tensor(labels).to(self.device)`
- **Root cause:** `torch.Tensor(...)` constructs a `torch.float32` tensor from the `int64` NumPy array returned by `faiss.Kmeans.assign(x)`.
- **Impact:**
  - `print_stats()` (`modularcluster_kmeans.py:260`) calls `torch.bincount(self.labels).tolist()`, crashing with `NotImplementedError: "bincount_cpu" not implemented for 'Float'`.
  - Causes 3 failures in `tests/test_kmeans.py` (`test_print_stats`, `test_sync_euclidean`, `test_sync_cosine`).
  - Breaks any downstream integer indexing or `torch.bincount(labels)` calls and contradicts the documented `int64` contract.
- **Fix:** Replace with `labels = torch.from_numpy(labels).long().to(self.device)`. (Note: `ModularClusterKmeansStream` does not have this bug because it assigns labels via `dists.argmin(dim=1)` which returns `torch.int64`.)

### B2. `_distance()` unconditionally calls `dist.fill_diagonal_(0.0)`, corrupting `predict()` and streaming `fit()` (High)
- **Locations:**
  - `positronixlab/modularcluster/modularcluster_kmeans.py:90`
  - `positronixlab/modularcluster/modularcluster_kmeans_stream.py:98`
- **Code:**
  ```python
  dist.fill_diagonal_(0.0) # added for eucledian
  dist.clamp_(min=0.0) # not alwys >=0 due to numerical errors
  ```
- **Root cause:** `_distance(x, y)` is shared between self-distance precomputation (`_distance(x, x)`, shape `(n, n)`) and point-to-centroid distance (`_distance(x, centroids)`, shape `(n, k)`). Zeroing the diagonal on a rectangular `(n, k)` distance matrix sets `dists[i, i] = 0.0` for all `0 <= i < min(n, k)`.
- **Impact:**
  - In `ModularClusterKmeans.predict(x)` (`modularcluster_kmeans.py:243-244`) and `ModularClusterKmeansStream.predict(x)` (`modularcluster_kmeans_stream.py:289-290`), the first `min(n, k)` query points are forced to cluster IDs `[0, 1, ..., min(n, k) - 1]` regardless of their actual coordinates.
  - In `ModularClusterKmeansStream._kmeans()` (`modularcluster_kmeans_stream.py:149-150`), label assignment uses `_distance(x, centroids).argmin(dim=1)`, so **every `fit()` call** corrupts `labels[:k]` to `[0, 1, ..., k - 1]`, distorting silhouette scores, `best_k` selection, and downstream `StreamingClusterTracker` state.
- **Fix:** Only zero the diagonal when computing self-distances (e.g., pass `zero_diagonal: bool = False` and set `zero_diagonal=True` only in `fit()` when computing `self.distance_cache`), or use `kmeans.assign(x)` in `ModularClusterKmeansStream._kmeans()` as `ModularClusterKmeans` does.

### B3. `StreamingClusterTracker._relabel_to_match_previous()` remaps `labels` without reordering `centroids` (High)
- **Location:** `positronixlab/modularcluster/streaming_cluster_tracker.py:123-149, 228-248`
- **Root cause:** `_relabel_to_match_previous()` solves `linear_sum_assignment(cost)` to obtain `remap[col_ind] = row_ind` (`new_id -> old_id`) and returns `remap_t[labels]`, but never reorders `self.kmeans.centroids` to match the remapped label IDs. Line 237 then exports `centroids_np = self.kmeans.centroids.cpu().numpy()` and stores it in `self.last_published_centroids`.
- **Impact:** Whenever the Hungarian assignment is not the identity permutation:
  - `stats["centroids"][c]` does **not** correspond to the cluster whose points have `stats["labels"] == c`.
  - `self.last_published_centroids` stores the un-reordered centroids, causing subsequent frames' Hungarian matching to align against mismatched centroid indices and compounding label drift.
- **Fix:** Return both remapped labels and reordered centroids (`centroids_reordered[row_ind] = centroids[col_ind]`), and store/publish the reordered centroids.

### B4. CUDA `device` crashes both `ModularClusterKmeans` and `ModularClusterKmeansStream` (High)
- **Locations:**
  - `positronixlab/modularcluster/modularcluster_kmeans.py:122-127, 205`
  - `positronixlab/modularcluster/modularcluster_kmeans_stream.py:128-141, 240`
- **Root cause:** Both classes move `x = x.to(self.device)` before calling `kmeans.train(x)` (and `kmeans.assign(x)`). FAISS's Python `Kmeans.train()` internally converts inputs with NumPy (`np.ascontiguousarray`), which raises `TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.` when `x` is a CUDA tensor — even when `gpu=True` is passed to `FAISS_KMeans`. Additionally, `ModularClusterKmeansStream` has `# gpu=True` commented out (`line 135`), so it never enables FAISS GPU mode at all.
- **Impact:** Passing `device=torch.device('cuda')` unconditionally crashes `fit()`.
- **Fix:** Pass `x_np = np.ascontiguousarray(x.detach().cpu().numpy(), dtype=np.float32)` to `kmeans.train(x_np)` and `kmeans.assign(x_np)` (FAISS handles GPU transfer internally when `gpu=True`), and wire `gpu=(self.device.type == 'cuda')` in `ModularClusterKmeansStream` (checking `hasattr(faiss, 'StandardGpuResources')` for `faiss-cpu` safety).

### B5. `_compute_k_range(n)` returns `k_min > k_max` for `n in {2, 3}` (Medium)
- **Locations:**
  - `positronixlab/modularcluster/modularcluster_kmeans.py:73-77, 231`
  - `positronixlab/modularcluster/modularcluster_kmeans_stream.py:81-85`
- **Root cause:** `fit()` checks `if n < 2: raise ValueError(...)`, but for `n = 2` or `n = 3`, `k_min = 2` while `k_max = min(int((2 * n) ** 0.5), n // 2) = 1`. `_compute_k_range` returns `(2, 1)`, so `_adaptive_search` runs `range(2, 2)` (0 iterations).
- **Impact:** `fit()` returns silently with `best_k=None`, `best_score=None`, `labels=None`, `centroids=None`. With `verbose=True`, `ModularClusterKmeans.fit()` crashes at line 231 with `KeyError: None`. (Furthermore, `sklearn.metrics.silhouette_score` requires `2 <= n_labels <= n_samples - 1`, so silhouette-based clustering mathematically requires `n >= 3`.)
- **Fix:** Require `n >= 3` (or handle `n == 2` without silhouette search), clamp `k_max = max(k_min, min(k_max, n - 1))`, and validate `k_min <= k_max <= n - 1`.

### B6. `ModularClusterKmeansStream.fit()` warm-start ignores configured `self.k_min` and `self.k_max` (Medium)
- **Location:** `positronixlab/modularcluster/modularcluster_kmeans_stream.py:259-265`
- **Root cause:** When `warm_start_k` and `k_search_radius` are provided, the window is clamped only to `[2, n - 1]` rather than `[self.k_min, self.k_max]`.
- **Impact:** A `ModularClusterKmeansStream` or `StreamingClusterTracker` configured with explicit `k_min`/`k_max` bounds will search and select `K` outside those bounds on warm-started continuation frames.
- **Fix:** Clamp the warm-start window against `self._compute_k_range(n)` (or explicit `self.k_min` / `self.k_max` when set).

### B7. Degenerate clustering (< 2 unique labels) crashes `_silhouette_score()` (Medium)
- **Locations:**
  - `positronixlab/modularcluster/modularcluster_kmeans.py:94-105`
  - `positronixlab/modularcluster/modularcluster_kmeans_stream.py:102-113`
- **Root cause:** Only `if k < 2` is checked. If input points are identical or FAISS produces empty clusters leaving only 1 unique assigned label, `sklearn.metrics.silhouette_score` raises `ValueError: Number of labels is 1. Valid values are 2 to n_samples - 1 (inclusive)`.
- **Fix:** Guard with `n_unique = len(np.unique(labels)); if n_unique < 2 or n_unique >= len(labels): return -1.0`.

### B8. `StreamingClusterTracker` hysteresis/debounce publishes stale assignments and leaks `_pending_k` (Low–Medium)
- **Location:** `positronixlab/modularcluster/streaming_cluster_tracker.py:173-181, 219-226`
- **Issues:**
  1. When `accepted_k != new_k`, `update()` republishes `self.last_stats` (previous frame's labels/centroids) instead of the current frame's clustering at `last_published_k` — even though `last_published_k` was just evaluated during the warm-start window.
  2. When `self._pending_k_streak >= self.debounce_steps` accepts `new_k` (`line 179-180`), `_pending_k` and `_pending_k_streak` are not reset immediately.
  3. `update(positions: np.ndarray)` assumes a NumPy array (`positions.copy()`, `np.abs(...)`) and raises `AttributeError` if passed a `torch.Tensor`, unlike `ModularClusterKmeans*`.

### B9. Unused parameters, dead constants, and exception-unsafe `_effective_max_iter` (Low)
- `tol` is accepted and stored in all three classes (`modularcluster_kmeans.py:34,56`, `modularcluster_kmeans_stream.py:37,59`, `streaming_cluster_tracker.py:41`) but never passed to FAISS or used.
- `EPS = 1e-8` is defined on both `ModularClusterKmeans` (`line 26`) and `ModularClusterKmeansStream` (`line 29`) and never used.
- Unused imports (`time`, `override`, `os`, `contextlib`) appear in both KMeans modules.
- In `ModularClusterKmeansStream.fit()` (`lines 253, 274`), `self._effective_max_iter` is reset outside a `try ... finally` block; an exception during `fit()` leaves the reduced `max_iter_override` active for future cold-start calls. Also `max_iter_override or self.max_iter` treats `0` as falsy.

---

## 3. Inconsistencies & Design Issues

1. **Stale pre-split test file (`tests/test_kmeans_plugin.py`):**
   - Imports `ModularClusterPlugin` from `positronixlab.modularcluster.modularcluster_plugin` and `PluginManager` from `positronixlab.plugin` — neither exists after the `parallelazy` / `AtlasCompose` split.
   - Breaks root `pytest` collection across the repository. Should be removed or migrated to `../AtlasCompose`.
2. **Massive code duplication between `ModularClusterKmeans` and `ModularClusterKmeansStream`:**
   - ~250 lines are copy-pasted verbatim between `modularcluster_kmeans.py` and `modularcluster_kmeans_stream.py`, causing bug fixes and implementation choices to diverge (e.g., `kmeans.assign` vs `_distance().argmin`, `gpu=` flag, `best_k is not None` check in `fit()`).
   - `ModularClusterKmeansStream` should either be merged into `ModularClusterKmeans` (as optional warm-start kwargs) or subclass it to avoid drift.
3. **Empty `__init__.py`:**
   - `positronixlab/modularcluster/__init__.py` is 0 bytes, forcing callers to import from internal filenames (`positronixlab.modularcluster.modularcluster_kmeans`, etc.) instead of `from positronixlab.modularcluster import ModularClusterKmeans, ...`.
4. **Overlap / relationship with `positronixlab/modularspace/custertracker.py` (`ClusterTracker`):**
   - Both `StreamingClusterTracker` (`modularcluster/streaming_cluster_tracker.py`) and `ClusterTracker` (`modularspace/custertracker.py`) track cluster identities over time, but at different levels:
     - `StreamingClusterTracker` wraps clustering itself, performs 1:1 Hungarian matching on centroid Euclidean distances **only when `k` is unchanged**, and suppresses `k` changes via hysteresis/debounce.
     - `ClusterTracker` consumes pre-computed `(assignment, centroids)` tensors, tracks full lineage transitions (`match`, `split`, `merge`, `switch`, `add`, `remove`) via item-overlap matrices, and projects clusters into an outer visualization space.
   - Neither references the other; documenting their complementary roles prevents confusion.
5. **Input type & metric inconsistency:**
   - `ModularClusterKmeans` / `ModularClusterKmeansStream` accept `torch.Tensor`, whereas `StreamingClusterTracker.update()` requires `np.ndarray` and returns NumPy arrays inside a dict.
   - `StreamingClusterTracker._relabel_to_match_previous()` always uses Euclidean `torch.cdist`, even when `metric='cosine'`. (For unit-normalized centroids from FAISS spherical K-Means, Euclidean distance is monotonic with cosine distance, so ordering is preserved, but this relies on an implicit invariant.)
6. **Stale `README.md`:**
   - `positronixlab/modularcluster/README.md` still documents `PluginModularCluster` and `modularcluster_plugin.py` and makes no mention of `ModularClusterKmeansStream` or `StreamingClusterTracker`.

---

## 4. Opportunities & Recommendations

1. **Immediate bug fixes (high value, low effort):**
   - Fix `B1` (`torch.from_numpy(labels).long().to(self.device)` in `modularcluster_kmeans.py:128`) — immediately fixes all 3 failing tests in `test_kmeans.py`.
   - Fix `B2` (only `fill_diagonal_(0.0)` on square self-distance matrices, not point-to-centroid distances) — fixes silent label corruption in `predict()` and `ModularClusterKmeansStream.fit()`.
   - Fix `B3` (reorder `centroids` alongside `labels` in `StreamingClusterTracker._relabel_to_match_previous()`).
   - Fix `B4` (convert `x` to contiguous CPU `float32` NumPy array before passing to FAISS `Kmeans.train`/`assign`, and guard `gpu=` with FAISS GPU availability).
   - Fix `B5`, `B6`, `B7` (small-`n` guard, clamp warm-start window to `[k_min, k_max]`, guard degenerate single-cluster silhouette).
2. **Consolidate `ModularClusterKmeans` and `ModularClusterKmeansStream`:**
   - Merge warm-start support (`warm_start_k`, `k_search_radius`, `max_iter_override`, `_centroid_cache`) into `ModularClusterKmeans` and keep `ModularClusterKmeansStream` as a thin alias/subclass for backward compatibility.
3. **Export public API in `__init__.py`:**
   - Re-export `ModularClusterKmeans`, `ModularClusterKmeansStream`, `StreamingClusterTracker` in `positronixlab/modularcluster/__init__.py`.
4. **Remove stale `tests/test_kmeans_plugin.py` and update `README.md`:**
   - Restores clean `pytest` collection across the repo and aligns root module docs with post-split architecture.
5. **Preserve per-`k` clustering results during `StreamingClusterTracker` debounce:**
   - When hysteresis/debounce holds `k` at `last_published_k`, evaluate/retain the current frame's clustering at `last_published_k` rather than republishing stale labels/centroids from the previous frame.

---

## 5. Docs / Tests / Examples Summary

- **Existing test results:**
  - `tests/test_kmeans.py`: **11 passed, 3 failed** (`test_print_stats`, `test_sync_euclidean`, `test_sync_cosine` — all due to **B1** `bincount_cpu` on `Float` labels).
  - `tests/test_kmeans_plugin.py`: **collection error** (`ModuleNotFoundError: positronixlab.modularcluster.modularcluster_plugin` — stale pre-split test, left intact per review scope).
- **Updated docs:**
  - `docs/MODULARCLUSTER_API.md` — removed stale `PluginModularCluster`; documented `ModularClusterKmeans`, `ModularClusterKmeansStream`, and `StreamingClusterTracker`.
  - `docs/MODULARCLUSTER_SYSTEM.md` — updated architecture, streaming pipeline flow, comparison with `modularspace.ClusterTracker`, and post-split notes.
  - `docs/MODULARCLUSTER_USAGE.md` — replaced pre-split plugin recipes with runnable `ModularClusterKmeans`, `ModularClusterKmeansStream`, and `StreamingClusterTracker` recipes.
- **Added tests:**
  - `tests/test_streaming_kmeans.py` — unit tests for `ModularClusterKmeansStream` and `StreamingClusterTracker`, plus `@pytest.mark.xfail(strict=True)` regression tests pinning bugs **B1**, **B2**, **B3**, **B5**, and **B6**.
- **Added example:**
  - `examples/example_modularcluster.py` — self-contained deterministic CPU example demonstrating `ModularClusterKmeans`, `ModularClusterKmeansStream` warm-starting, and `StreamingClusterTracker` over an evolving 10-step stream.
