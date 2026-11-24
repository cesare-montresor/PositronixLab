# ModularCluster System Architecture

`positronixlab/modularcluster` provides FAISS-accelerated K-Means clustering with adaptive `K` selection and stateful streaming cluster tracking for evolving neural-network embedding spaces.

> **Ecosystem note:** Concrete multiprocess plugin wrappers (previously `PluginModularCluster` / `ModularClusterPlugin`) were moved to `../AtlasCompose` on top of `../parallelazy`. `positronixlab/modularcluster` contains only the reusable library components.

---

## Components

```
ModularClusterKmeans           (one-shot / batch adaptive K-Means)

ModularClusterKmeansStream     (warm-startable adaptive K-Means with per-K centroid cache)
        ▲
        │ composed inside
        │
StreamingClusterTracker        (per-layer streaming state: EMA drift gating,
                                periodic full search, Hungarian relabeling,
                                ARI diagnostics, hysteresis + debounce on K)
```

### 1. `ModularClusterKmeans` (`modularcluster_kmeans.py`)
- Wraps `faiss.Kmeans` with `'euclidean'` and `'cosine'` (spherical, L2-normalized) metrics.
- Selects `K` automatically by maximizing `sklearn.metrics.silhouette_score`.
- Optionally caches the `(n, n)` pairwise distance matrix (`precompute_distances=True`) so silhouette evaluation across candidate `K` values avoids recomputing distances.

### 2. `ModularClusterKmeansStream` (`modularcluster_kmeans_stream.py`)
- Extends the adaptive K-Means workflow for sequential frames where embeddings move smoothly:
  - **Per-`k` centroid cache (`_centroid_cache`)**: persists converged centroids across `fit()` calls and passes them as `init_centroids` to FAISS, making continuation steps fast.
  - **Narrow warm-start window**: `fit(x, warm_start_k=..., k_search_radius=..., max_iter_override=...)` exhaustively checks only `[warm_start_k - r, warm_start_k + r]` with a reduced iteration budget.

### 3. `StreamingClusterTracker` (`streaming_cluster_tracker.py`)
Owns all per-layer temporal state for a continuously evolving position stream:
1. **EMA drift gating**: tracks an exponential moving average `ema_positions` (`ema_alpha`). If `mean(|positions - ema_positions|) < drift_threshold` and no full search is due, `update()` skips clustering and returns `None`.
2. **Warm-start vs periodic full search**: runs a full adaptive search on frame 0 and every `full_search_every` steps; runs a narrow warm-started search (`k_search_radius`, `warm_start_max_iter`) on intermediate frames.
3. **Hysteresis + debounce on `K`**: a new `best_k != last_published_k` must beat `last_published_k`'s silhouette score by more than `hysteresis_margin` for `debounce_steps` consecutive evaluations before being accepted.
4. **Hungarian relabeling**: when `K` is unchanged, solves `scipy.optimize.linear_sum_assignment` on centroid distances to keep cluster IDs consistent across frames.
5. **Diagnostics**: reports `drift` and `ari_vs_previous` (`sklearn.metrics.adjusted_rand_score`).

---

## Adaptive `K` Search Algorithm

### Default Heuristic Range
When `k_min` / `k_max` are omitted:
- `k_min = max(2, int(sqrt(n) / 2))`
- `k_max = min(int(sqrt(2 * n)), n // 2)` (clamped to `n - 1`)

### Three-Phase Search
If `k_max - k_min <= 10`, every integer `k` in `[k_min, k_max]` is evaluated exhaustively. Otherwise:
1. **Coarse sampling**: evaluates up to 10 evenly spaced `k` values across `[k_min, k_max]` (always including endpoints).
2. **Peak detection**: identifies interior local maxima (`score[k] > score[prev]` and `score[k] > score[next]`) plus boundary peaks.
3. **Local refinement**: evaluates `[peak - 3, peak + 3]` (clamped to neighboring coarse samples) around each peak. Already-evaluated `k` values are skipped via `self.silhouette_scores`.

---

## Streaming Data Flow

```
positions (np.ndarray, [N, D])
       │
       ▼
  _drift(positions) vs drift_threshold
       ├── static & not full_search_every ──► _update_ema() ──► return None
       │
       ▼ (active or periodic full search)
  ModularClusterKmeansStream.fit(x, ...)
       │
       ▼
  _gate_k_change(new_k)   (hysteresis_margin + debounce_steps)
       ├── held back ──► refresh drift/ARI=1.0 on last_stats ──► return stats
       │
       ▼ (accepted)
  _relabel_to_match_previous()  (Hungarian assignment when K unchanged)
       │
       ▼
  compute ARI, update last_published_*, return stats dict
```

---

## Relationship to `modularspace.ClusterTracker`

PositronixLab has two cluster-tracking classes with distinct scopes:

| Aspect | `modularcluster.StreamingClusterTracker` | `modularspace.ClusterTracker` (`custertracker.py`) |
| :--- | :--- | :--- |
| Owns clustering? | Yes (wraps `ModularClusterKmeansStream`) | No (consumes pre-computed `(assignment, centroids)`) |
| Matching method | Centroid distance + Hungarian assignment (1:1, same `K` only) | Item-overlap matrix across arbitrary `K` changes |
| Transitions modeled | Same-`K` continuity + debounced `K` change | `match`, `split`, `merge`, `switch`, `add`, `remove` |
| Outer-space projection | No | Yes (maintains stable outer-space coordinates) |

They can be composed: `StreamingClusterTracker` stabilizes online clustering, and `ClusterTracker` maps the resulting cluster assignments into a persistent outer-space atlas.

---

## Performance Considerations

- **Distance precomputation (`precompute_distances=True`)**: allocates an `(N, N)` float32 matrix (~400 MB for `N = 10,000`). Set `precompute_distances=False` for large `N`.
- **Silhouette bottleneck**: `sklearn.metrics.silhouette_score` is $O(N^2)$ per evaluated `K`. Narrowing `[k_min, k_max]` or using `StreamingClusterTracker` warm-starts significantly reduces wall time.
- **Known caveats**: see [../analysis.md](../analysis.md) for current bugs (including CUDA tensor handling, `float32` labels in `ModularClusterKmeans`, and `_distance()` diagonal zeroing).

---

## Testing

```bash
# Synchronous K-Means tests
.venv/bin/pytest positronixlab/modularcluster/tests/test_kmeans.py -v

# Streaming K-Means & StreamingClusterTracker tests
.venv/bin/pytest positronixlab/modularcluster/tests/test_streaming_kmeans.py -v
```
