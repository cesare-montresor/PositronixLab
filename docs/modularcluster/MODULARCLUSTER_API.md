# ModularCluster API Reference

API documentation for `ModularClusterKmeans`, `ModularClusterKmeansStream`, and `StreamingClusterTracker`.

> **Note (Post-Split Architecture):** Concrete async plugin wrappers (`PluginModularCluster` / `ModularClusterPlugin`) were moved out of `positronixlab` into `AtlasCompose` (built on `parallelazy`). This package provides the reusable clustering and streaming-tracking engines only.

---

## `ModularClusterKmeans`

Synchronous adaptive K-Means clustering engine with 3-phase silhouette-score `K` selection.

### Import
```python
from positronixlab.modularcluster.modularcluster_kmeans import ModularClusterKmeans
```

### Constructor
```python
ModularClusterKmeans(
    metric: Literal['euclidean', 'cosine'] = 'cosine',
    k_min: Optional[int] = None,
    k_max: Optional[int] = None,
    max_iter: int = 100,
    tol: float = 1e-5,
    n_init: int = 1,
    device: Optional[torch.device] = None,
    verbose: bool = False,
    precompute_distances: bool = True,
)
```

**Parameters:**
- `metric` (`'euclidean' | 'cosine'`, default `'cosine'`): Distance metric. `'cosine'` L2-normalizes inputs and uses FAISS spherical K-Means.
- `k_min` (`int | None`): Minimum `K` to evaluate. Heuristic default: `max(2, int(sqrt(n) / 2))`.
- `k_max` (`int | None`): Maximum `K` to evaluate. Heuristic default: `min(int(sqrt(2 * n)), n // 2)`, clamped to `n - 1`.
- `max_iter` (`int`, default `100`): Maximum FAISS K-Means iterations (`niter`).
- `tol` (`float`, default `1e-5`): Stored on `self.tol` (currently unused by FAISS K-Means).
- `n_init` (`int`, default `1`): Number of FAISS random restarts (`nredo`).
- `device` (`torch.device | None`): Target PyTorch device (default CPU). Inputs are automatically detached to CPU NumPy `float32` before FAISS training, and `ModularClusterKmeans` enables FAISS GPU mode when `device.type == 'cuda'`.
- `verbose` (`bool`, default `False`): Print search progress.
- `precompute_distances` (`bool`, default `True`): Precompute the `(n, n)` pairwise distance matrix once per `fit()` for faster silhouette evaluation.

### Methods
- `fit(x: torch.Tensor) -> ModularClusterKmeans`: Runs adaptive `K` search on `x` of shape `(n_samples, n_features)`. Raises `ValueError` if `n_samples < 2`.
- `predict(x: torch.Tensor) -> torch.Tensor`: Assigns new points `(n_samples, n_features)` to nearest learned centroid. Raises `ValueError` if called before `fit()`.
- `fit_predict(x: torch.Tensor) -> torch.Tensor`: Calls `fit(x)` and returns `self.labels`.
- `print_stats() -> None`: Prints `best_k`, `best_score`, silhouette score, cluster sizes, and evaluated `K` values.

### Attributes (populated after `fit()`)
- `centroids: Optional[torch.Tensor]` — shape `(best_k, n_features)`.
- `labels: Optional[torch.Tensor]` — shape `(n_samples,)`.
- `best_k: Optional[int]` — selected optimal `K`.
- `best_score: Optional[float]` — highest silhouette score in `[-1, 1]`.
- `silhouette_scores: Dict[int, float]` — mapping of every evaluated `k -> silhouette_score`.
- `distance_cache: Optional[np.ndarray]` — `(n_samples, n_samples)` distance matrix when `precompute_distances=True`.

---

## `ModularClusterKmeansStream`

Warm-startable adaptive K-Means clustering engine for streaming / sequential frames.

### Import
```python
from positronixlab.modularcluster.modularcluster_kmeans_stream import ModularClusterKmeansStream
```

### Constructor
Identical signature to `ModularClusterKmeans`.

Additionally maintains:
- `_centroid_cache: Dict[int, np.ndarray]` — persists converged centroids per `k` **across** `fit()` calls so subsequent frames warm-start FAISS via `init_centroids` (automatically invalidated if feature dimensionality changes).
- `_effective_max_iter: int` — active iteration budget for the current `fit()` call.

### Methods
- `fit(x: torch.Tensor, warm_start_k: Optional[int] = None, k_search_radius: Optional[int] = None, max_iter_override: Optional[int] = None) -> ModularClusterKmeansStream`
  - When both `warm_start_k` and `k_search_radius` are provided, evaluates every `k` in `[max(2, warm_start_k - k_search_radius), min(warm_start_k + k_search_radius, n - 1)]` instead of running the full adaptive search.
  - `max_iter_override` temporarily overrides `self.max_iter` for this `fit()` call (reset to `self.max_iter` after `fit()` returns).
- `fit_predict(x: torch.Tensor, **fit_kwargs) -> torch.Tensor`: Forwards `**fit_kwargs` to `fit()` and returns `self.labels` (`torch.int64`).
- `predict(x: torch.Tensor) -> torch.Tensor`: Assigns new points to nearest centroid.
- `print_stats() -> None`: Prints clustering statistics.

---

## `StreamingClusterTracker`

Stateful per-layer streaming tracker wrapping `ModularClusterKmeansStream`.

### Import
```python
from positronixlab.modularcluster.streaming_cluster_tracker import StreamingClusterTracker
```

### Constructor
```python
StreamingClusterTracker(
    layer_id: str,
    metric: str = "cosine",
    k_min: Optional[int] = None,
    k_max: Optional[int] = None,
    max_iter: int = 100,
    tol: float = 1e-5,
    n_init: int = 1,
    device: Optional[torch.device] = None,
    verbose: bool = False,
    precompute_distances: bool = True,
    # Streaming-specific parameters:
    ema_alpha: float = 0.2,
    drift_threshold: float = 0.02,
    k_search_radius: int = 3,
    full_search_every: int = 20,
    warm_start_max_iter: int = 10,
    hysteresis_margin: float = 0.01,
    debounce_steps: int = 2,
)
```

**Streaming Parameters:**
- `layer_id` (`str`): Layer identifier owned by this tracker.
- `ema_alpha` (`float`, default `0.2`): EMA decay for the drift-gating baseline (`ema = alpha * pos + (1 - alpha) * ema`).
- `drift_threshold` (`float`, default `0.02`): Mean absolute deviation from `ema_positions` below which clustering is skipped.
- `k_search_radius` (`int`, default `3`): Half-width of the narrow `K` window searched around `last_published_k` on continuation frames.
- `full_search_every` (`int`, default `20`): Force a full-range adaptive search every N calls to `update()`, regardless of drift.
- `warm_start_max_iter` (`int`, default `10`): FAISS `niter` used on warm-started continuation frames.
- `hysteresis_margin` (`float`, default `0.01`): Minimum silhouette-score improvement required over `last_published_k` before considering a `K` change.
- `debounce_steps` (`int`, default `2`): Consecutive evaluations a candidate `K` must win by `hysteresis_margin` before being accepted.

### Methods

#### `update(positions: np.ndarray) -> Optional[Dict[str, Any]]`
Processes one timestep of positions (shape `(n_samples, n_features)`, NumPy array).

- Returns `None` when the frame is skipped as static (`drift < drift_threshold` and not a forced full search and `last_stats is not None`). Callers can republish `tracker.last_stats` if needed.
- Otherwise returns (and stores in `self.last_stats`) a dictionary:

| Key | Type | Description |
| :--- | :--- | :--- |
| `"best_k"` | `int` | Accepted cluster count |
| `"best_score"` | `float` | Silhouette score for `best_k` |
| `"labels"` | `np.ndarray` | Cluster assignments, shape `(n_samples,)` |
| `"centroids"` | `np.ndarray` | Cluster centroids, shape `(best_k, n_features)` |
| `"silhouette_scores"` | `Dict[int, float]` | Evaluated `k -> score` mapping from this step |
| `"ari_vs_previous"` | `Optional[float]` | Adjusted Rand Index vs `last_published_labels` (`None` on first frame or shape change; `1.0` when held back by hysteresis/debounce) |
| `"drift"` | `float` | Mean absolute deviation vs `ema_positions` (`inf` on first frame or shape change) |

### Key Attributes
- `kmeans: ModularClusterKmeansStream` — underlying warm-startable clusterer.
- `ema_positions: Optional[np.ndarray]` — running EMA position baseline.
- `last_published_labels: Optional[np.ndarray]` — last accepted labels.
- `last_published_centroids: Optional[np.ndarray]` — last accepted centroids.
- `last_published_k: Optional[int]` — last accepted `K`.
- `last_stats: Optional[Dict[str, Any]]` — last published stats dict.

---

## See Also
- [MODULARCLUSTER_SYSTEM.md](MODULARCLUSTER_SYSTEM.md) — Architecture, search phases, and design notes
- [MODULARCLUSTER_USAGE.md](MODULARCLUSTER_USAGE.md) — Practical examples and recipes
- [`reports/modularcluster_analysis.md`](file:///home/cesare/Projects/PositronixLab/reports/modularcluster_analysis.md) — Module review notes
