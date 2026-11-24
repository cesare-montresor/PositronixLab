# ModularCluster Usage Guide

Practical recipes for `ModularClusterKmeans`, `ModularClusterKmeansStream`, and `StreamingClusterTracker`.

> For a runnable end-to-end script, see [../examples/example_modularcluster.py](../examples/example_modularcluster.py).

---

## 1. Basic Adaptive K-Means (`ModularClusterKmeans`)

```python
import torch
from positronixlab.modularcluster.modularcluster_kmeans import ModularClusterKmeans

torch.manual_seed(42)
embeddings = torch.cat([
    torch.randn(100, 16) + 5.0,
    torch.randn(100, 16) - 5.0,
    torch.randn(100, 16) + torch.tensor([5.0, -5.0] * 8),
], dim=0)

kmeans = ModularClusterKmeans(metric="cosine", k_min=2, k_max=6, verbose=False)
labels = kmeans.fit_predict(embeddings)

print(f"Best K: {kmeans.best_k}, Silhouette: {kmeans.best_score:.4f}")
for k, score in sorted(kmeans.silhouette_scores.items()):
    print(f"  K={k}: {score:.4f}")
```

---

## 2. Comparing Euclidean vs Cosine Metrics

```python
import torch
from positronixlab.modularcluster.modularcluster_kmeans import ModularClusterKmeans

embeddings = torch.randn(300, 32)

for metric in ("euclidean", "cosine"):
    km = ModularClusterKmeans(metric=metric, k_min=2, k_max=8)
    km.fit(embeddings)
    print(f"{metric:9s} -> best_k={km.best_k}, best_score={km.best_score:.4f}")
```

---

## 3. Warm-Started Sequential Clustering (`ModularClusterKmeansStream`)

When clustering consecutive snapshots of an evolving embedding space, use `ModularClusterKmeansStream` to reuse converged centroids across calls and restrict the `K` search window:

```python
import torch
from positronixlab.modularcluster.modularcluster_kmeans_stream import ModularClusterKmeansStream

torch.manual_seed(0)
x0 = torch.cat([
    torch.randn(60, 8) + 4.0,
    torch.randn(60, 8) - 4.0,
], dim=0)

stream_km = ModularClusterKmeansStream(metric="euclidean", k_min=2, k_max=6)

# Cold start: full adaptive search
labels_t0 = stream_km.fit_predict(x0)
k0 = stream_km.best_k

# Warm start on slightly shifted embeddings
x1 = x0 + 0.1 * torch.randn_like(x0)
labels_t1 = stream_km.fit_predict(
    x1,
    warm_start_k=k0,
    k_search_radius=1,
    max_iter_override=10,
)
print(f"t=0: K={k0}, t=1: K={stream_km.best_k}, cached Ks={sorted(stream_km._centroid_cache)}")
```

---

## 4. Full Streaming Pipeline (`StreamingClusterTracker`)

`StreamingClusterTracker` manages EMA drift gating, warm-start continuations, periodic full searches, Hungarian label stabilization, ARI diagnostics, and hysteresis/debounce on `K` changes:

```python
import numpy as np
from positronixlab.modularcluster.streaming_cluster_tracker import StreamingClusterTracker

rng = np.random.default_rng(42)
centers = np.array([[5.0, 5.0, 0.0, 0.0], [-5.0, -5.0, 0.0, 0.0], [5.0, -5.0, 5.0, -5.0]], dtype=np.float32)
positions = np.vstack([c + rng.normal(0, 0.4, size=(40, 4)).astype(np.float32) for c in centers])

tracker = StreamingClusterTracker(
    layer_id="latent_0",
    metric="euclidean",
    k_min=2,
    k_max=6,
    ema_alpha=0.2,
    drift_threshold=0.05,
    k_search_radius=1,
    full_search_every=5,
    warm_start_max_iter=10,
    hysteresis_margin=0.02,
    debounce_steps=2,
)

for step in range(8):
    # Static on steps 1-2, drifting afterward
    if step >= 3:
        positions = positions + rng.normal(0, 0.15, size=positions.shape).astype(np.float32)

    stats = tracker.update(positions)
    if stats is None:
        print(f"step={step}: skipped (static frame); reusing last_k={tracker.last_published_k}")
    else:
        print(
            f"step={step}: K={stats['best_k']}, score={stats['best_score']:.4f}, "
            f"drift={stats['drift']:.4f}, ari={stats['ari_vs_previous']}"
        )
```

---

## 5. Integrating with `parallelazy` / `AtlasCompose`

Because `positronixlab` is a pure library, wrap `StreamingClusterTracker` inside your application's `parallelazy` worker/plugin (e.g., holding one `StreamingClusterTracker` per `layer_id` in a dict and calling `tracker.update(positions)` on each received activation snapshot):

```python
trackers: dict[str, StreamingClusterTracker] = {}

def on_positions(layer_id: str, positions: np.ndarray):
    if layer_id not in trackers:
        trackers[layer_id] = StreamingClusterTracker(layer_id=layer_id, metric="cosine")
    stats = trackers[layer_id].update(positions)
    return stats if stats is not None else trackers[layer_id].last_stats
```

---

## Tips
- Use `precompute_distances=False` if `N` is large (e.g. `N > 5000`) to avoid allocating an `(N, N)` distance matrix.
- Keep `k_min` and `k_max` tight when you know the expected cluster-count range to speed up silhouette search.
- See [../analysis.md](../analysis.md) for current known bugs and workarounds.
