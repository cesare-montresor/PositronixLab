"""
Runnable CPU example for positronixlab.modularcluster.

Demonstrates:
  1. ModularClusterKmeans (adaptive K selection on synthetic clusters)
  2. ModularClusterKmeansStream (warm-start continuation with centroid caching)
  3. StreamingClusterTracker (EMA drift gating, periodic full search, ARI diagnostics)
"""

import numpy as np
import torch

from positronixlab.modularcluster.modularcluster_kmeans import ModularClusterKmeans
from positronixlab.modularcluster.modularcluster_kmeans_stream import ModularClusterKmeansStream
from positronixlab.modularcluster.streaming_cluster_tracker import StreamingClusterTracker


def make_synthetic_embeddings(n_per_cluster: int = 40, dim: int = 8, seed: int = 42) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    centers = torch.tensor(
        [
            [6.0] * dim,
            [-6.0] * dim,
            [6.0 if i % 2 == 0 else -6.0 for i in range(dim)],
        ],
        dtype=torch.float32,
    )
    return torch.cat(
        [c + 0.5 * torch.randn((n_per_cluster, dim), generator=gen) for c in centers],
        dim=0,
    )


def main() -> None:
    embeddings = make_synthetic_embeddings()

    print("=== 1. ModularClusterKmeans (Adaptive K-Means) ===")
    km = ModularClusterKmeans(metric="cosine", k_min=2, k_max=6, verbose=False)
    labels = km.fit_predict(embeddings)
    print(f"Selected K={km.best_k}, silhouette={km.best_score:.4f}, labels.shape={tuple(labels.shape)}")
    print(f"Evaluated K scores: { {k: round(v, 4) for k, v in sorted(km.silhouette_scores.items())} }\n")

    print("=== 2. ModularClusterKmeansStream (Warm-Started) ===")
    stream_km = ModularClusterKmeansStream(metric="euclidean", k_min=2, k_max=6, max_iter=50)
    stream_km.fit(embeddings)
    print(f"Cold start: K={stream_km.best_k}, silhouette={stream_km.best_score:.4f}")

    shifted = embeddings + 0.1 * torch.randn_like(embeddings)
    stream_km.fit(shifted, warm_start_k=stream_km.best_k, k_search_radius=1, max_iter_override=10)
    print(
        f"Warm start: K={stream_km.best_k}, silhouette={stream_km.best_score:.4f}, "
        f"evaluated Ks={sorted(stream_km.silhouette_scores.keys())}\n"
    )

    print("=== 3. StreamingClusterTracker (10-Step Stream) ===")
    rng = np.random.default_rng(42)
    positions = embeddings.numpy().copy()
    tracker = StreamingClusterTracker(
        layer_id="latent_0",
        metric="euclidean",
        k_min=2,
        k_max=6,
        ema_alpha=0.25,
        drift_threshold=0.05,
        k_search_radius=1,
        full_search_every=5,
        warm_start_max_iter=10,
        hysteresis_margin=0.02,
        debounce_steps=2,
    )

    for step in range(10):
        # Steps 1-2: static (triggers drift gating); Steps 3+: smooth drift
        if step >= 3:
            positions = positions + rng.normal(0.0, 0.12, size=positions.shape).astype(np.float32)

        stats = tracker.update(positions)
        if stats is None:
            print(f"  step {step:2d}: skipped (static frame; last_k={tracker.last_published_k})")
        else:
            ari_str = "None" if stats["ari_vs_previous"] is None else f"{stats['ari_vs_previous']:.4f}"
            print(
                f"  step {step:2d}: K={stats['best_k']}, score={stats['best_score']:.4f}, "
                f"drift={stats['drift']:.4f}, ARI={ari_str}"
            )


if __name__ == "__main__":
    main()
