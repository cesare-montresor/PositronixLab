"""
Unit tests for ModularClusterKmeansStream and StreamingClusterTracker, plus
strict xfail regression tests documenting confirmed bugs in modularcluster.
"""

import numpy as np
import pytest
import torch

from positronixlab.modularcluster.modularcluster_kmeans import ModularClusterKmeans
from positronixlab.modularcluster.modularcluster_kmeans_stream import ModularClusterKmeansStream
from positronixlab.modularcluster.streaming_cluster_tracker import StreamingClusterTracker


def _make_clustered_data(n_per_cluster: int = 30, dim: int = 8, seed: int = 0) -> torch.Tensor:
    """Create 3 well-separated Gaussian clusters."""
    gen = torch.Generator().manual_seed(seed)
    centers = torch.tensor(
        [
            [8.0] * dim,
            [-8.0] * dim,
            [8.0 if i % 2 == 0 else -8.0 for i in range(dim)],
        ],
        dtype=torch.float32,
    )
    parts = [c + 0.4 * torch.randn((n_per_cluster, dim), generator=gen) for c in centers]
    return torch.cat(parts, dim=0)


class TestModularClusterKmeansStream:
    """Unit tests for ModularClusterKmeansStream."""

    def test_fit_predict_euclidean_and_cosine(self):
        x = _make_clustered_data(n_per_cluster=30, dim=8, seed=1)
        for metric in ("euclidean", "cosine"):
            km = ModularClusterKmeansStream(metric=metric, k_min=2, k_max=5)
            labels = km.fit_predict(x)
            assert labels.shape == (x.shape[0],)
            assert labels.dtype == torch.int64
            assert 2 <= km.best_k <= 5
            assert km.centroids.shape == (km.best_k, 8)
            assert -1.0 <= km.best_score <= 1.0
            assert set(km.silhouette_scores.keys()) == {2, 3, 4, 5}

    def test_centroid_cache_and_warm_start(self):
        x = _make_clustered_data(n_per_cluster=30, dim=8, seed=2)
        km = ModularClusterKmeansStream(metric="euclidean", k_min=2, k_max=5, max_iter=50)
        km.fit(x)
        assert 3 in km._centroid_cache
        assert km._centroid_cache[3].shape == (3, 8)

        # Warm-start narrow window around k=3 with reduced iterations
        x_next = x + 0.05 * torch.randn_like(x)
        km.fit(x_next, warm_start_k=3, k_search_radius=1, max_iter_override=5)
        assert set(km.silhouette_scores.keys()) == {2, 3, 4}
        assert km._effective_max_iter == 50  # reset after fit()

    def test_centroid_cache_handles_feature_dim_change(self):
        x8 = _make_clustered_data(n_per_cluster=20, dim=8, seed=3)
        x12 = _make_clustered_data(n_per_cluster=20, dim=12, seed=4)
        km = ModularClusterKmeansStream(metric="euclidean", k_min=2, k_max=4)
        km.fit(x8)
        assert km._centroid_cache[3].shape == (3, 8)

        # Should ignore stale 8-dim cached centroids when fitting 12-dim data
        km.fit(x12)
        assert km._centroid_cache[3].shape == (3, 12)
        assert km.centroids.shape == (km.best_k, 12)

    def test_print_stats_succeeds(self, capsys):
        x = _make_clustered_data(n_per_cluster=20, dim=8, seed=5)
        km = ModularClusterKmeansStream(metric="euclidean", k_min=2, k_max=4)
        km.fit(x)
        km.print_stats()
        out = capsys.readouterr().out
        assert "Best K:" in out
        assert "Cluster sizes:" in out

    def test_error_conditions(self):
        km = ModularClusterKmeansStream(metric="cosine")
        with pytest.raises(ValueError, match="Must call fit"):
            km.predict(torch.randn(10, 4))
        with pytest.raises(ValueError, match="at least 2 samples"):
            km.fit(torch.randn(1, 4))


class TestStreamingClusterTracker:
    """Unit tests for StreamingClusterTracker."""

    def test_first_frame_and_static_drift_gating(self):
        x = _make_clustered_data(n_per_cluster=25, dim=8, seed=10).numpy()
        tracker = StreamingClusterTracker(
            layer_id="layer_0",
            metric="euclidean",
            k_min=2,
            k_max=5,
            drift_threshold=0.05,
            full_search_every=10,
        )

        # Frame 0: always clusters (drift = inf)
        s0 = tracker.update(x)
        assert s0 is not None
        assert s0["best_k"] == 3
        assert s0["drift"] == float("inf")
        assert s0["ari_vs_previous"] is None
        assert s0["labels"].shape == (x.shape[0],)
        assert s0["centroids"].shape == (3, 8)

        # Frame 1: identical positions -> static -> returns None
        s1 = tracker.update(x.copy())
        assert s1 is None
        assert tracker.last_stats is s0

    def test_periodic_full_search_overrides_static_gate(self):
        x = _make_clustered_data(n_per_cluster=20, dim=6, seed=11).numpy()
        tracker = StreamingClusterTracker(
            layer_id="layer_1",
            metric="euclidean",
            k_min=2,
            k_max=4,
            drift_threshold=1.0,  # high threshold so normal frames are skipped
            full_search_every=3,
        )
        assert tracker.update(x) is not None  # frame 0 (initial full search, counter reset to 0)
        assert tracker.update(x) is None      # step 1
        assert tracker.update(x) is None      # step 2
        s3 = tracker.update(x)                # step 3 == full_search_every -> forced full search
        assert s3 is not None
        assert s3["best_k"] == 3
        assert s3["ari_vs_previous"] == pytest.approx(1.0)

    def test_warm_start_continuation_and_label_stability(self):
        rng = np.random.default_rng(12)
        x = _make_clustered_data(n_per_cluster=25, dim=8, seed=12).numpy()
        tracker = StreamingClusterTracker(
            layer_id="layer_2",
            metric="euclidean",
            k_min=2,
            k_max=5,
            drift_threshold=0.01,
            k_search_radius=1,
            full_search_every=20,
            warm_start_max_iter=10,
        )
        s0 = tracker.update(x)
        assert s0 is not None

        # Small perturbation exceeding drift_threshold
        x1 = x + rng.normal(0.0, 0.1, size=x.shape).astype(np.float32)
        s1 = tracker.update(x1)
        assert s1 is not None
        assert s1["best_k"] == 3
        assert s1["ari_vs_previous"] > 0.9
        # Warm-start should only evaluate [2, 3, 4]
        assert set(s1["silhouette_scores"].keys()) == {2, 3, 4}

    def test_hysteresis_and_debounce_gating(self):
        tracker = StreamingClusterTracker(
            layer_id="layer_3",
            metric="euclidean",
            hysteresis_margin=0.05,
            debounce_steps=2,
        )
        tracker.last_published_k = 3

        # Below hysteresis margin -> rejected
        tracker.kmeans.silhouette_scores = {3: 0.50, 4: 0.52}
        tracker.kmeans.best_score = 0.52
        assert tracker._gate_k_change(4) == 3

        # Above hysteresis margin, streak 1 < debounce_steps (2) -> still held back
        tracker.kmeans.silhouette_scores = {3: 0.50, 4: 0.60}
        tracker.kmeans.best_score = 0.60
        assert tracker._gate_k_change(4) == 3

        # Above hysteresis margin, streak 2 == debounce_steps -> accepted
        assert tracker._gate_k_change(4) == 4


class TestKnownBugsXfail:
    """Strict xfail tests pinning confirmed defects documented in analysis.md."""

    def test_b1_kmeans_labels_dtype_is_int64(self):
        x = _make_clustered_data(n_per_cluster=20, dim=8, seed=20)
        km = ModularClusterKmeans(metric="euclidean", k_min=2, k_max=4)
        labels = km.fit_predict(x)
        assert labels.dtype == torch.int64

    @pytest.mark.xfail(
        strict=True,
        reason="Bug B2: _distance() calls fill_diagonal_(0.0) on (n, k) point-to-centroid matrix, corrupting predict()[:k]",
    )
    def test_b2_predict_does_not_force_diagonal_assignment(self):
        x = _make_clustered_data(n_per_cluster=30, dim=8, seed=21)
        km = ModularClusterKmeans(metric="euclidean", k_min=3, k_max=3)
        km.fit(x)
        # All 10 query points belong to cluster 0's region
        same_cluster_pts = x[:10].clone()
        preds = km.predict(same_cluster_pts)
        assert len(torch.unique(preds)) == 1

    @pytest.mark.xfail(
        strict=True,
        reason="Bug B2: ModularClusterKmeansStream._kmeans() uses _distance(x, centroids) which zeros diagonal and corrupts labels[:k]",
    )
    def test_b2_stream_fit_does_not_corrupt_first_k_labels(self):
        x = _make_clustered_data(n_per_cluster=30, dim=8, seed=22)
        km = ModularClusterKmeansStream(metric="euclidean", k_min=3, k_max=3)
        labels = km.fit_predict(x)
        # First 30 points are all from the exact same tight cluster
        assert len(torch.unique(labels[:30])) == 1

    def test_b3_tracker_reorders_centroids_on_relabel(self):
        tracker = StreamingClusterTracker("layer_x", metric="euclidean", k_min=2, k_max=2, drift_threshold=0.0)
        # Seed previous centroids in order [cA, cB]
        c_a = np.array([10.0, 0.0, 0.0, 0.0], dtype=np.float32)
        c_b = np.array([-10.0, 0.0, 0.0, 0.0], dtype=np.float32)
        tracker.last_published_centroids = np.stack([c_a, c_b], axis=0)
        tracker.last_published_k = 2
        tracker.last_published_labels = np.array([0] * 10 + [1] * 10, dtype=np.int64)

        # Simulate FAISS returning swapped centroid order [cB, cA]
        swapped_centroids = torch.tensor(np.stack([c_b, c_a], axis=0))
        swapped_labels = torch.tensor([1] * 10 + [0] * 10, dtype=torch.int64)
        tracker.kmeans.centroids = swapped_centroids
        tracker.kmeans.labels = swapped_labels

        reordered_centroids, remapped_labels = tracker._relabel_to_match_previous(swapped_centroids, swapped_labels)
        assert (remapped_labels[:10] == 0).all()
        assert torch.allclose(reordered_centroids[0], torch.tensor(c_a))

    def test_b5_small_n_three_samples_finds_valid_k(self):
        km = ModularClusterKmeans(metric="euclidean")
        km.fit(torch.tensor([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0]]))
        assert km.best_k is not None
        assert km.labels is not None

    @pytest.mark.xfail(
        strict=True,
        reason="Bug B6: ModularClusterKmeansStream.fit(warm_start_k=..., k_search_radius=...) ignores self.k_min/self.k_max",
    )
    def test_b6_warm_start_respects_configured_k_bounds(self):
        x = _make_clustered_data(n_per_cluster=20, dim=8, seed=23)
        km = ModularClusterKmeansStream(metric="euclidean", k_min=3, k_max=4)
        km.fit(x, warm_start_k=3, k_search_radius=2)
        assert all(3 <= k <= 4 for k in km.silhouette_scores.keys())
