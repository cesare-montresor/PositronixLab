"""
Synchronous tests for ModularClusterKmeans.

Tests the core clustering algorithm with adaptive K selection,
supporting both Euclidean and cosine (spherical) distance metrics.
"""
import time
import torch
import pytest

from modularnet.modularcluster.modularcluster_kmeans import ModularClusterKmeans


class TestModularClusterKmeans:
    """Tests for ModularClusterKmeans synchronous clustering."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample embedding data for testing."""
        torch.manual_seed(42)
        n_samples = 1000
        n_features = 128
        embeddings = torch.randn(n_samples, n_features)
        return embeddings
    
    def test_euclidean_kmeans_basic(self, sample_data):
        """Test basic Euclidean KMeans clustering."""
        kmeans = ModularClusterKmeans(metric='euclidean', verbose=False)
        labels = kmeans.fit_predict(sample_data)
        
        assert labels is not None
        assert len(labels) == len(sample_data)
        assert kmeans.centroids is not None
        assert kmeans.best_k is not None
        assert kmeans.best_score is not None
        assert kmeans.best_k > 0
        assert len(torch.unique(labels)) == kmeans.best_k
    
    def test_cosine_kmeans_basic(self, sample_data):
        """Test basic Cosine KMeans clustering (spherical)."""
        kmeans = ModularClusterKmeans(metric='cosine', verbose=False)
        labels = kmeans.fit_predict(sample_data)
        
        assert labels is not None
        assert len(labels) == len(sample_data)
        assert kmeans.centroids is not None
        assert kmeans.best_k is not None
        assert kmeans.best_score is not None
    
    def test_adaptive_k_selection(self, sample_data):
        """Test that adaptive K selection finds reasonable K."""
        kmeans = ModularClusterKmeans(metric='cosine', verbose=False)
        kmeans.fit(sample_data)
        
        # For 1000 samples, K should be reasonable
        # sqrt(1000/2) ≈ 22, sqrt(2*1000) ≈ 45
        assert 2 <= kmeans.best_k <= len(sample_data) // 2
        
        # Best K should appear in silhouette scores
        assert kmeans.best_k in kmeans.silhouette_scores
    
    def test_k_range_heuristic(self, sample_data):
        """Test K range heuristic based on dataset size."""
        kmeans = ModularClusterKmeans(metric='cosine', verbose=False)
        n = len(sample_data)
        
        k_min, k_max = kmeans._compute_k_range(n)
        
        assert k_min >= 2
        assert k_max <= n - 1
        assert k_min <= k_max
    
    def test_predict_with_fitted_model(self, sample_data):
        """Test predict() on new data after fitting."""
        kmeans = ModularClusterKmeans(metric='euclidean', verbose=False)
        kmeans.fit(sample_data[:500])
        
        # Predict on rest of data
        new_labels = kmeans.predict(sample_data[500:])
        
        assert len(new_labels) == 500
        assert all(l < kmeans.best_k for l in new_labels.tolist())
    
    def test_predict_before_fit_raises(self):
        """Test that predict() raises error before fit()."""
        kmeans = ModularClusterKmeans(metric='cosine')
        data = torch.randn(100, 32)
        
        with pytest.raises(ValueError, match="Must call fit"):
            kmeans.predict(data)
    
    def test_fit_with_small_data_raises(self):
        """Test that fit() raises error with insufficient data."""
        kmeans = ModularClusterKmeans(metric='cosine')
        data = torch.randn(1, 32)
        
        with pytest.raises(ValueError, match="at least 2 samples"):
            kmeans.fit(data)
    
    def test_explicit_k_range(self, sample_data):
        """Test clustering with explicit k_min and k_max."""
        kmeans = ModularClusterKmeans(
            metric='cosine',
            k_min=5,
            k_max=15,
            verbose=False
        )
        kmeans.fit(sample_data)
        
        # Best K should be within specified range
        assert 5 <= kmeans.best_k <= 15
    
    def test_device_handling(self, sample_data):
        """Test device handling (CPU)."""
        device = torch.device('cpu')
        kmeans = ModularClusterKmeans(metric='cosine', device=device, verbose=False)
        labels = kmeans.fit_predict(sample_data)
        
        assert labels.device.type == 'cpu'
        assert kmeans.centroids.device.type == 'cpu'
    
    def test_silhouette_scores_computed(self, sample_data):
        """Test that silhouette scores are computed for all K values."""
        kmeans = ModularClusterKmeans(metric='cosine', k_min=5, k_max=10, verbose=False)
        kmeans.fit(sample_data)
        
        # Should have scores for K values in range
        assert len(kmeans.silhouette_scores) > 0
        
        # Scores should be reasonable
        for k, score in kmeans.silhouette_scores.items():
            assert -1 <= score <= 1, f"Invalid silhouette score {score} for K={k}"
    
    def test_precompute_distances_option(self, sample_data):
        """Test precompute_distances option."""
        kmeans_precomp = ModularClusterKmeans(
            metric='cosine',
            precompute_distances=True,
            verbose=False
        )
        kmeans_no_precomp = ModularClusterKmeans(
            metric='cosine',
            precompute_distances=False,
            verbose=False
        )
        
        torch.manual_seed(42)
        labels1 = kmeans_precomp.fit_predict(sample_data)
        
        torch.manual_seed(42)
        labels2 = kmeans_no_precomp.fit_predict(sample_data)
        
        # Both should produce clusterings (quality may differ slightly due to randomness)
        assert len(labels1) == len(labels2)
    
    def test_print_stats(self, sample_data, capsys):
        """Test print_stats output."""
        kmeans = ModularClusterKmeans(metric='cosine', verbose=False)
        kmeans.fit(sample_data)
        kmeans.print_stats()
        
        captured = capsys.readouterr()
        
        # Check that output contains expected information
        assert "Best K:" in captured.out
        assert "Cluster sizes:" in captured.out
        assert "Silhouette Score:" in captured.out


def test_sync_euclidean(verbose: bool = False):
    """Quick synchronous test for Euclidean KMeans."""
    torch.manual_seed(42)
    n_samples = 1000
    n_features = 128
    embeddings = torch.randn(n_samples, n_features)
    
    print("=== Euclidean KMeans ===")
    start = time.time()
    kmeans = ModularClusterKmeans(metric='euclidean', verbose=verbose)
    labels = kmeans.fit_predict(embeddings)
    kmeans.print_stats()
    print(f"time taken: {time.time() - start:.2f} seconds")
    print(f"Cluster sizes: {torch.bincount(labels)}\n")


def test_sync_cosine(verbose: bool = False):
    """Quick synchronous test for Cosine KMeans."""
    torch.manual_seed(42)
    n_samples = 1000
    n_features = 128
    embeddings = torch.randn(n_samples, n_features)
    
    print("=== Cosine (Spherical) KMeans ===")
    start = time.time()
    kmeans_cos = ModularClusterKmeans(metric='cosine', verbose=verbose)
    labels_cos = kmeans_cos.fit_predict(embeddings)
    kmeans_cos.print_stats()
    print(f"Cluster sizes: {torch.bincount(labels_cos)}\n")
    print(f"time taken: {time.time() - start:.2f} seconds")
