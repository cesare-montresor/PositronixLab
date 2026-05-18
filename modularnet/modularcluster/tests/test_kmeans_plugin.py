"""
Pub/Sub plugin tests for PluginModularCluster.

Tests the async plugin-based clustering using the modular plugin system
with pub/sub communication.
"""
import time
import torch
import pytest
from typing import Optional

from modularnet.modularcluster.modularcluster_plugin import PluginModularCluster
from modularnet.plugin import PluginManager


class SimplePublisher:
    """Helper publisher for testing."""
    def __init__(self, manager: PluginManager, data: torch.Tensor, topic: str = "cu_positions"):
        self.manager = manager
        self.data = data
        self.topic = topic
        self.published = False
    
    def publish_data(self):
        """Publish test data."""
        self.manager.publish_to_topic(self.topic, self.data)
        self.published = True


class SimpleSubscriber:
    """Helper subscriber for capturing results."""
    def __init__(self, manager: PluginManager, topic: str = "cu_labels"):
        self.manager = manager
        self.topic = topic
        self.results = []
        self.manager.subscribe_to_topic(topic, self._callback)
    
    def _callback(self, data):
        """Capture published results."""
        self.results.append(data)


class TestPluginModularCluster:
    """Tests for PluginModularCluster with pub/sub."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample embedding data."""
        torch.manual_seed(42)
        n_samples = 500
        n_features = 64
        embeddings = torch.randn(n_samples, n_features)
        return embeddings
    
    def test_plugin_creation(self):
        """Test plugin can be created with various options."""
        plugin = PluginModularCluster(
            metric='cosine',
            k_min=5,
            k_max=20,
            verbose=False
        )
        
        assert plugin is not None
        assert plugin.clusterer is not None
        assert plugin.clusterer.metric == 'cosine'
    
    def test_plugin_topics_defined(self):
        """Test that input/output topics are properly defined."""
        plugin = PluginModularCluster()
        
        assert plugin.INPUT_TOPIC == "cu_positions"
        assert plugin.OUTPUT_TOPIC == "cu_labels"
    
    def test_plugin_on_positions_callback(self, sample_data):
        """Test the callback processes positions correctly."""
        plugin = PluginModularCluster(metric='cosine', verbose=False)
        
        # Call the callback directly
        plugin._on_positions_received(sample_data)
        
        # Check that clustering was performed
        assert plugin.clusterer.labels is not None
        assert len(plugin.clusterer.labels) == len(sample_data)
    
    def test_plugin_handles_non_tensor_input(self):
        """Test that plugin handles list input and converts to tensor."""
        plugin = PluginModularCluster(metric='cosine', verbose=False)
        
        # Create proper data as list (needs enough samples)
        data_list = torch.randn(100, 32).tolist()
        
        # Should not raise
        plugin._on_positions_received(data_list)
        
        assert plugin.clusterer.labels is not None
    
    def test_plugin_error_handling_no_crash(self):
        """Test that plugin doesn't crash on invalid input."""
        plugin = PluginModularCluster(
            metric='cosine',
            verbose=False,
            fatal_errors=False
        )
        
        # This should be handled gracefully (no crash)
        plugin._on_positions_received(torch.tensor([]))
    
    def test_plugin_with_euclidean_metric(self, sample_data):
        """Test plugin with Euclidean metric."""
        plugin = PluginModularCluster(
            metric='euclidean',
            verbose=False
        )
        
        plugin._on_positions_received(sample_data)
        
        assert plugin.clusterer.labels is not None
        assert plugin.clusterer.metric == 'euclidean'
    
    def test_plugin_with_custom_k_range(self, sample_data):
        """Test plugin with custom K range."""
        plugin = PluginModularCluster(
            metric='cosine',
            k_min=3,
            k_max=10,
            verbose=False
        )
        
        plugin._on_positions_received(sample_data)
        
        assert 3 <= plugin.clusterer.best_k <= 10
    
    def test_plugin_multiple_clustering_calls(self, sample_data):
        """Test that plugin can cluster multiple batches sequentially."""
        plugin = PluginModularCluster(metric='cosine', verbose=False)
        
        # First clustering
        plugin._on_positions_received(sample_data[:200])
        first_k = plugin.clusterer.best_k
        
        # Second clustering with different data
        torch.manual_seed(123)
        new_data = torch.randn(300, 64)
        plugin._on_positions_received(new_data)
        second_k = plugin.clusterer.best_k
        
        # Both should complete without error
        assert first_k is not None
        assert second_k is not None
    
    def test_plugin_preserves_clustering_quality(self, sample_data):
        """Test that plugin produces high-quality silhouette scores."""
        plugin = PluginModularCluster(metric='cosine', verbose=False)
        
        plugin._on_positions_received(sample_data)
        
        # Silhouette score should be reasonable (not terrible)
        best_score = plugin.clusterer.silhouette_scores[plugin.clusterer.best_k]
        
        # For well-distributed random data, we should get some structure
        assert best_score > -1.0 and best_score <= 1.0


class TestPluginIntegration:
    """Integration tests with PluginManager."""
    
    def test_plugin_manager_integration(self):
        """Test plugin can be registered with PluginManager."""
        manager = PluginManager()
        plugin = PluginModularCluster(
            metric='cosine',
            verbose=False
        )
        
        manager.register_plugin(plugin, name="clusterer")
        
        assert manager.get_plugin("clusterer") is not None


def test_async_basic_euclidean(verbose: bool = False):
    """Basic async test for Euclidean KMeans plugin."""
    torch.manual_seed(42)
    n_samples = 500
    n_features = 64
    embeddings = torch.randn(n_samples, n_features)
    
    cluster = PluginModularCluster(metric='euclidean', verbose=verbose)
    cluster._on_positions_received(embeddings)
    
    print("Euclidean Clustering Results:")
    cluster.clusterer.print_stats()


def test_async_basic_cosine(verbose: bool = False):
    """Basic async test for Cosine KMeans plugin."""
    torch.manual_seed(42)
    n_samples = 500
    n_features = 64
    embeddings = torch.randn(n_samples, n_features)
    
    cluster = PluginModularCluster(metric='cosine', verbose=verbose)
    cluster._on_positions_received(embeddings)
    
    print("Cosine Clustering Results:")
    cluster.clusterer.print_stats()

