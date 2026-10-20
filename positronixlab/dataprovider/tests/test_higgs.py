"""
Comprehensive tests for HIGGS DataProvider and Dataset.

Tests cover:
- Float epoch_size (percentage mode)
- Int epoch_size (absolute mode with percentage propagation)
- Normalization (mean/std computation and application)
- Train/val/test splits
- Random sampling with and without seeding
- Data shapes and class counts
- Dataloader functionality
"""

import pytest
import numpy as np
import torch
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from positronixlab.dataprovider.higgs_dataprovider import HIGGSDataProvider, HIGGSDataset


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_higgs_data(temp_data_dir):
    """Create mock HIGGS data for testing."""
    # Create mock data: 1000 samples, 28 features
    np.random.seed(42)
    n_samples = 1000
    n_features = 28
    
    features = np.random.randn(n_samples, n_features).astype(np.float32)
    labels = np.random.randint(0, 2, n_samples).astype(np.int64)
    
    data = np.column_stack([labels, features])
    
    npy_file = Path(temp_data_dir) / "HIGGS.npy"
    np.save(npy_file, data)
    
    return temp_data_dir, n_samples


class TestHIGGSDatasetFloatEpochSize:
    """Test HIGGS Dataset with float epoch_size (percentage mode)."""
    
    def test_float_epoch_size_default(self, mock_higgs_data):
        """Test default float epoch_size of 0.1 (10%)."""
        temp_dir, total_samples = mock_higgs_data
        
        dataset = HIGGSDataset(
            root=temp_dir,
            epoch_size=0.1,
            normalize=False
        )
        
        # Should be 10% of 1000 = 100
        expected_size = int(0.1 * total_samples)
        assert dataset.epoch_size == expected_size, f"Expected {expected_size}, got {dataset.epoch_size}"
    
    def test_float_epoch_size_custom(self, mock_higgs_data):
        """Test custom float epoch_size."""
        temp_dir, total_samples = mock_higgs_data
        
        dataset = HIGGSDataset(
            root=temp_dir,
            epoch_size=0.25,
            normalize=False
        )
        
        expected_size = int(0.25 * total_samples)
        assert dataset.epoch_size == expected_size
    
    def test_float_epoch_size_len(self, mock_higgs_data):
        """Test __len__ returns epoch_size."""
        temp_dir, _ = mock_higgs_data
        
        dataset = HIGGSDataset(
            root=temp_dir,
            epoch_size=0.2,
            normalize=False
        )
        
        assert len(dataset) == dataset.epoch_size
    
    def test_float_epoch_size_invalid_high(self, mock_higgs_data):
        """Test that epoch_size > 1.0 raises error."""
        temp_dir, _ = mock_higgs_data
        
        with pytest.raises(ValueError, match="epoch_size as float must be"):
            HIGGSDataset(
                root=temp_dir,
                epoch_size=1.5,
                normalize=False
            )
    
    def test_float_epoch_size_invalid_zero(self, mock_higgs_data):
        """Test that epoch_size <= 0 raises error."""
        temp_dir, _ = mock_higgs_data
        
        with pytest.raises(ValueError, match="epoch_size as float must be"):
            HIGGSDataset(
                root=temp_dir,
                epoch_size=0.0,
                normalize=False
            )


class TestHIGGSDatasetIntEpochSize:
    """Test HIGGS Dataset with int epoch_size (absolute mode)."""
    
    def test_int_epoch_size_exact(self, mock_higgs_data):
        """Test int epoch_size returns exact sample count."""
        temp_dir, _ = mock_higgs_data
        
        dataset = HIGGSDataset(
            root=temp_dir,
            epoch_size=100,
            normalize=False
        )
        
        assert dataset.epoch_size == 100
    
    def test_int_epoch_size_clipped_to_dataset_size(self, mock_higgs_data):
        """Test int epoch_size clamped to dataset size."""
        temp_dir, total_samples = mock_higgs_data
        
        dataset = HIGGSDataset(
            root=temp_dir,
            epoch_size=5000,  # More than available
            normalize=False
        )
        
        # Should be clipped to dataset size
        assert dataset.epoch_size <= total_samples
    
    def test_int_epoch_size_len(self, mock_higgs_data):
        """Test __len__ with int epoch_size."""
        temp_dir, _ = mock_higgs_data
        
        dataset = HIGGSDataset(
            root=temp_dir,
            epoch_size=250,
            normalize=False
        )
        
        assert len(dataset) == 250


class TestHIGGSDatasetRandomSampling:
    """Test random sampling behavior with and without seeds."""
    
    def test_sampling_without_seed_varies(self, mock_higgs_data):
        """Test that sampling without seed produces different samples."""
        temp_dir, _ = mock_higgs_data
        
        dataset1 = HIGGSDataset(
            root=temp_dir,
            epoch_size=100,
            normalize=False
        )
        dataset1._sample_epoch(seed=None)
        indices1 = dataset1.current_epoch_indices.copy()
        
        dataset2 = HIGGSDataset(
            root=temp_dir,
            epoch_size=100,
            normalize=False
        )
        dataset2._sample_epoch(seed=None)
        indices2 = dataset2.current_epoch_indices.copy()
        
        # Different random seeds should produce different indices (with high probability)
        # We can't guarantee they're different, but overlap should be small
        overlap = len(np.intersect1d(indices1, indices2))
        assert overlap < 90, "Samples with different seeds should have minimal overlap"
    
    def test_sampling_with_same_seed_identical(self, mock_higgs_data):
        """Test that same seed produces identical samples."""
        temp_dir, _ = mock_higgs_data
        
        dataset1 = HIGGSDataset(
            root=temp_dir,
            epoch_size=100,
            normalize=False
        )
        dataset1._sample_epoch(seed=42)
        indices1 = dataset1.current_epoch_indices.copy()
        
        dataset2 = HIGGSDataset(
            root=temp_dir,
            epoch_size=100,
            normalize=False
        )
        dataset2._sample_epoch(seed=42)
        indices2 = dataset2.current_epoch_indices.copy()
        
        # Same seed should produce identical indices
        assert np.array_equal(indices1, indices2), "Same seed should produce identical indices"
    
    def test_sampling_different_seeds_different(self, mock_higgs_data):
        """Test that different seeds produce different samples."""
        temp_dir, _ = mock_higgs_data
        
        dataset = HIGGSDataset(
            root=temp_dir,
            epoch_size=100,
            normalize=False
        )
        
        dataset._sample_epoch(seed=42)
        indices1 = dataset.current_epoch_indices.copy()
        
        dataset._sample_epoch(seed=99)
        indices2 = dataset.current_epoch_indices.copy()
        
        assert not np.array_equal(indices1, indices2), "Different seeds should produce different indices"


class TestHIGGSDatasetNormalization:
    """Test normalization computation and application."""
    
    def test_compute_normalization(self, mock_higgs_data):
        """Test normalization statistics computation."""
        temp_dir, _ = mock_higgs_data
        
        dataset = HIGGSDataset(
            root=temp_dir,
            epoch_size=0.5,
            normalize=False
        )
        dataset.compute_normalization()
        
        assert dataset.mean is not None
        assert dataset.std is not None
        assert dataset.mean.shape == (28,)
        assert dataset.std.shape == (28,)
    
    def test_apply_normalization(self, mock_higgs_data):
        """Test applying precomputed normalization."""
        temp_dir, _ = mock_higgs_data
        
        dataset = HIGGSDataset(
            root=temp_dir,
            epoch_size=0.5,
            normalize=False
        )
        dataset.compute_normalization()
        
        mean = dataset.mean.clone()
        std = dataset.std.clone()
        
        dataset2 = HIGGSDataset(
            root=temp_dir,
            epoch_size=0.5,
            normalize=False
        )
        dataset2.apply_normalization(mean, std)
        
        assert torch.allclose(dataset2.mean, mean)
        assert torch.allclose(dataset2.std, std)
    
    def test_normalization_values_reasonable(self, mock_higgs_data):
        """Test that normalized data has reasonable mean and std."""
        temp_dir, _ = mock_higgs_data
        
        dataset = HIGGSDataset(
            root=temp_dir,
            epoch_size=1.0,
            normalize=True
        )
        dataset.compute_normalization()
        
        # Sample data to access normalized values
        dataset._sample_epoch(seed=42)
        
        # After normalization, mean and std of sampled data should be reasonable
        # (not exact 0 and 1 due to random sampling)
        sampled_mean = dataset.x.mean(dim=0)
        sampled_std = dataset.x.std(dim=0)
        
        # Mean should be close to 0 (within reasonable tolerance for sampled data)
        assert torch.allclose(sampled_mean, torch.zeros(28), atol=0.1)
        # Std should be close to 1
        assert torch.allclose(sampled_std, torch.ones(28), atol=0.1)


class TestHIGGSDatasetGetItem:
    """Test __getitem__ functionality."""
    
    def test_getitem_returns_tuple(self, mock_higgs_data):
        """Test that __getitem__ returns (x, y) tuple."""
        temp_dir, _ = mock_higgs_data
        
        dataset = HIGGSDataset(
            root=temp_dir,
            epoch_size=0.1,
            normalize=False
        )
        
        x, y = dataset[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
    
    def test_getitem_feature_shape(self, mock_higgs_data):
        """Test that features have correct shape (28,)."""
        temp_dir, _ = mock_higgs_data
        
        dataset = HIGGSDataset(
            root=temp_dir,
            epoch_size=0.1,
            normalize=False
        )
        
        x, y = dataset[0]
        assert x.shape == (28,)
    
    def test_getitem_label_binary(self, mock_higgs_data):
        """Test that labels are binary (0 or 1)."""
        temp_dir, _ = mock_higgs_data
        
        dataset = HIGGSDataset(
            root=temp_dir,
            epoch_size=0.1,
            normalize=False
        )
        
        for i in range(min(10, len(dataset))):
            _, y = dataset[i]
            assert y in [0, 1], f"Label should be 0 or 1, got {y}"


class TestHIGGSDataProvider:
    """Test HIGGSDataProvider (high-level API)."""
    
    def test_provider_float_epoch_size(self, mock_higgs_data):
        """Test provider with float epoch_size."""
        temp_dir, total_samples = mock_higgs_data
        
        provider = HIGGSDataProvider(
            batch_size=32,
            epoch_size=0.1,
            root=temp_dir,
            normalize=False,
            num_workers=0
        )
        
        # Each split should have 10% of its size
        train_expected = int(0.1 * len(provider.train_indices))
        val_expected = int(0.1 * len(provider.val_indices))
        
        assert provider.dataset_train.epoch_size == train_expected
        assert provider.dataset_val.epoch_size == val_expected
    
    def test_provider_int_epoch_size_propagation(self, mock_higgs_data):
        """Test provider with int epoch_size propagates percentage."""
        temp_dir, total_samples = mock_higgs_data
        
        epoch_size_train = 200  # int > 1
        
        provider = HIGGSDataProvider(
            batch_size=32,
            epoch_size=epoch_size_train,
            root=temp_dir,
            normalize=False,
            num_workers=0
        )
        
        # Compute expected percentage and propagated sizes
        percentage = epoch_size_train / len(provider.train_indices)
        val_expected = int(percentage * len(provider.val_indices))
        
        assert provider.dataset_train.epoch_size == epoch_size_train
        assert provider.dataset_val.epoch_size == val_expected
    
    def test_provider_dataset_properties(self, mock_higgs_data):
        """Test provider sets correct dataset properties."""
        temp_dir, _ = mock_higgs_data
        
        provider = HIGGSDataProvider(
            batch_size=32,
            epoch_size=0.1,
            root=temp_dir,
            normalize=False,
            num_workers=0
        )
        
        assert provider.name == 'higgs'
        assert provider.num_features == 28
        assert provider.num_classes == 2
        assert provider.x_shape == (28,)
        assert provider.y_shape == ()
    
    def test_provider_has_dataloaders(self, mock_higgs_data):
        """Test provider creates train/val/test dataloaders."""
        temp_dir, _ = mock_higgs_data
        
        provider = HIGGSDataProvider(
            batch_size=32,
            epoch_size=0.2,
            root=temp_dir,
            normalize=False,
            num_workers=0
        )
        
        assert provider.train is not None
        assert provider.val is not None
        assert provider.test is not None
    
    def test_provider_dataloader_batches(self, mock_higgs_data):
        """Test dataloaders return correct batch shapes."""
        temp_dir, _ = mock_higgs_data
        
        provider = HIGGSDataProvider(
            batch_size=16,
            epoch_size=0.3,
            root=temp_dir,
            normalize=False,
            num_workers=0
        )
        
        # Get first batch from train
        for batch_x, batch_y in provider.train:
            assert batch_x.shape[0] <= 16
            assert batch_x.shape[1] == 28
            assert batch_y.shape[0] == batch_x.shape[0]
            break


class TestHIGGSDataProviderSplits:
    """Test train/val/test splits."""
    
    def test_three_way_split_default(self, mock_higgs_data):
        """Test default three-way split (0.7, 0.15, 0.15)."""
        temp_dir, total_samples = mock_higgs_data
        
        provider = HIGGSDataProvider(
            batch_size=32,
            split=[0.7, 0.15, 0.15],
            epoch_size=1.0,
            root=temp_dir,
            normalize=False,
            num_workers=0
        )
        
        train_size = len(provider.train_indices)
        val_size = len(provider.val_indices)
        test_size = len(provider.test_indices)
        
        assert train_size == int(0.7 * total_samples)
        assert val_size == int(0.15 * total_samples)
        assert test_size == total_samples - train_size - val_size
    
    def test_two_way_split(self, mock_higgs_data):
        """Test two-way split (no test set)."""
        temp_dir, total_samples = mock_higgs_data
        
        provider = HIGGSDataProvider(
            batch_size=32,
            split=[0.8, 0.2],
            epoch_size=0.1,
            root=temp_dir,
            normalize=False,
            num_workers=0
        )
        
        assert len(provider.train_indices) == int(0.8 * total_samples)
        assert len(provider.val_indices) == int(0.2 * total_samples)
        assert provider.dataset_test is None
    
    def test_splits_no_overlap(self, mock_higgs_data):
        """Test that splits have no overlapping indices."""
        temp_dir, _ = mock_higgs_data
        
        provider = HIGGSDataProvider(
            batch_size=32,
            split=[0.7, 0.15, 0.15],
            epoch_size=1.0,
            root=temp_dir,
            normalize=False,
            num_workers=0
        )
        
        train_set = set(provider.train_indices)
        val_set = set(provider.val_indices)
        test_set = set(provider.test_indices)
        
        assert len(train_set & val_set) == 0, "Train and val overlap"
        assert len(train_set & test_set) == 0, "Train and test overlap"
        assert len(val_set & test_set) == 0, "Val and test overlap"


class TestHIGGSDataProviderNormalization:
    """Test provider-level normalization."""
    
    def test_provider_normalization_enabled(self, mock_higgs_data):
        """Test provider computes and applies normalization."""
        temp_dir, _ = mock_higgs_data
        
        provider = HIGGSDataProvider(
            batch_size=32,
            epoch_size=0.5,
            root=temp_dir,
            normalize=True,
            num_workers=0
        )
        
        # All datasets should have normalization stats
        assert provider.dataset_train.mean is not None
        assert provider.dataset_val.mean is not None
        assert provider.dataset_test.mean is not None
    
    def test_provider_normalization_from_train(self, mock_higgs_data):
        """Test that val/test use train's normalization stats."""
        temp_dir, _ = mock_higgs_data
        
        provider = HIGGSDataProvider(
            batch_size=32,
            epoch_size=0.5,
            root=temp_dir,
            normalize=True,
            num_workers=0
        )
        
        # Val and test should use train's stats
        assert torch.allclose(provider.dataset_val.mean, provider.dataset_train.mean)
        assert torch.allclose(provider.dataset_val.std, provider.dataset_train.std)
        assert torch.allclose(provider.dataset_test.mean, provider.dataset_train.mean)
        assert torch.allclose(provider.dataset_test.std, provider.dataset_train.std)
    
    def test_provider_normalization_disabled(self, mock_higgs_data):
        """Test provider without normalization."""
        temp_dir, _ = mock_higgs_data
        
        provider = HIGGSDataProvider(
            batch_size=32,
            epoch_size=0.5,
            root=temp_dir,
            normalize=False,
            num_workers=0
        )
        
        assert provider.dataset_train.mean is None
        assert provider.dataset_val.mean is None


class TestHIGGSFeatureGroups:
    """Test feature group information."""
    
    def test_get_feature_groups(self, mock_higgs_data):
        """Test get_feature_groups returns correct ranges."""
        temp_dir, _ = mock_higgs_data
        
        provider = HIGGSDataProvider(
            batch_size=32,
            epoch_size=0.1,
            root=temp_dir,
            normalize=False,
            num_workers=0
        )
        
        groups = provider.get_feature_groups()
        
        assert 'low_level' in groups
        assert 'high_level' in groups
        assert groups['low_level'] == list(range(21))
        assert groups['high_level'] == list(range(21, 28))


class TestHIGGSEdgeCases:
    """Test edge cases and error handling."""
    
    def test_invalid_epoch_size_type(self, mock_higgs_data):
        """Test that invalid epoch_size type raises error."""
        temp_dir, _ = mock_higgs_data
        
        with pytest.raises(TypeError, match="epoch_size must be int or float"):
            HIGGSDataset(
                root=temp_dir,
                epoch_size="invalid",
                normalize=False
            )
    
    def test_batch_size_smaller_than_epoch_size(self, mock_higgs_data):
        """Test dataloader with batch_size < epoch_size."""
        temp_dir, _ = mock_higgs_data
        
        provider = HIGGSDataProvider(
            batch_size=10,
            epoch_size=100,
            root=temp_dir,
            normalize=False,
            num_workers=0
        )
        
        # Should still work, just produce smaller batches
        batches = list(provider.train)
        assert len(batches) > 0
    
    def test_batch_size_larger_than_epoch_size(self, mock_higgs_data):
        """Test dataloader with batch_size > epoch_size."""
        temp_dir, _ = mock_higgs_data
        
        provider = HIGGSDataProvider(
            batch_size=200,
            epoch_size=50,
            root=temp_dir,
            normalize=False,
            num_workers=0
        )
        
        # Should handle gracefully (drop_last=True in dataloader)
        batches = list(provider.train)
        # May be empty if drop_last discards undersized batches
        assert isinstance(batches, list)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
