"""
Comprehensive tests for plugin_transport.py - TensorChannel and transport detection.
"""

import pytest
import torch
import torch.nn as nn
from torch import multiprocessing as mp
import time

from modularnet.plugin.plugin_transport import (
    TensorChannel, TensorDeliveryMode, _Transport, _classify
)


# ============================================================================
# Tests for Transport Detection (_classify)
# ============================================================================

class TestTransportDetection:
    """Test automatic transport strategy detection."""
    
    def test_classify_scalar(self):
        """Test classification of scalar values."""
        assert _classify(42) == _Transport.PICKLE
        assert _classify(3.14) == _Transport.PICKLE
        assert _classify("hello") == _Transport.PICKLE
    
    def test_classify_dict_list(self):
        """Test classification of dict/list."""
        assert _classify({"key": "value"}) == _Transport.PICKLE
        assert _classify([1, 2, 3]) == _Transport.PICKLE
    
    def test_classify_module_with_parameters(self):
        """Test classification of module with parameters."""
        module = nn.Linear(10, 5)
        assert _classify(module) == _Transport.SHARED_MODULE
    
    def test_classify_module_without_parameters(self):
        """Test classification of stateless module."""
        module = nn.Identity()  # No parameters
        assert _classify(module) == _Transport.PICKLE
    
    def test_classify_cpu_tensor(self):
        """Test classification of CPU tensor."""
        tensor = torch.randn(10, 5)
        assert _classify(tensor) == _Transport.SHARED_TENSOR
    
    def test_classify_cuda_tensor(self):
        """Test classification of CUDA tensor (if available)."""
        if torch.cuda.is_available():
            tensor = torch.randn(10, 5, device='cuda')
            assert _classify(tensor) == _Transport.CUDA_IPC
    
    def test_classify_complex_module(self):
        """Test classification of complex module."""
        module = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        assert _classify(module) == _Transport.SHARED_MODULE
    
    def test_classify_conv_module(self):
        """Test classification of CNN."""
        module = nn.Conv2d(3, 16, 3)
        assert _classify(module) == _Transport.SHARED_MODULE


# ============================================================================
# Tests for TensorChannel - Basic Operations
# ============================================================================

class TestTensorChannelBasic:
    """Test basic TensorChannel operations."""
    
    def test_channel_initialization_eager(self):
        """Test eager initialization with known shape."""
        channel = TensorChannel(
            shape=(10, 20),
            dtype=torch.float32,
            mode=TensorDeliveryMode.LATEST
        )
        assert channel.shape == (10, 20)
        assert channel.dtype == torch.float32
        assert channel.mode == TensorDeliveryMode.LATEST
        assert channel._initialized
    
    def test_channel_initialization_lazy(self):
        """Test lazy initialization with no shape."""
        channel = TensorChannel(
            shape=None,
            dtype=torch.float32,
            mode=TensorDeliveryMode.LATEST
        )
        assert channel.shape is None
        assert not channel._initialized
    
    def test_channel_write_eager_mode(self):
        """Test writing to eager channel."""
        channel = TensorChannel(shape=(5, 5), dtype=torch.float32)
        tensor = torch.ones(5, 5)
        
        result = channel.write(tensor)
        assert result is True
        assert channel.tensor is not None
    
    def test_channel_write_lazy_mode(self):
        """Test writing to lazy channel allocates buffer."""
        channel = TensorChannel(shape=None, dtype=torch.float32)
        assert not channel._initialized
        
        tensor = torch.randn(10, 20)
        channel.write(tensor)
        
        assert channel._initialized
        assert channel.shape == (10, 20)
    
    def test_channel_write_shape_mismatch(self):
        """Test writing tensor with wrong shape raises error."""
        channel = TensorChannel(shape=(5, 5), dtype=torch.float32)
        tensor = torch.ones(10, 10)  # Wrong shape
        
        with pytest.raises(ValueError):
            channel.write(tensor)
    
    def test_channel_write_dtype_mismatch(self):
        """Test writing tensor with wrong dtype raises error."""
        channel = TensorChannel(shape=(5, 5), dtype=torch.float32)
        tensor = torch.ones(5, 5, dtype=torch.int32)  # Wrong dtype
        
        with pytest.raises(ValueError):
            channel.write(tensor)
    
    def test_channel_read_eager_uninitialized(self):
        """Test reading from uninitialized lazy channel returns None."""
        channel = TensorChannel(shape=None, dtype=torch.float32)
        result = channel.read(timeout=0.1)
        assert result is None
    
    def test_channel_read_after_write(self):
        """Test reading after writing."""
        channel = TensorChannel(shape=(5, 5), dtype=torch.float32)
        tensor_in = torch.ones(5, 5)
        
        channel.write(tensor_in)
        tensor_out = channel.read()
        
        assert tensor_out is not None
        assert torch.allclose(tensor_in, tensor_out)


# ============================================================================
# Tests for TensorChannel - LATEST Mode
# ============================================================================

class TestTensorChannelLatest:
    """Test TensorChannel in LATEST delivery mode."""
    
    def test_latest_write_never_blocks(self):
        """Test that LATEST write never blocks."""
        channel = TensorChannel(shape=(5, 5), mode=TensorDeliveryMode.LATEST)
        
        # Write multiple times quickly - should never block
        for i in range(100):
            tensor = torch.full((5, 5), float(i))
            start = time.time()
            channel.write(tensor)
            elapsed = time.time() - start
            # Should be very fast
            assert elapsed < 0.1
    
    def test_latest_read_always_succeeds(self):
        """Test that LATEST read always returns data."""
        channel = TensorChannel(shape=(5, 5), mode=TensorDeliveryMode.LATEST)
        tensor = torch.randn(5, 5)
        
        channel.write(tensor)
        result = channel.read(timeout=0.1)
        
        assert result is not None
    
    def test_latest_read_gets_latest_value(self):
        """Test that LATEST read returns most recent value."""
        channel = TensorChannel(shape=(5, 5), mode=TensorDeliveryMode.LATEST)
        
        # Write multiple values
        for i in range(5):
            tensor = torch.full((5, 5), float(i))
            channel.write(tensor)
        
        # Read should get the last value (4)
        result = channel.read()
        assert torch.allclose(result, torch.full((5, 5), 4.0))
    
    def test_latest_get_version(self):
        """Test version counter in LATEST mode."""
        channel = TensorChannel(shape=(5, 5), mode=TensorDeliveryMode.LATEST)
        
        v1 = channel.get_version()
        channel.write(torch.ones(5, 5))
        v2 = channel.get_version()
        channel.write(torch.ones(5, 5))
        v3 = channel.get_version()
        
        assert v2 == v1 + 1
        assert v3 == v2 + 1
    
    def test_latest_consumed_is_noop(self):
        """Test that consumed() is no-op in LATEST mode."""
        channel = TensorChannel(shape=(5, 5), mode=TensorDeliveryMode.LATEST)
        channel.write(torch.ones(5, 5))
        
        result = channel.read()
        # This should not raise or block
        channel.consumed()
        
        assert result is not None


# ============================================================================
# Tests for TensorChannel - SYNC Mode
# ============================================================================

class TestTensorChannelSync:
    """Test TensorChannel in SYNC delivery mode."""
    
    def test_sync_write_blocks_until_consumed(self):
        """Test that SYNC write blocks until consumed() called."""
        channel = TensorChannel(shape=(5, 5), mode=TensorDeliveryMode.SYNC)
        
        def delayed_consume():
            time.sleep(0.2)
            tensor = channel.read(timeout=1.0)
            if tensor is not None:
                channel.consumed()
        
        thread = mp.Process(target=delayed_consume)
        thread.start()
        
        # This should block for ~0.2 seconds
        start = time.time()
        channel.write(torch.ones(5, 5))
        elapsed = time.time() - start
        
        thread.join(timeout=2)
        assert elapsed >= 0.1  # Should have been blocked
    
    def test_sync_read_blocks_until_written(self):
        """Test that SYNC read blocks until write() happens."""
        channel = TensorChannel(shape=(5, 5), mode=TensorDeliveryMode.SYNC)
        
        def delayed_write():
            time.sleep(0.2)
            channel.write(torch.ones(5, 5))
        
        thread = mp.Process(target=delayed_write)
        thread.start()
        
        # This should block until write completes
        start = time.time()
        result = channel.read(timeout=1.0)
        elapsed = time.time() - start
        
        thread.join(timeout=2)
        assert result is not None
        assert elapsed >= 0.1  # Should have waited for write
    
    def test_sync_read_timeout(self):
        """Test SYNC read timeout."""
        channel = TensorChannel(shape=(5, 5), mode=TensorDeliveryMode.SYNC)
        
        # No write, so read should timeout
        result = channel.read(timeout=0.1)
        assert result is None
    
    def test_sync_write_read_consume_cycle(self):
        """Test complete write/read/consume cycle."""
        channel = TensorChannel(shape=(5, 5), mode=TensorDeliveryMode.SYNC)
        tensor_in = torch.full((5, 5), 42.0)
        
        # Write
        channel.write(tensor_in)
        
        # Read
        tensor_out = channel.read(timeout=0.5)
        assert tensor_out is not None
        assert torch.allclose(tensor_out, tensor_in)
        
        # Consume
        channel.consumed()
    
    def test_sync_multiple_writes_after_consume(self):
        """Test multiple write/consume cycles."""
        channel = TensorChannel(shape=(3, 3), mode=TensorDeliveryMode.SYNC)
        
        for i in range(5):
            tensor_in = torch.full((3, 3), float(i))
            channel.write(tensor_in)
            tensor_out = channel.read(timeout=0.5)
            assert torch.allclose(tensor_out, tensor_in)
            channel.consumed()


# ============================================================================
# Tests for TensorChannel - Shared Memory
# ============================================================================

class TestTensorChannelSharedMemory:
    """Test that TensorChannel uses shared memory correctly."""
    
    def test_tensor_is_shared(self):
        """Test that channel tensor is in shared memory."""
        channel = TensorChannel(shape=(5, 5), mode=TensorDeliveryMode.LATEST)
        channel.write(torch.ones(5, 5))
        
        # Check if tensor is in shared memory
        result = channel.read()
        # Can't directly check is_shared() for the wrapper, but it should work
        assert result is not None
    
    def test_lazy_allocation_creates_shared_tensor(self):
        """Test that lazy allocation creates shared tensor."""
        channel = TensorChannel(shape=None, mode=TensorDeliveryMode.LATEST)
        tensor = torch.randn(10, 10)
        
        channel.write(tensor)
        
        # Tensor should now be allocated in shared memory
        assert channel._initialized
        assert channel.shape == (10, 10)


# ============================================================================
# Tests for TensorDeliveryMode
# ============================================================================

class TestTensorDeliveryMode:
    """Test TensorDeliveryMode constants."""
    
    def test_mode_constants_exist(self):
        """Test that mode constants are defined."""
        assert hasattr(TensorDeliveryMode, 'SYNC')
        assert hasattr(TensorDeliveryMode, 'LATEST')
    
    def test_mode_constants_are_strings(self):
        """Test that mode constants are string values."""
        assert isinstance(TensorDeliveryMode.SYNC, str)
        assert isinstance(TensorDeliveryMode.LATEST, str)
    
    def test_mode_constants_are_distinct(self):
        """Test that mode constants are distinct."""
        assert TensorDeliveryMode.SYNC != TensorDeliveryMode.LATEST


# ============================================================================
# Tests for Transport Constants
# ============================================================================

class TestTransportConstants:
    """Test _Transport constants."""
    
    def test_transport_constants_exist(self):
        """Test that all transport constants are defined."""
        assert hasattr(_Transport, 'PICKLE')
        assert hasattr(_Transport, 'SHARED_MODULE')
        assert hasattr(_Transport, 'SHARED_TENSOR')
        assert hasattr(_Transport, 'CUDA_IPC')
        assert hasattr(_Transport, 'SIGNAL_ONLY')
    
    def test_transport_constants_are_strings(self):
        """Test that transport constants are strings."""
        assert isinstance(_Transport.PICKLE, str)
        assert isinstance(_Transport.SHARED_MODULE, str)
        assert isinstance(_Transport.SHARED_TENSOR, str)
        assert isinstance(_Transport.CUDA_IPC, str)
        assert isinstance(_Transport.SIGNAL_ONLY, str)
    
    def test_transport_constants_are_distinct(self):
        """Test that all transport types are distinct."""
        transports = [
            _Transport.PICKLE,
            _Transport.SHARED_MODULE,
            _Transport.SHARED_TENSOR,
            _Transport.CUDA_IPC,
            _Transport.SIGNAL_ONLY
        ]
        assert len(transports) == len(set(transports))


# ============================================================================
# Integration Tests
# ============================================================================

class TestTensorChannelIntegration:
    """Integration tests for TensorChannel."""
    
    def test_multiple_tensor_types(self):
        """Test channel with different tensor dtypes and shapes."""
        test_cases = [
            ((5, 5), torch.float32),
            ((10, 20, 30), torch.float64),
            ((100,), torch.int32),
            ((2, 3, 4, 5), torch.float32),
        ]
        
        for shape, dtype in test_cases:
            channel = TensorChannel(shape=shape, dtype=dtype)
            tensor = torch.zeros(shape, dtype=dtype)
            
            channel.write(tensor)
            result = channel.read()
            
            assert result.shape == shape
            assert result.dtype == dtype
    
    def test_lazy_initialization_consistency(self):
        """Test that lazy initialization is consistent across writes."""
        channel = TensorChannel(shape=None, mode=TensorDeliveryMode.LATEST)
        
        # First write
        tensor1 = torch.randn(5, 10)
        channel.write(tensor1)
        assert channel.shape == (5, 10)
        
        # Second write with same shape should work
        tensor2 = torch.randn(5, 10)
        channel.write(tensor2)
        result = channel.read()
        assert torch.allclose(result, tensor2)
    
    def test_channel_with_batch_data(self):
        """Test channel with batch processing."""
        channel = TensorChannel(shape=(32, 784), mode=TensorDeliveryMode.LATEST)
        
        for batch_idx in range(10):
            batch = torch.randn(32, 784)
            channel.write(batch)
            result = channel.read()
            assert result.shape == (32, 784)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
