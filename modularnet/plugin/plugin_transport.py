"""
Unified transport system for ModularNet plugins.

Provides automatic detection and handling of different data types
across process boundaries, including:
- Pickle for small objects and stateless modules
- Shared memory for modules with parameters
- Shared tensors for CPU tensors (zero-copy)
- CUDA IPC for CUDA tensors
- Signal-only mode for large/unserializable objects
"""

from typing import Optional, Tuple
import ctypes
import torch
import torch.nn as nn
from torch import multiprocessing as mp




# ---------------------------------------------------------------------------
# Transport strategies — auto-detected from value type
# ---------------------------------------------------------------------------

class _Transport:
    """Available transport mechanisms for inter-process data sharing."""
    PICKLE        = "pickle"         # small objects, stateless modules, criterion
    SHARED_MODULE = "shared_module"  # nn.Module with parameters → share_memory_()
    SHARED_TENSOR = "shared_tensor"  # CPU tensor → zero-copy shared memory
    CUDA_IPC      = "cuda_ipc"       # CUDA tensor → CUDA IPC handle
    SIGNAL_ONLY   = "signal_only"    # large/unserializable → notify only, no value


def _classify(value) -> str:
    """
    Automatically detect the transport strategy for a value.
    
    Returns one of: PICKLE, SHARED_MODULE, SHARED_TENSOR, CUDA_IPC, SIGNAL_ONLY
    """
    if isinstance(value, nn.Module):
        if len(list(value.parameters())) > 0:
            return _Transport.SHARED_MODULE
        return _Transport.PICKLE           # stateless module (pure functional)
    if isinstance(value, torch.Tensor):
        if value.is_cuda:
            return _Transport.CUDA_IPC
        return _Transport.SHARED_TENSOR
    return _Transport.PICKLE


# ---------------------------------------------------------------------------
# Delivery modes for tensor channels
# ---------------------------------------------------------------------------

class TensorDeliveryMode:
    """Delivery guarantees for tensor topics."""
    SYNC   = "sync"    # writer blocks until reader consumed — every frame guaranteed
    LATEST = "latest"  # writer never blocks — reader gets newest, may skip frames



# ---------------------------------------------------------------------------
# ActivationTap — forward hook with lazy buffer allocation
# ---------------------------------------------------------------------------

class ActivationTap:
    """
    Attaches to a nn.Module as a forward hook and publishes the output
    activation to a named tensor topic — with zero allocation after the
    very first forward pass.

    The shared buffer is created on the first hook invocation by inferring
    the output shape, so you never need to know layer output sizes upfront.
    The hook itself does only: detach() + copy_() + Event.set() — the
    forward pass is blocked for the absolute minimum time.

    Usage
    -----
        tap    = ActivationTap("layer3_act", manager, mode=TensorDeliveryMode.LATEST)
        handle = model.layer3.register_forward_hook(tap.hook)

        # Later, to detach:
        handle.remove()

    Any plugin subscribed to "layer3_act" will receive the activation
    through its normal TensorChannel callback.
    """

    def __init__(
        self,
        topic_name: str,
        manager: "PluginManagerPubSub",
        dtype: torch.dtype = torch.float32,
        mode: str = TensorDeliveryMode.LATEST,
    ):
        self.topic_name = topic_name
        self.manager    = manager
        self.dtype      = dtype
        self.mode       = mode

    def hook(self, module, input, output):
        """Forward hook — runs in the training process, must return fast."""
        out = output.detach()
        self.manager.publish_tensor_to_topic(
            self.topic_name, out, lazy=True, dtype=self.dtype, mode=self.mode
        )



# ---------------------------------------------------------------------------
# TensorChannel — one writer, one reader, optional lazy buffer allocation
# ---------------------------------------------------------------------------

class TensorChannel:
    """
    A shared-memory slot between exactly one writer and one reader.

    Parameters
    ----------
    shape : Optional[Tuple]
        None  →  lazy mode: the buffer is allocated on the very first write(),
                 inferring shape from the incoming tensor.  After that the
                 shape is fixed forever.  No knowledge of layer output size
                 required at construction time — ideal for forward hooks.
        Tuple →  eager mode: buffer is allocated immediately.  Use when you
                 know the shape upfront (e.g. gradient / batch topics).

    dtype : torch.dtype
        Data type for the tensor buffer. Default: torch.float32

    mode : str
        TensorDeliveryMode.SYNC   — writer blocks until reader calls consumed().
                                    Every frame delivered.
        TensorDeliveryMode.LATEST — writer never blocks; reader always sees the
                                    newest value. May skip frames if reader is slow.

    Notes
    -----
    Hot path (every forward pass): copy_() into pre-allocated shared buffer
    + one Event signal.  No pickle, no allocation after the first write.
    """

    def __init__(
        self,
        shape: Optional[Tuple] = None,
        dtype: torch.dtype = torch.float32,
        mode: str = TensorDeliveryMode.SYNC,
    ):
        self.dtype = dtype
        self.mode  = mode

        # Lock must be created before any process spawns so it lives in
        # shared memory and is visible to child processes.
        self._init_lock = mp.Lock()

        if shape is not None:
            self.shape        = shape
            self.tensor       = torch.zeros(shape, dtype=dtype).share_memory_()
            self._initialized = True
        else:
            self.shape        = None
            self.tensor       = None
            self._initialized = False

        # Sync primitives — always created eagerly so child processes inherit them.
        if mode == TensorDeliveryMode.SYNC:
            self._ready    = mp.Event()
            self._consumed = mp.Event()
            self._consumed.set()         # initially free for writing
        else:
            self._version = mp.Value(ctypes.c_uint64, 0)

    def _lazy_init(self, src: torch.Tensor):
        """Allocate the shared buffer on the very first write (double-checked lock)."""
        if not self._initialized:
            with self._init_lock:
                if not self._initialized:
                    self.shape = tuple(src.shape)
                    self.tensor = torch.zeros(self.shape, dtype=src.dtype).share_memory_()
                    self._initialized = True

    def write(self, src: torch.Tensor) -> bool:
        """
        Copy src into the shared buffer.

        First call in lazy mode allocates the buffer from src.shape.
        Subsequent calls must match that shape exactly.

        SYNC:   blocks until reader has consumed the previous frame.
        LATEST: always overwrites immediately.
        """
        self._lazy_init(src)

        if tuple(src.shape) != self.shape or src.dtype != self.dtype:
            raise ValueError(
                f"TensorChannel shape/dtype mismatch: "
                f"expected {self.shape}/{self.dtype}, "
                f"got {tuple(src.shape)}/{src.dtype}"
            )

        if self.mode == TensorDeliveryMode.SYNC:
            self._consumed.wait()
            self._consumed.clear()
            self.tensor.copy_(src)
            self._ready.set()
        else:
            self.tensor.copy_(src)
            with self._version.get_lock():
                self._version.value += 1

        return True

    def read(self, timeout: Optional[float] = None) -> Optional[torch.Tensor]:
        """
        Read from the shared buffer.

        SYNC:   blocks until new data arrives (or timeout expires).
                Returns a zero-copy view into shared memory.
                Caller MUST call consumed() when done to unblock the writer.
        LATEST: returns immediately.
                Returns None if the buffer has not been written to yet.
        """
        if not self._initialized:
            return None

        if self.mode == TensorDeliveryMode.SYNC:
            if not self._ready.wait(timeout=timeout):
                return None
            self._ready.clear()
            return self.tensor
        else:
            return self.tensor

    def consumed(self):
        """
        SYNC mode: signal that you are done reading, unblocking the writer.
        LATEST mode: no-op.
        """
        if self.mode == TensorDeliveryMode.SYNC:
            self._consumed.set()

    def get_version(self) -> int:
        """LATEST mode: monotonic counter incremented on every write."""
        if self.mode == TensorDeliveryMode.LATEST:
            return self._version.value
        return 0
