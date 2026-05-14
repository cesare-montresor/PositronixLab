import ctypes
from typing import Callable, Dict, Optional, Tuple

import torch
from torch import multiprocessing as mp

from draft.plugin_pubsub_old import PluginManagerPubSub, PluginPubSub


# ---------------------------------------------------------------------------
# Delivery modes
# ---------------------------------------------------------------------------

class TensorDeliveryMode:
    SYNC   = "sync"    # writer blocks until reader consumed — every frame guaranteed
    LATEST = "latest"  # writer never blocks — reader gets newest, may skip frames


# ---------------------------------------------------------------------------
# TensorChannel — one writer, one reader, optional lazy buffer allocation
# ---------------------------------------------------------------------------

class TensorChannel:
    """
    A shared-memory slot between exactly one writer and one reader.

    shape=None  →  lazy mode: the buffer is allocated on the very first write(),
                   inferring shape from the incoming tensor.  After that the
                   shape is fixed forever.  No knowledge of layer output size
                   required at construction time — ideal for forward hooks.

    shape=Tuple →  eager mode: buffer is allocated immediately.  Use when you
                   know the shape upfront (e.g. gradient / batch topics).

    SYNC   — writer blocks until reader calls consumed().  Every frame delivered.
    LATEST — writer never blocks; reader always sees the newest value.
             Returns None if the buffer has not been written to yet.

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
            self._ready    = mp.Event()  # set by writer  → reader may read
            self._consumed = mp.Event()  # set by reader  → writer may write
            self._consumed.set()         # initially free for writing
        else:
            self._version = mp.Value(ctypes.c_uint64, 0)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _lazy_init(self, src: torch.Tensor):
        """Allocate the shared buffer on the very first write (double-checked lock)."""
        if not self._initialized:
            with self._init_lock:
                if not self._initialized:           # second check inside the lock
                    self.shape        = tuple(src.shape)
                    self.tensor       = torch.zeros(
                        src.shape, dtype=self.dtype
                    ).share_memory_()
                    self._initialized = True

    # ------------------------------------------------------------------
    # Writer side
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Reader side
    # ------------------------------------------------------------------

    def read(self, timeout: Optional[float] = None) -> Optional[torch.Tensor]:
        """
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
        manager: "PluginManagerPubSubTensor",
        dtype: torch.dtype = torch.float32,
        mode: str = TensorDeliveryMode.LATEST,  # hooks rarely need every frame
    ):
        self.topic_name = topic_name
        self.manager    = manager
        self.dtype      = dtype
        self.mode       = mode

    def hook(self, module, input, output):
        """Forward hook — runs in the training process, must return fast."""
        out = output.detach()
        # lazy=True → topic and channels created on first call if absent
        self.manager.publish_tensor_to_topic(
            self.topic_name, out, lazy=True, dtype=self.dtype, mode=self.mode
        )


# ---------------------------------------------------------------------------
# TensorTopic — one TensorChannel per subscriber
# ---------------------------------------------------------------------------

class TensorTopic:
    """
    A named shared-memory topic.  One TensorChannel per subscriber.

    shape=None → lazy: each channel allocates its buffer on the first publish().
    shape=Tuple → eager: buffers allocated immediately at add_subscriber() time.
    """

    def __init__(
        self,
        name: str,
        dtype: torch.dtype,
        mode: str,
        shape: Optional[Tuple] = None,
    ):
        self.name  = name
        self.shape = shape      # None → lazy
        self.dtype = dtype
        self.mode  = mode
        self.channels: Dict[str, TensorChannel] = {}

    def add_subscriber(self, plugin_name: str):
        if plugin_name not in self.channels:
            # shape=None → channel lazy-inits on first write
            self.channels[plugin_name] = TensorChannel(self.shape, self.dtype, self.mode)

    def remove_subscriber(self, plugin_name: str):
        self.channels.pop(plugin_name, None)

    def get_channel(self, plugin_name: str) -> Optional[TensorChannel]:
        return self.channels.get(plugin_name)

    def publish(self, src: torch.Tensor):
        """copy_() into every subscriber's buffer. Lazy channels self-init here."""
        for channel in self.channels.values():
            channel.write(src)


# ---------------------------------------------------------------------------
# PluginPubSubTensor — drop-in extension of PluginPubSub
# ---------------------------------------------------------------------------

class PluginPubSubTensor(PluginPubSub):
    """
    Extends PluginPubSub with zero-copy tensor pub/sub via shared memory.
    Regular (non-tensor) topics from PluginPubSub work unchanged.

    Quick start
    -----------
        class AnalyserPlugin(PluginPubSubTensor):
            def onStart(self):
                # shape omitted → lazy, inferred on first received tensor
                self.subscribe_tensor(
                    "layer3_act", self.on_activation,
                    mode=TensorDeliveryMode.LATEST,
                )

            def on_activation(self, tensor: torch.Tensor):
                # tensor is a zero-copy view into shared memory
                print(tensor.mean(), tensor.std())

        class TrainerPlugin(PluginPubSubTensor):
            def onStart(self):
                self.publish_tensor_topic("gradients", shape=(256, 512))

            def onUpdate(self):
                self.publish_tensor("gradients", self.model.grad)

        # Attach hooks directly via ActivationTap — no shape needed
        tap = ActivationTap("layer3_act", manager)
        model.layer3.register_forward_hook(tap.hook)
    """

    def __init__(self, options: dict = {}, name: str = None, fatal_errors: bool = False):
        super().__init__(options, name, fatal_errors)

        # topic_name → (shape|None, dtype, mode)
        self._tensor_sub_specs:  Dict[str, Tuple] = {}
        # topic_name → (shape|None, dtype, mode)
        self._tensor_pub_specs:  Dict[str, Tuple] = {}
        # topic_name → callback
        self._tensor_callbacks:  Dict[str, Callable] = {}
        # topic_name → TensorChannel  (bound after process starts)
        self._tensor_channels:   Dict[str, TensorChannel] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def subscribe_tensor(
        self,
        topic_name: str,
        callback: Callable,
        shape: Optional[Tuple] = None,          # None → lazy
        dtype: torch.dtype = torch.float32,
        mode: str = TensorDeliveryMode.LATEST,
    ):
        """
        Declare a tensor subscription.  Call before register_plugin().

        shape=None → buffer allocated on the first received tensor (lazy).
        callback(tensor) is invoked in the receiver thread.
        consumed() is called automatically after the callback in SYNC mode.
        """
        self._tensor_sub_specs[topic_name] = (shape, dtype, mode)
        self._tensor_callbacks[topic_name] = callback

    def publish_tensor_topic(
        self,
        topic_name: str,
        shape: Optional[Tuple] = None,
        dtype: torch.dtype = torch.float32,
        mode: str = TensorDeliveryMode.SYNC,
    ):
        """Declare a tensor topic this plugin will publish to."""
        self._tensor_pub_specs[topic_name] = (shape, dtype, mode)

    def publish_tensor(self, topic_name: str, tensor: torch.Tensor):
        """Publish a tensor — zero-copy write into pre-allocated shared buffers."""
        if self.manager and isinstance(self.manager, PluginManagerPubSubTensor):
            self.manager.publish_tensor_to_topic(topic_name, tensor)
        else:
            raise RuntimeError("publish_tensor requires a PluginManagerPubSubTensor manager.")

    def get_tensor_channel(self, topic_name: str) -> Optional[TensorChannel]:
        """Return this plugin's TensorChannel for a subscribed topic."""
        return self._tensor_channels.get(topic_name)

    # ------------------------------------------------------------------
    # Internal wiring
    # ------------------------------------------------------------------

    def _bind_tensor_channels(self):
        """Resolve channel handles inside the child process."""
        if not isinstance(self.manager, PluginManagerPubSubTensor):
            return
        for topic_name in self._tensor_sub_specs:
            channel = self.manager.get_tensor_channel(topic_name, self.name)
            if channel is not None:
                self._tensor_channels[topic_name] = channel

    def _run_main(self, options: dict = {}, **kwargs):
        self._bind_tensor_channels()
        super()._run_main(options, **kwargs)

    def _receive_messages(self):
        """
        Poll both regular queues (PluginPubSub) and tensor channels.
        Tensor SYNC reads use a short timeout to stay responsive to stop_event.
        """
        while not self.stop_event.is_set():

            # --- regular pubsub (unchanged behaviour) ---
            for topic_name, callback in self.subscriptions.items():
                if topic_name in self.reception_queues:
                    try:
                        data = self.reception_queues[topic_name].get_nowait()
                        callback(data)
                    except Exception:
                        pass

            # --- tensor channels ---
            for topic_name, channel in self._tensor_channels.items():
                callback = self._tensor_callbacks.get(topic_name)
                if callback is None:
                    continue

                if channel.mode == TensorDeliveryMode.SYNC:
                    tensor = channel.read(timeout=0.005)
                    if tensor is not None:
                        try:
                            callback(tensor)
                        finally:
                            channel.consumed()   # always unblock the writer
                else:
                    # LATEST: only fire when a new version has arrived
                    version = channel.get_version()
                    last    = getattr(self, f"_last_ver_{topic_name}", -1)
                    if version != last:
                        setattr(self, f"_last_ver_{topic_name}", version)
                        tensor = channel.read()
                        if tensor is not None:
                            try:
                                callback(tensor)
                            except Exception:
                                pass


# ---------------------------------------------------------------------------
# PluginManagerPubSubTensor
# ---------------------------------------------------------------------------

class PluginManagerPubSubTensor(PluginManagerPubSub):
    """
    Extends PluginManagerPubSub with tensor topic routing.
    Regular (non-tensor) topics and plugins work exactly as before.
    """

    def __init__(self):
        super().__init__()
        self._tensor_topics: Dict[str, TensorTopic] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_plugin(self, plugin, name: str = None):
        super().register_plugin(plugin, name)

        if not isinstance(plugin, PluginPubSubTensor):
            return

        for topic_name, (shape, dtype, mode) in plugin._tensor_sub_specs.items():
            self._ensure_tensor_topic(topic_name, dtype, mode, shape)
            self._tensor_topics[topic_name].add_subscriber(plugin.name)

        for topic_name, (shape, dtype, mode) in plugin._tensor_pub_specs.items():
            if topic_name not in self._tensor_topics:
                self._ensure_tensor_topic(topic_name, dtype, mode, shape)

    def _ensure_tensor_topic(
        self,
        topic_name: str,
        dtype: torch.dtype,
        mode: str,
        shape: Optional[Tuple] = None,
    ) -> TensorTopic:
        if topic_name not in self._tensor_topics:
            self._tensor_topics[topic_name] = TensorTopic(topic_name, dtype, mode, shape)
        else:
            existing = self._tensor_topics[topic_name]
            # Shape conflict only matters when both sides declare a concrete shape
            if shape is not None and existing.shape is not None and existing.shape != shape:
                raise ValueError(
                    f"Tensor topic '{topic_name}' shape conflict: "
                    f"registered={existing.shape}, requested={shape}."
                )
            if existing.dtype != dtype:
                raise ValueError(
                    f"Tensor topic '{topic_name}' dtype conflict: "
                    f"registered={existing.dtype}, requested={dtype}."
                )
        return self._tensor_topics[topic_name]

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------

    def get_tensor_channel(self, topic_name: str, plugin_name: str) -> Optional[TensorChannel]:
        topic = self._tensor_topics.get(topic_name)
        return topic.get_channel(plugin_name) if topic else None

    def publish_tensor_to_topic(
        self,
        topic_name: str,
        tensor: torch.Tensor,
        lazy: bool = False,
        dtype: torch.dtype = torch.float32,
        mode: str = TensorDeliveryMode.LATEST,
    ):
        """
        Write tensor into every subscriber's shared buffer.

        lazy=True  — used by ActivationTap: creates the topic on the fly if
                     absent (first forward pass).  Subscribers added afterwards
                     will auto-register their channels via add_subscriber().
        lazy=False — standard path; topic must be registered already.
        """
        if lazy and topic_name not in self._tensor_topics:
            self._ensure_tensor_topic(topic_name, dtype, mode, shape=None)

        topic = self._tensor_topics.get(topic_name)
        if topic is not None:
            topic.publish(tensor)
