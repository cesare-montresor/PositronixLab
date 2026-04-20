from typing import Callable, Dict, Optional, Tuple
import ctypes
import torch
import torch.nn as nn
from torch import multiprocessing as mp
import time
import threading

from modularnet.plugin.plugin_base import PluginBase
from modularnet.plugin.plugin_manager import PluginManagerBase
from modularnet.task.task import Task


# ---------------------------------------------------------------------------
# Transport strategies — auto-detected from value type (autosense from plugin_env)
# ---------------------------------------------------------------------------

class _Transport:
    PICKLE        = "pickle"         # small objects, stateless modules, criterion
    SHARED_MODULE = "shared_module"  # nn.Module with parameters → share_memory_()
    SHARED_TENSOR = "shared_tensor"  # CPU tensor → zero-copy shared memory
    CUDA_IPC      = "cuda_ipc"       # CUDA tensor → CUDA IPC handle
    SIGNAL_ONLY   = "signal_only"    # large/unserializable → notify only, no value

 
def _classify(value) -> str:
    """Automatically detect the transport strategy for a value."""
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
# PluginTopic — manages both regular messages and tensor channels
# ---------------------------------------------------------------------------

class PluginTopic:
    """
    Manages a single pub/sub topic with both regular messages and tensor channels.
    Intelligently routes data based on its type using autosense.
    """
    def __init__(self, name: str):
        self.name = name
        # Regular message subscribers: plugin_name → mp.Queue
        self.message_subscribers: Dict[str, mp.Queue] = {}
        # Tensor channel subscribers: plugin_name → TensorChannel
        self.tensor_channels: Dict[str, TensorChannel] = {}
        # Tensor topic metadata
        self.tensor_mode = TensorDeliveryMode.LATEST
        self.tensor_shape = None
        self.tensor_dtype = torch.float32

    def add_message_subscriber(self, plugin_name: str, queue: mp.Queue):
        """Register a message queue for a plugin."""
        self.message_subscribers[plugin_name] = queue

    def add_tensor_subscriber(
        self,
        plugin_name: str,
        shape: Optional[Tuple] = None,
        dtype: torch.dtype = torch.float32,
        mode: str = TensorDeliveryMode.LATEST,
    ):
        """Register a tensor channel for a plugin."""
        if plugin_name not in self.tensor_channels:
            self.tensor_channels[plugin_name] = TensorChannel(shape, dtype, mode)
        self.tensor_mode = mode
        self.tensor_shape = shape
        self.tensor_dtype = dtype

    def remove_subscriber(self, plugin_name: str):
        """Unregister a subscriber."""
        self.message_subscribers.pop(plugin_name, None)
        self.tensor_channels.pop(plugin_name, None)

    def broadcast_message(self, data):
        """Put data in all message subscriber queues (non-blocking)."""
        for queue in self.message_subscribers.values():
            try:
                queue.put_nowait(data)
            except:
                pass

    def broadcast_tensor(self, tensor: torch.Tensor):
        """Write tensor into all tensor subscriber channels."""
        for channel in self.tensor_channels.values():
            channel.write(tensor)


# ---------------------------------------------------------------------------
# PluginPubSub — unified plugin with auto-sensing
# ---------------------------------------------------------------------------

class PluginPubSub(PluginBase):
    """
    Unified pub/sub plugin with automatic transport detection.
    Supports regular messages, tensors, and modules transparently.

    Usage
    -----
        class MyPlugin(PluginPubSub):
            def onStart(self):
                # Subscribe to any data type — auto-sensed on receive
                self.subscribe("model", self.on_model_changed)
                self.subscribe("gradients", self.on_grad)

            def on_model_changed(self, model):
                self.model = model

            def on_grad(self, grad_tensor):
                print(grad_tensor.mean())

            def onUpdate(self):
                # Publish any data type — auto-sensed on send
                self.publish("loss", loss_value)
                self.publish("weights", self.model.weight)

    For tensor-specific optimizations (zero-copy with shape inference):
        self.subscribe_tensor("layer_act", self.on_activation,
                            mode=TensorDeliveryMode.LATEST)
    """

    def __init__(self, options: dict = {}, name: str = None, fatal_errors: bool = False):
        super().__init__(options, name, fatal_errors)

        # Regular subscriptions: topic_name → callback
        self.subscriptions: Dict[str, Callable] = {}
        # Reception queues: topic_name → mp.Queue
        self.reception_queues: Dict[str, mp.Queue] = {}
        # Tensor subscriptions: topic_name → (TensorChannel, callback)
        self.tensor_subscriptions: Dict[str, Tuple[Optional[TensorChannel], Callable]] = {}
        # Receiver thread
        self.task_receiver = None

    def set_manager(self, manager: "PluginManagerPubSub"):
        self.manager = manager

    def subscribe(self, topic_name: str, callback: Callable):
        """
        Subscribe to a topic with a callback function.
        Callback receives any type of data; transport auto-detected on publish.
        """
        self.subscriptions[topic_name] = callback

    def subscribe_tensor(
        self,
        topic_name: str,
        callback: Callable,
        shape: Optional[Tuple] = None,
        dtype: torch.dtype = torch.float32,
        mode: str = TensorDeliveryMode.LATEST,
    ):
        """
        Subscribe to a tensor topic with lazy or eager buffer allocation.

        shape=None → lazy: buffer allocated on first received tensor.
        callback(tensor) is invoked with a zero-copy view into shared memory.
        For SYNC mode, callback must call consumed() when done.
        """
        self.tensor_subscriptions[topic_name] = (None, callback)

    def publish(self, topic_name: str, data):
        """
        Publish data to a topic.
        Transport auto-detected based on data type:
          - nn.Module with params → SHARED_MODULE
          - torch.Tensor (CPU) → SHARED_TENSOR
          - torch.Tensor (CUDA) → CUDA_IPC
          - other → PICKLE
        """
        if self.manager:
            self.manager.publish_to_topic(topic_name, data)

    def publish_tensor(self, topic_name: str, tensor: torch.Tensor):
        """Publish a tensor to a tensor topic (guaranteed zero-copy path)."""
        if self.manager:
            self.manager.publish_tensor_to_topic(topic_name, tensor)

    def get_tensor_channel(self, topic_name: str) -> Optional[TensorChannel]:
        """Get this plugin's tensor channel for a subscribed topic."""
        if topic_name in self.tensor_subscriptions:
            channel, _ = self.tensor_subscriptions[topic_name]
            return channel
        return None

    def run_main(self, options: dict = {}, **kwargs):
        """Override to set up receiver thread in the plugin process."""
        self.options = options
        if 'fatal_errors' in kwargs:
            self.fatal_errors = kwargs['fatal_errors']

        # Setup reception queues and channels in the plugin process
        if self.manager:
            for topic_name in self.subscriptions.keys():
                queue = self.manager.get_topic_queue(topic_name, self.name)
                self.reception_queues[topic_name] = queue

            for topic_name, (_, callback) in self.tensor_subscriptions.items():
                channel = self.manager.get_tensor_channel(topic_name, self.name)
                if channel:
                    self.tensor_subscriptions[topic_name] = (channel, callback)

        # Start receiver thread
        self.task_receiver = Task.create(target=self._receive_messages, mode="thread")
        self.task_receiver.start()

        self.onStart()

        while not self.stop_event.is_set():
            try:
                self.onUpdate()
            except Exception as e:
                print(f"{self.__class__.__name__} encountered an error: {e}")
                import traceback
                traceback.print_exc()
                if self.fatal_errors:
                    self.stop_event.set()
                    raise e
            time.sleep(0.001)

        self.onStop()
        if self.task_receiver:
            self.task_receiver.join()

    def _receive_messages(self):
        """Continuously receive and dispatch messages and tensors."""
        while not self.stop_event.is_set():
            # Regular message subscriptions
            for topic_name, callback in self.subscriptions.items():
                if topic_name in self.reception_queues:
                    queue = self.reception_queues[topic_name]
                    try:
                        data = queue.get(timeout=0.01)
                        try:
                            callback(data)
                        except Exception as e:
                            print(f"Message callback error on {topic_name}: {e}")
                    except:
                        pass

            # Tensor subscriptions
            for topic_name, (channel, callback) in self.tensor_subscriptions.items():
                if channel is not None:
                    timeout = 0.01 if channel.mode == TensorDeliveryMode.SYNC else None
                    tensor = channel.read(timeout=timeout)
                    if tensor is not None:
                        try:
                            callback(tensor)
                            if channel.mode == TensorDeliveryMode.SYNC:
                                channel.consumed()
                        except Exception as e:
                            print(f"Tensor callback error on {topic_name}: {e}")

            time.sleep(0.01)


# ---------------------------------------------------------------------------
# PluginManagerPubSub — unified manager with auto-sensing
# ---------------------------------------------------------------------------

class PluginManagerPubSub(PluginManagerBase):
    """
    Manager for registering plugins and handling unified pub/sub with auto-sensing.

    Supports:
      - Regular pub/sub messaging (any pickleable object)
      - Tensor topics with zero-copy shared memory (SYNC and LATEST modes)
      - Automatic transport selection based on value type
      - Activation hooks via ActivationTap for forward pass monitoring

    Usage
    -----
        manager = PluginManagerPubSub()

        # Register plugins
        manager.register_plugin(trainer_plugin)
        manager.register_plugin(analyzer_plugin)

        # Declare tensor topics (optional, auto-created on first publish)
        manager.publish_tensor_topic("activations", shape=(256, 512))

        # Attach forward hooks
        tap = ActivationTap("layer_out", manager, mode=TensorDeliveryMode.LATEST)
        model.layer.register_forward_hook(tap.hook)

        manager.start_all()
    """

    def __init__(self):
        super().__init__()
        self.topics: Dict[str, PluginTopic] = {}
        self._mp_manager = None
        self._shared_state = None

    def _init_shared_state(self):
        """Initialize shared multiprocessing state."""
        if self._mp_manager is None:
            self._mp_manager = mp.Manager()
            self._shared_state = self._mp_manager.dict()

    def register_plugin(self, plugin: PluginBase, name: str = None):
        """Register a plugin with the manager."""
        if name is None:
            name = plugin.name
        self.plugins[name] = plugin

        self._init_shared_state()

        if isinstance(plugin, PluginPubSub):
            # Register message subscriptions
            for topic_name in plugin.subscriptions.keys():
                self._register_topic_subscription(plugin.name, topic_name)

            # Register tensor subscriptions
            for topic_name, (_, _) in plugin.tensor_subscriptions.items():
                # Will be registered via manager when called from plugin
                pass

            plugin.set_manager(self)

    def _register_topic_subscription(self, plugin_name: str, topic_name: str):
        """Register a message subscription queue."""
        if topic_name not in self._shared_state:
            self._shared_state[topic_name] = self._mp_manager.dict()

        if plugin_name not in self._shared_state[topic_name]:
            queue = self._mp_manager.Queue()
            self._shared_state[topic_name][plugin_name] = queue

    def get_topic(self, topic_name: str) -> PluginTopic:
        """Get or create a topic."""
        if topic_name not in self.topics:
            self.topics[topic_name] = PluginTopic(topic_name)
        return self.topics[topic_name]

    def get_topic_queue(self, topic_name: str, plugin_name: str) -> mp.Queue:
        """Get or create a reception queue for a plugin on a topic."""
        self._init_shared_state()

        if topic_name not in self._shared_state:
            self._shared_state[topic_name] = self._mp_manager.dict()

        if plugin_name not in self._shared_state[topic_name]:
            queue = self._mp_manager.Queue()
            self._shared_state[topic_name][plugin_name] = queue

        return self._shared_state[topic_name][plugin_name]

    def get_tensor_channel(self, topic_name: str, plugin_name: str) -> Optional[TensorChannel]:
        """Get a tensor channel for a plugin."""
        topic = self.topics.get(topic_name)
        if topic:
            return topic.tensor_channels.get(plugin_name)
        return None

    def publish_to_topic(self, topic_name: str, data):
        """
        Publish data to a topic.
        Auto-detects transport based on value type and routes accordingly.
        """
        self._init_shared_state()

        transport = _classify(data)

        # Prepare data for sharing if needed
        if transport == _Transport.SHARED_MODULE:
            data.share_memory_()
        elif transport in (_Transport.SHARED_TENSOR, _Transport.CUDA_IPC):
            if not data.is_shared():
                data = data.share_memory_()

        topic = self.get_topic(topic_name)

        # Broadcast to message subscribers
        if topic_name in self._shared_state:
            for queue in self._shared_state[topic_name].values():
                try:
                    queue.put(data, block=False)
                except:
                    pass

        # Also broadcast to tensor channels if it's a tensor
        if isinstance(data, torch.Tensor):
            topic.broadcast_tensor(data)

    def publish_tensor_to_topic(
        self,
        topic_name: str,
        tensor: torch.Tensor,
        lazy: bool = False,
        dtype: torch.dtype = torch.float32,
        mode: str = TensorDeliveryMode.LATEST,
    ):
        """
        Publish a tensor to all subscribers on this topic.

        lazy=True  — used by ActivationTap: creates the topic on the fly if
                     absent (first forward pass).  Subscribers auto-register.
        lazy=False — standard path; topic must be registered already.
        """
        if lazy and topic_name not in self.topics:
            self._ensure_tensor_topic(topic_name, dtype, mode, None)

        topic = self.get_topic(topic_name)
        if topic:
            topic.broadcast_tensor(tensor)

    def _ensure_tensor_topic(
        self,
        topic_name: str,
        dtype: torch.dtype = torch.float32,
        mode: str = TensorDeliveryMode.LATEST,
        shape: Optional[Tuple] = None,
    ) -> PluginTopic:
        """Ensure a tensor topic exists with the given parameters."""
        if topic_name not in self.topics:
            topic = self.get_topic(topic_name)
            topic.tensor_mode = mode
            topic.tensor_dtype = dtype
            topic.tensor_shape = shape
        return self.topics[topic_name]

    def publish_tensor_topic(
        self,
        topic_name: str,
        shape: Optional[Tuple] = None,
        dtype: torch.dtype = torch.float32,
        mode: str = TensorDeliveryMode.SYNC,
    ):
        """
        Declare a tensor topic this manager will publish to.
        Useful for eager allocation with known shape; use lazy if shape unknown.
        """
        self._ensure_tensor_topic(topic_name, dtype, mode, shape)
