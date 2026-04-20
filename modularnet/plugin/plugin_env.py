import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import multiprocessing as mp

from modularnet.plugin.plugin_base import PluginBase, PluginManagerBase



# ---------------------------------------------------------------------------
# Transport strategies — auto-detected from value type
# ---------------------------------------------------------------------------

class _Transport:
    PICKLE        = "pickle"         # small objects, stateless modules, criterion
    SHARED_MODULE = "shared_module"  # nn.Module with parameters → share_memory_()
    SHARED_TENSOR = "shared_tensor"  # CPU tensor → TensorChannel-style slot
    CUDA_IPC      = "cuda_ipc"       # CUDA tensor → CUDA IPC handle
    SIGNAL_ONLY   = "signal_only"    # large/unserializable → notify only, no value


def _classify(value: Any) -> str:
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
# EnvSlot — one registered variable in the registry
# ---------------------------------------------------------------------------

class EnvSlot:
    """
    Holds the current value of one env variable and the notification queues
    for all watching processes.

    The value itself always lives here (manager process memory).
    Watchers receive a notification through a small mp.Queue; what is inside
    the notification depends on the transport:

      PICKLE / SHARED_MODULE / SHARED_TENSOR / CUDA_IPC
          → queue.put((old_value, new_value))
            Receivers get the actual objects.  For SHARED_MODULE the module
            has already had share_memory_() called, so all parameter tensors
            are in shared memory — the pickle is just the object graph, not
            the weights.

      SIGNAL_ONLY
          → queue.put((None, None))
            Receivers call env_get() themselves.
    """

    def __init__(self, key: str, default: Any = None, signal_only: bool = False):
        self.key         = key
        self.value       = default
        self.signal_only = signal_only
        self.transport   = _Transport.SIGNAL_ONLY if signal_only else _classify(default) if default is not None else None
        # plugin_name → mp.Queue  (created before processes fork)
        self.watcher_queues: Dict[str, mp.Queue] = {}
        self._lock = threading.Lock()

    def add_watcher(self, plugin_name: str) -> mp.Queue:
        """Register a watcher queue for a plugin. Call before plugin.start()."""
        if plugin_name not in self.watcher_queues:
            self.watcher_queues[plugin_name] = mp.Queue(maxsize=4)
        return self.watcher_queues[plugin_name]

    def set(self, new_value: Any) -> Tuple[Any, Any, str]:
        """
        Update the stored value.  Returns (old, new, transport) so the
        manager can fan out notifications.
        """
        with self._lock:
            old = self.value

            if self.signal_only:
                self.value = new_value
                return old, new_value, _Transport.SIGNAL_ONLY

            transport = _classify(new_value)

            if transport == _Transport.SHARED_MODULE:
                new_value.share_memory_()   # in-place, moves all params to shm

            elif transport == _Transport.SHARED_TENSOR:
                if not new_value.is_shared():
                    new_value = new_value.share_memory_()

            elif transport == _Transport.CUDA_IPC:
                # CUDA tensors: share_memory_() is the right call for IPC
                new_value = new_value.share_memory_()

            self.value    = new_value
            self.transport = transport
            return old, new_value, transport

    def notify_all(self, old: Any, new: Any, transport: str):
        """Push notification into every watcher's queue (non-blocking, drop if full)."""
        payload = (None, None) if transport == _Transport.SIGNAL_ONLY else (old, new)
        for q in self.watcher_queues.values():
            try:
                q.put_nowait(payload)
            except Exception:
                pass  # watcher is not keeping up — drop, env vars change slowly


# ---------------------------------------------------------------------------
# PluginVar — one env variable declaration (used by plugins before registration)
# ---------------------------------------------------------------------------

class PluginVar:
    """
    Lightweight descriptor a plugin uses to declare an env variable it needs.

        class MyPlugin(PluginEnv):
            def __init__(self):
                super().__init__()
                self._model_var = PluginVar("model", watch=True)
                self._lr_var    = PluginVar("learning_rate", default=1e-3, watch=True)

    Actual get/set goes through PluginEnv.env_get / env_set.
    """
    def __init__(self, key: str, default: Any = None, watch: bool = False, signal_only: bool = False):
        self.key         = key
        self.default     = default
        self.watch       = watch          # if True, plugin wants onChange callbacks
        self.signal_only = signal_only


# ---------------------------------------------------------------------------
# PluginEnv — mixin for plugins
# ---------------------------------------------------------------------------

class PluginEnv(PluginBase):
    """
    Mixin that adds env variable get/set/watch to a plugin.

    Designed for multiple inheritance:

        class TrainerPlugin(PluginEnv, PluginPubSubTensor):
            ...

    Usage inside a plugin
    ---------------------
        # In __init__, declare what you need:
        self.env_declare("model",          watch=True, signal_only=False)
        self.env_declare("learning_rate",  watch=True, default=1e-3)
        self.env_declare("data",           watch=True, signal_only=True)

        # In onStart or wherever:
        self.env_set("learning_rate", 3e-4)

        # Callbacks:
        def on_model_changed(self, old, new):
            self.model = new
            new.register_forward_hook(...)

    onChange is called in a dedicated receiver thread inside the child process,
    same pattern as the pubsub receiver.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # key → PluginVar  (declared by the plugin)
        self._env_declarations: Dict[str, PluginVar] = {}
        # key → Callable[[old, new], None]
        self._env_callbacks: Dict[str, Callable] = {}
        # key → mp.Queue  (assigned by manager at register time)
        self._env_queues: Dict[str, mp.Queue] = {}
        # receiver thread handle
        self._env_receiver = None

    # ------------------------------------------------------------------
    # Plugin-facing API
    # ------------------------------------------------------------------

    def env_declare(
        self,
        key: str,
        default: Any = None,
        watch: bool = False,
        signal_only: bool = False,
        on_change: Optional[Callable] = None,
    ):
        """
        Declare interest in an env variable.  Call in __init__ before
        register_plugin() so the manager can wire up queues before fork.

        on_change(old, new) — shorthand for declaring a callback inline.
        """
        var = PluginVar(key, default=default, watch=watch, signal_only=signal_only)
        self._env_declarations[key] = var
        if on_change is not None:
            self._env_callbacks[key] = on_change

    def env_watch(self, key: str, callback: Callable):
        """Attach a callback to an already-declared variable."""
        self._env_callbacks[key] = callback

    def env_set(self, key: str, value: Any):
        """Write a value to the env registry (works from child processes)."""
        if self.manager and isinstance(self.manager, PluginManagerEnv):
            self.manager._env_set_from_plugin(key, value, self.name)
        else:
            raise RuntimeError("env_set requires a PluginManagerEnv manager.")

    def env_get(self, key: str) -> Any:
        """Read the current value of an env variable."""
        if self.manager and isinstance(self.manager, PluginManagerEnv):
            return self.manager.env_get(key)
        raise RuntimeError("env_get requires a PluginManagerEnv manager.")

    # ------------------------------------------------------------------
    # Internal wiring — called by run_main
    # ------------------------------------------------------------------

    def _bind_env_queues(self):
        """Resolve notification queues inside the child process."""
        if not isinstance(self.manager, PluginManagerEnv):
            return
        for key, var in self._env_declarations.items():
            if var.watch:
                q = self.manager._get_env_queue(key, self.name)
                if q is not None:
                    self._env_queues[key] = q

    def _start_env_receiver(self):
        """Start the notification receiver thread."""
        t = threading.Thread(target=self._env_receive_loop, daemon=True)
        t.start()
        self._env_receiver = t

    def _env_receive_loop(self):
        while not self.stop_event.is_set():
            for key, queue in self._env_queues.items():
                callback = self._env_callbacks.get(key)
                if callback is None:
                    continue
                try:
                    old, new = queue.get(timeout=0.05)
                    # If signal_only, new == None — caller should use env_get()
                    try:
                        callback(old, new)
                    except Exception as e:
                        print(f"[{self.name}] env_watch callback error on '{key}': {e}")
                except Exception:
                    pass  # timeout or empty

    def run_main(self, options: dict = {}, **kwargs):
        self._bind_env_queues()
        self._start_env_receiver()
        super().run_main(options, **kwargs)


# ---------------------------------------------------------------------------
# PluginManagerEnv — mixin for managers
# ---------------------------------------------------------------------------

class PluginManagerEnv(PluginManagerBase):
    """
    Mixin that adds an env registry to a manager.

    Designed for multiple inheritance:

        class MyManager(PluginManagerEnv, PluginManagerPubSubTensor):
            ...

    Usage
    -----
        manager = MyManager()
        manager.env_register("model",         signal_only=False)
        manager.env_register("learning_rate", default=1e-3)
        manager.env_register("data",          signal_only=True)

        manager.register_plugin(model_plugin)
        manager.register_plugin(trainer_plugin)
        manager.register_plugin(hook_plugin)

        # After all plugins are registered, before start:
        manager.env_set("learning_rate", 3e-4)  # can also be set from main process
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._env_slots: Dict[str, EnvSlot] = {}
        # Command queue: child processes send (key, value, sender_name) here
        # The manager's dispatch thread picks them up and calls _do_set()
        self._env_cmd_queue: mp.Queue = mp.Queue()
        self._env_dispatch_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Setup API (call before register_plugin)
    # ------------------------------------------------------------------

    def env_register(
        self,
        key: str,
        default: Any = None,
        signal_only: bool = False,
    ) -> "PluginManagerEnv":
        """
        Declare an env variable in the registry.
        Returns self for chaining.
        """
        if key not in self._env_slots:
            self._env_slots[key] = EnvSlot(key, default=default, signal_only=signal_only)
        return self

    # ------------------------------------------------------------------
    # Manager-facing API
    # ------------------------------------------------------------------

    def env_set(self, key: str, value: Any):
        """Set a value from the main process (before or after plugins start)."""
        self._ensure_slot(key)
        self._do_set(key, value)

    def env_get(self, key: str) -> Any:
        """Get the current value. Safe to call from any process."""
        slot = self._env_slots.get(key)
        return slot.value if slot else None

    # ------------------------------------------------------------------
    # Registration — wire up watcher queues before fork
    # ------------------------------------------------------------------

    def register_plugin(self, plugin: PluginBase, name: str = None):
        super().register_plugin(plugin, name)

        if not isinstance(plugin, PluginEnv):
            return

        for key, var in plugin._env_declarations.items():
            self._ensure_slot(key, default=var.default, signal_only=var.signal_only)
            if var.watch:
                q = self._env_slots[key].add_watcher(plugin.name)
                # Store reference so _bind_env_queues() can find it
                plugin._env_queues[key] = q

        # Start the command dispatcher if not already running
        self._ensure_dispatch_thread()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_slot(self, key: str, default: Any = None, signal_only: bool = False):
        if key not in self._env_slots:
            self._env_slots[key] = EnvSlot(key, default=default, signal_only=signal_only)

    def _get_env_queue(self, key: str, plugin_name: str) -> Optional[mp.Queue]:
        slot = self._env_slots.get(key)
        if slot is None:
            return None
        return slot.watcher_queues.get(plugin_name)

    def _do_set(self, key: str, value: Any):
        """Actually update the slot and fan out notifications. Runs in manager process."""
        slot = self._env_slots.get(key)
        if slot is None:
            return
        old, new, transport = slot.set(value)
        slot.notify_all(old, new, transport)

    def _env_set_from_plugin(self, key: str, value: Any, sender: str):
        """
        Called by PluginEnv.env_set() from a child process.
        Sends a command to the manager's dispatch thread via mp.Queue.
        The value is pickled here (once) and unpickled in the manager process.
        For SHARED_MODULE / SHARED_TENSOR the pickle is just the object graph —
        weights already live in shared memory.
        """
        self._ensure_slot(key)
        self._env_cmd_queue.put((key, value, sender))

    def _ensure_dispatch_thread(self):
        if self._env_dispatch_thread is None or not self._env_dispatch_thread.is_alive():
            t = threading.Thread(target=self._dispatch_loop, daemon=True)
            t.start()
            self._env_dispatch_thread = t

    def _dispatch_loop(self):
        """
        Manager-side thread that picks up env_set commands from child processes
        and applies them to the registry, then fans out notifications.
        """
        while True:
            try:
                key, value, sender = self._env_cmd_queue.get(timeout=0.1)
                self._do_set(key, value)
            except Exception:
                pass  # timeout — keep spinning