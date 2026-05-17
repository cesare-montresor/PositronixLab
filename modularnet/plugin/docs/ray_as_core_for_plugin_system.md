# Ray High-Performance Architecture Specification for Hook Interception

*A highly optimized blueprint for ultra-low latency data plane pipelines and real-time environment variable multi-casting on a single machine.*

---

## 1. System Communication Architecture

To maintain peak execution speeds during high-frequency tasks like forward hook interception, the system splits communication into two strictly isolated planes:

* **The Data Plane (High-Bandwidth / Zero-Copy):** Handles intermediate layer activations. By pushing metadata references (`ObjectRef`) through a `ray.util.queue.Queue` rather than raw bytes, tensors remain static in Ray’s Plasma shared memory object store. Downstream plugins map directly to this memory space, completely eliminating CPU serialization and RAM-to-RAM copying bottlenecks.


* **The State/Env Plane (Microsecond Multi-Cast):** Handles live configurations, system flags, and tuning parameters. Instead of plugins polling a central broker, the Orchestrator holds explicit reference handles to all running plugin actors and pushes updates directly into their localized memory cache concurrently via asynchronous out-of-band remote calls.

---

## 2. High-Performance Configuration Matrix

| Requirement | Ray Primitive Used | Architectural Pattern | Performance Trick |
| --- | --- | --- | --- |
| **Plugin / Mod Wrapper** | State-Managed Actor (`@ray.remote`) | Isolated OS Process Space | Assign specific hardware resources (`num_gpus=0.2`). |
| **Hook Interception Data Stream** | `ray.util.queue.Queue` + `ray.put()` | Shared Memory Object Passing

 | Wrap transfers in an independent PyTorch CUDA Stream. |
| **Blazing-Fast Env Subs** | Asynchronous Actor Methods (`async def`) | Direct Event-Driven Push Cache | Read variables synchronously from local `O(1)` dicts. |
| **Request-Response Services** | P2P Remote Method Invocation | Direct Peer Actor Handles | Bypasses the topic broker entirely for low latency. |

---

## 3. Production Code Implementations

### The High-Speed Core Engine (`orchestrator.py`)

```python
import logging
from dataclasses import dataclass
from typing import Any, Dict, List
import ray
from ray.util.queue import Queue

@dataclass
class EnvUpdate:
    key: str
    value: Any

@ray.remote
class NodeOrchestrator:
    def __init__(self):
        self._nodes: Dict[str, ray.actor.ActorHandle] = {}
        self._topics: Dict[str, List[Queue]] = {}

    def register_node(self, name: str, handle: ray.actor.ActorHandle):
        self._nodes[name] = handle
        return True

    def get_node(self, name: str):
        return self._nodes[name]

    # --- High-Throughput Pub/Sub Routing ---
    def subscribe(self, topic: str, max_backlog: int = 50) -> Queue:
        if topic not in self._topics:
            self._topics[topic] = []
        sub_queue = Queue(maxsize=max_backlog)
        self._topics[topic].append(sub_queue)
        return sub_queue

    def publish(self, topic: str, msg: Any):
        if topic in self._topics:
            for q in self._topics[topic]:
                try:
                    q.put(msg, block=False) # Never block the producer
                except Exception:
                    pass # Drops frames if subscriber drops behind boundary limits

    # --- Blazing Fast Environment Multi-Cast ---
    def set_env_variable(self, key: str, value: Any):
        """Pushes updates directly into plugin local memory profiles concurrently."""
        update = EnvUpdate(key=key, value=value)
        for node_handle in self._nodes.values():
            node_handle.receive_env_update.remote(update)

```

### The System Integration Framework (`wrapper.py`)

```python
import asyncio
from pathlib import Path
import ray

@ray.remote
class PluginWrapperNode:
    def __init__(self, node_name: str, orchestrator_handle):
        self.node_name = node_name
        self.orchestrator = orchestrator_handle
        self.status = "INITIALIZED"
        self._subscriptions = []
        self._local_env = {} # Nanosecond local cache lookups

    def load_plugin(self, script_path: str, class_name: str, config: dict):
        import importlib
        import sys
        path = Path(script_path)
        sys.path.insert(0, str(path.parent))
        module = importlib.import_module(path.stem)
        
        self.plugin_instance = getattr(module, class_name)(self.node_name, config)
        self.status = "RUNNING"
        
        # Connect Data Plane
        for sub in self.plugin_instance.get_subscriptions():
            q_handle = ray.get(self.orchestrator.subscribe.remote(sub["topic"]))
            callback = getattr(self.plugin_instance, sub["callback"])
            self._subscriptions.append({"queue": q_handle, "callback": callback, "publish_to": sub.get("publish_to")})

    # --- Event-Driven State Channel (Control Plane) ---
    async def receive_env_update(self, update) -> bool:
        """Runs on a concurrent thread loop. Overwrites local variable pointer instantly."""
        self._local_env[update.key] = update.value
        return True

    # --- Point-To-Point Request/Response Channel (Services) ---
    def execute_service(self, method_name: str, *args, **kwargs):
        method = getattr(self.plugin_instance, method_name)
        return method(*args, **kwargs)

    # --- Core Pipeline Execution Loop ---
    def spin_once(self):
        if self.status != "RUNNING":
            return
        for sub in self._subscriptions:
            q = sub["queue"]
            if not q.empty():
                msg = q.get()
                
                # Resolve reference dynamically from shared memory space[cite: 1]
                data = ray.get(msg["data_ref"]) if isinstance(msg, dict) and "data_ref" in msg else msg
                
                out = sub["callback"](data, self._local_env) # Pass O(1) env vars along
                if out is not None and sub["publish_to"]:
                    self.orchestrator.publish.remote(sub["publish_to"], out)

```

### Optimized Hook Publisher Mod (`hook_model.py`)

```python
import torch
import ray

class HighSpeedHookModel:
    def __init__(self, model, orchestrator_handle):
        self.model = model
        self.orchestrator = orchestrator_handle
        self.transfer_stream = torch.cuda.Stream() # Dedicated side-car CUDA stream
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if "layer" in name: # Intercept targeted hidden feature spaces
                module.register_forward_hook(self._create_cuda_stream_hook(name))

    def _create_cuda_stream_hook(self, layer_name: str):
        def hook_fn(module, input, output):
            # Intercept on a parallel GPU timeline to unblock the main forward path
            with torch.cuda.stream(self.transfer_stream):
                tensor_detached = output.detach()
                cpu_tensor = tensor_detached.to("cpu", non_blocking=True)
                
                # Register data reference onto Plasma Shared memory allocation block[cite: 1]
                data_ref = ray.put(cpu_tensor.numpy())
                
                # Push small 64-bit reference payload down the pipeline
                self.orchestrator.publish.remote("activation_stream", {
                    "layer": layer_name,
                    "data_ref": data_ref
                })
        return hook_fn

```

---

## 4. Architectural Implementation Safeguards

1. **Enforce Read-Only Access across Shared Data Slices:** Tensors accessed via Ray `ObjectRef` definitions point directly to underlying Plasma memory segments. Attempting to modify these tensors in-place will cause memory faults. Always isolate mutations using explicit memory duplicates: `tensor.clone()`.


2. **Decouple Execution Realities:** Keep parameter updates locked behind `async def` declarations to maintain administrative priority inside the worker process. Keep heavy algorithmic compute structures enclosed inside standard synchronous frameworks to utilize dedicated hardware throughput.
3. **Boundary Constraints:** Set strict bounds on queue execution variables (`maxsize` limits). If your processing nodes temporarily choke, unconstrained publishing allocations will rapidly cause out-of-memory panics.