# ModularObserver Usage Guide

Practical recipes for mapping PyTorch models, exporting Dash Cytoscape graphs, and collecting activations/gradients with `ModelMapper` and `ModularObserver`.

> **Note on imports**: `positronixlab.modularobserver` lazily exports `ModularObserver`, `ActivationCache`, `infer_layer_info`, and `reduce_activation_2d` alongside eager `ModelMapper`/`LayerObserver` exports, so standalone scripts and notebooks can use `ActivationCache` without `parallelazy`.

---

## 1. Standalone Topology Mapping (`ModelMapper`)

### Static-Only Mapping (No Forward Pass)

When you only need the list of leaf layers, unit counts, and parameter IDs (no edges or functional activations):

```python
import torch.nn as nn
from positronixlab.modularobserver.modelmapper import ModelMapper
from positronixlab.spacemath.common import is_layer

model = nn.Sequential(
    nn.Linear(16, 32),
    nn.ReLU(),
    nn.Linear(32, 4),
)

mapper = ModelMapper(model, trace_samples=0, is_suitable_layer=lambda m, name: is_layer(m))
graph = mapper.map(model)

for name, node in graph.nodes.items():
    print(name, node.class_name, node.in_units, "->", node.out_units, node.unit_type)
```

### Dynamic Traced Mapping (Edges + Functional Activations)

Passing a sample tensor and a positive integer `trace_samples` runs one `eval()` + `torch.no_grad()` forward pass to discover directed edges and functional activations (`F.relu`, `F.gelu`, `F.silu`, `F.softmax`, `F.log_softmax`, `torch.relu`, `torch.sigmoid`, `torch.tanh`):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from positronixlab.modularobserver.modelmapper import ModelMapper, GraphFormatter
from positronixlab.spacemath.common import is_layer

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        return F.softmax(self.fc2(F.relu(self.fc1(x))), dim=-1)

model = SimpleMLP()
mapper = ModelMapper(model, is_suitable_layer=lambda m, name: is_layer(m))
graph = mapper.map(model, data=torch.randn(2, 8), trace_samples=1)

print("Edges:", [(e.source, e.target) for e in graph.edges])
print("fc1 activation:", graph.nodes["fc1"].activation_fn)
print("fc2 activation:", graph.nodes["fc2"].activation_fn)
```

> **Tip**: `ModelMapper.map(..., data=...)` automatically infers the target device from `model.device`, then `next(model.parameters()).device`, falling back to CPU, so standard `nn.Module`s work without a `.device` attribute.

### Tracing Inside an Existing Training/Eval Loop (`trace_context`)

Use `trace_context()` when you want `ModelMapper` to observe real batches without triggering an extra forward pass:

```python
from positronixlab.modularobserver.modelmapper import ModelMapper, TRACE_ALWAYS

mapper = ModelMapper(model, trace_samples=1)  # or TRACE_ALWAYS to trace every batch
mapper.map(model)  # optional one-time static pass

with mapper.trace_context():
    y = model(batch_x)
```

- `trace_samples=0`: `trace_context()` is a zero-overhead no-op.
- `trace_samples=N` (`int > 0`): traces the next `N` calls, then automatically drops to `0`.
- `trace_samples=TRACE_ALWAYS` (`-1`): traces every call indefinitely.

---

## 2. Exporting Graphs (`GraphFormatter`)

```python
from positronixlab.modularobserver.modelmapper import GraphFormatter

# Full Dash Cytoscape payload: {"elements": [node_elems..., edge_elems...]}
cyto = GraphFormatter.to_cytoscape(graph)

# Plain Python dictionary: {"nodes": {id: {...}}, "edges": [{"source": ..., "target": ...}]}
as_dict = GraphFormatter.to_dict(graph)

# Convenience accessors on ModelMapper
nodes_by_id = mapper.nodes()   # Dict[str, dict]
edges_list = mapper.edges()    # List[dict]
```

---

## 3. Observing Activations & Gradients (`ModularObserver`)

`ModularObserver` combines `ModelMapper` with hook registration across observable layers. By default it observes learnable `nn.Linear` layers, and can be configured via `observable_types`, `target_layers`, and `exclude_patterns`.

```python
import torch
import torch.nn as nn
from positronixlab.modularobserver.modularobserver import ModularObserver

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 12)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(12, 3)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

model = TinyNet()
observer = ModularObserver(config=None, model=model)

# Listen for graph updates (optional)
observer.on_graph_update(lambda g: print("Graph updated:", list(g.nodes.keys())))

# Run static + dynamic mapping
observer.map(dummy_data=torch.randn(2, 8), trace_samples=1)
print("Observable layers:", observer.layer_names())

# Capture forward activations
activations = {}
def on_fwd(module, inputs, output):
    node = observer.by_layer(module)
    activations[node.name] = output.detach()

observer.on_forward_layer(on_fwd)

# Capture backward layer gradients
grad_outputs = {}
def on_bwd(module, grad_in, grad_out):
    node = observer.by_layer(module)
    grad_outputs[node.name] = grad_out[0].detach()

observer.on_backward_layer(on_bwd)

# Run a forward + backward pass
x = torch.randn(4, 8)
out = model(x)
out.sum().backward()

print("Captured activations:", {k: tuple(v.shape) for k, v in activations.items()})
print("Captured grad_outputs:", {k: tuple(v.shape) for k, v in grad_outputs.items()})
```

---

## 4. Customizing Observable Layers & Hook Cleanup

`ModularObserver` supports observing arbitrary layer types (e.g. `nn.Conv2d`, `nn.Linear`, `nn.LSTM`) or explicit layer lists directly via constructor arguments:

```python
# Observe both Linear and Conv2d layers, excluding embeddings/classifier heads
observer = ModularObserver(
    config=None,
    model=model,
    observable_types=(nn.Linear, nn.Conv2d),
    exclude_patterns=("embedding", "head"),
)

# Or target an explicit list of dotted layer names
observer = ModularObserver(
    config=None,
    model=model,
    target_layers=["conv1", "fc1", "fc2"],
)
```

Re-registering any `on_*` callback automatically removes the previous hook handles, and `observer.clear()` detaches all pre-forward, forward, backward, and parameter-gradient hooks:

```python
observer.clear()
```

---

## 5. Frozen-Model Activation & Score Disk Cache (`ActivationCache`)

For offline analysis, HPO, or standalone notebooks without `parallelazy`, `ActivationCache` persists detached 2D layer activations (`cache_mode="activations"`) or `[-1, 1]` unit scores (`cache_mode="scores"`) atomically to disk:

```python
from positronixlab.modularobserver import ActivationCache

cache_dir = ActivationCache.compute_cache_path(
    "/tmp/act_cache",
    signature_fields={"model": "mlp", "split": "val"},
    cache_mode="activations",
)
cache = ActivationCache(cache_dir, cache_mode="activations", cache_dtype="float32")
meta, tensors, hit = cache.load_or_build(lambda: (model, val_loader), device="cpu")
```
