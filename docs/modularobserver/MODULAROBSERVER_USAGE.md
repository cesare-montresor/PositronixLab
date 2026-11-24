# ModularObserver Usage Guide

Practical recipes for mapping PyTorch models, exporting Dash Cytoscape graphs, and collecting activations/gradients with `ModelMapper` and `ModularObserver`.

> **Note on imports**: `positronixlab/modularobserver/__init__.py` is currently empty, so import classes from their submodules (`positronixlab.modularobserver.modelmapper` and `positronixlab.modularobserver.modularobserver`).

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
        self.device = torch.device("cpu")  # required by ModelMapper.map(..., data=...)
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

> **Tip**: If your `nn.Module` does not have a `.device` attribute (e.g., `nn.Sequential`), either attach `model.device = torch.device("cpu")` or drive tracing directly via `trace_context()` (below).

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

`ModularObserver` combines `ModelMapper` with hook registration across all observable layers (currently learnable linear layers).

```python
import torch
import torch.nn as nn
from positronixlab.modularobserver.modularobserver import ModularObserver

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
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

## 4. Hook Lifecycle Best Practices (Current Caveats)

Until the hook lifecycle issues documented in [analysis.md](../analysis.md) are fixed:

1. **Always unregister before re-registering** a callback to avoid leaking the previous hook handle:
   ```python
   observer.on_forward_layer(None)
   observer.on_forward_layer(new_callback)
   ```
2. **Unregister pre-forward hooks explicitly before `clear()`**, because `ModularObserver.clear()` currently omits `on_pre_forward_layer(None)`:
   ```python
   observer.on_pre_forward_layer(None)
   observer.clear()
   ```
3. **Avoid `on_grad_weights` on multi-layer/multi-parameter models** until the late-binding closure bug (`modularobserver.py:125`) is fixed; if you need parameter gradients today, read `param.grad` after `loss.backward()` or register `param.register_hook(lambda g, n=node.name, p=param: ...)` directly.
4. **Customizing observable layers**: Subclass `ModularObserver` and override `_is_observable_layer(self, module, name)` if you need to observe `nn.Conv2d` or custom `nn.Linear` subclasses:
   ```python
   class ConvAndLinearObserver(ModularObserver):
       def _is_observable_layer(self, module, name):
           return isinstance(module, (nn.Linear, nn.Conv2d))
   ```
