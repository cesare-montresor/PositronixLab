# PositronixLab ModularObserver System

`positronixlab.modularobserver` maps PyTorch `nn.Module` hierarchies into a serializable topology graph (nodes, edges, I/O unit metrics, and discovered functional activations) and manages forward, pre-forward, backward, and parameter-gradient hooks on observable layers.

It sits between a PyTorch model and downstream consumers such as `ModularSpace` / `LayerSpace` (activation harvesting), `ModularUI` (Dash Cytoscape visualization), and `tasks/compositionality` (post-hoc atlas evaluation).

---

## Architecture

```
                         PyTorch nn.Module
                                 │
          ┌──────────────────────┴──────────────────────┐
          │ Static walk (named_modules)                 │ Dynamic trace (forward pre-hooks
          │ one-time leaf discovery                     │ + TorchFunctionMode interceptor)
          ▼                                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                          ModelMapper                            │
│  • extract_layer_io_metrics(module) -> in/out units, param_ids  │
│  • _MapperInterceptor (TorchFunctionMode) -> F.relu, F.softmax… │
│  • trace_budget: 0 | N | TRACE_ALWAYS (-1) | float              │
└───────────────────────────────┬─────────────────────────────────┘
                                │ populates & updates
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                           ModelGraph                            │
│  • nodes: Dict[str, ModelNode]                                  │
│  • edges: Set[ModelEdge]  (self-loops rejected)                 │
└──────────────┬────────────────────────────────┬─────────────────┘
               │                                │ on_graph_changed
               ▼                                ▼
┌──────────────────────────┐     ┌────────────────────────────────┐
│      GraphFormatter      │     │        ModularObserver         │
│  • to_cytoscape(graph)   │     │  • filters observable layers   │
│  • to_dict(graph)        │     │    (target_layers / types /    │
│  • nodes_dict/nodes_list │     │     default learnable linear)  │
│  • edges_list/edges_dict │     │  • on_pre_forward_layer(cb)    │
└──────────────────────────┘     │  • on_forward_layer(cb)        │
                                 │  • on_backward_layer(cb)       │
                                 │  • on_grad_weights(cb)         │
                                 │  • clear()                     │
                                 └────────────────────────────────┘
```

### Two-Tier Layer Filtering

`ModularObserver` separates **graph visibility** from **hook observation**:

1. **Graph suitability (`_is_suitable_layer`)**: Passed to `ModelMapper.is_suitable_layer`. Accepts any module whose name is in `target_layers`, whose type matches `observable_types`, or for which `spacemath.common.is_layer(module)` is `True` (`torch.nn.modules.{linear,conv,rnn,transformer}`). Every matching leaf module becomes a `ModelNode` in `observer.graph()`.
2. **Hook observability (`_is_observable_layer`)**: Filters graph nodes synced into `nodes_list` / `nodes_by_name` / `nodes_by_layer`. Rejects zero-parameter modules and names containing any substring in `exclude_patterns`, then matches `target_layers` (if provided), else `observable_types` (if provided), else falls back to `is_linear(module)`. Only observable layers receive hooks from `on_*` methods.

---

## Module Layout

| File | Contents | Status |
| :--- | :--- | :--- |
| `positronixlab/modularobserver/__init__.py` | Package marker (currently empty; import from submodules directly) | Active |
| `positronixlab/modularobserver/modelmapper.py` | `TRACE_ALWAYS`, `extract_layer_io_metrics`, `ModelNode`, `ModelEdge`, `ModelGraph`, `GraphFormatter`, `_MapperInterceptor`, `ModelMapper` | Active |
| `positronixlab/modularobserver/modularobserver.py` | `ModularObserver` and callback type aliases (`PreForwardCallback`, `ForwardCallback`, `WeightCallback`) | Active |
| `positronixlab/modularobserver/layerobserver.py` | `LayerObserver` standalone per-layer hook wrapper (not currently wired into `ModularObserver`) | Orphaned |

---

## API Reference

### `modelmapper.py`

#### `TRACE_ALWAYS = -1`
Sentinel budget value instructing `ModelMapper` to trace every forward pass without decrementing the budget.

#### `extract_layer_io_metrics(module: nn.Module) -> Dict[str, Any]`
Inspects a single module (non-recursive `parameters(recurse=False)`) and returns:
```python
{
    "in_units": Optional[int],
    "out_units": Optional[int],
    "unit_type": str,       # 'neurons' | 'feature_maps' | 'embeddings' | 'hidden_states' |
                            # 'attention_dim' | 'normalized_channels' | 'normalized_features' | 'unknown'
    "param_ids": Set[int],  # id(p) for direct trainable parameters
}
```
Recognized module families: `nn.Linear`, `nn.Conv{1,2,3}d`, `nn.ConvTranspose{1,2}d`, `nn.Embedding`, `nn.RNNBase`/`RNN`/`LSTM`/`GRU`, `nn.MultiheadAttention`, `nn.BatchNorm{1,2}d`, `nn.LayerNorm`, `nn.GroupNorm`.

#### `ModelNode` (dataclass)
Represents one mapped leaf module:
- Structural fields: `name`, `class_name`, `module_ref`, `layer_type` (`LayerType.get_layer_type(module)`), `param_ids`, `in_units`, `out_units`, `unit_type`.
- Discovery fields: `activation_fn: Optional[Callable]`, `depth: Optional[int]`.
- `update_from(new: ModelNode) -> bool`: Merges structural changes in place without clobbering previously discovered `activation_fn` / `depth` when `new` has `None`. Returns `True` if any field changed.
- `to_dict() -> Dict[str, Any]`: JSON-friendly dictionary with stringified `module_ref` and `activation_fn.__name__`, and `param_ids` as `list[int]`.

#### `ModelEdge` (frozen dataclass)
Directed edge `(source: str, target: str)` with `name` property (`"{source}-{target}"`) and `to_dict() -> {'source': ..., 'target': ...}`.

#### `ModelGraph`
Container holding `nodes: Dict[str, ModelNode]` and `edges: Set[ModelEdge]`.
- `add_node(node: ModelNode) -> bool`: Inserts or updates in place; returns `True` if new or modified.
- `add_edge(source: str, target: str) -> bool`: Inserts directed edge; rejects self-loops (`source == target`) and duplicates.

#### `GraphFormatter`
Stateless serializer classmethods over `ModelGraph`:
- `to_cytoscape(graph) -> {'elements': [...]}`: Dash Cytoscape payload combining `node_to_cytoscape` and `edge_to_cytoscape`.
- `node_to_cytoscape(node) -> {'data': ..., 'classes': 'node-<classname> node-module'}`: Adds multiline `data['label']`.
- `edge_to_cytoscape(edge) -> {'data': {'source': ..., 'target': ...}}`.
- `to_dict(graph) -> {'nodes': Dict[str, dict], 'edges': List[dict]}`.
- `nodes_dict(graph)`, `nodes_list(graph)`, `edges_list(graph)`, `edges_dict(graph)`.

#### `ModelMapper`
```python
ModelMapper(
    model: nn.Module,
    trace_samples: Union[int, float] = 0,
    is_suitable_layer: Optional[Callable[[nn.Module, str], bool]] = None,
    on_graph_changed: Optional[Callable[['ModelMapper'], None]] = None,
)
```
- `map(model, data=None, trace_samples=None) -> ModelGraph`: Runs one-time static mapping, then (if `data` is provided and `trace_budget != 0`) runs a `torch.no_grad()` traced forward pass in `eval()` mode (restoring prior `training` flag afterward).
- `trace_context()`: Context manager for tracing an externally driven forward pass (attaches temporary forward pre-hooks and `_MapperInterceptor`, cleans them up on exit, decrements integer `trace_budget`, and fires `on_graph_changed` if the graph mutated).
- `nodes() -> Dict[str, Dict[str, Any]]` and `edges() -> List[Dict[str, Any]]`: Convenience delegators to `GraphFormatter`.

---

### `modularobserver.py`

#### `ModularObserver`
```python
ModularObserver(
    config: PipelineConfig,
    model: nn.Module,
    device: torch.device = None,
    observable_types: tuple[type[nn.Module], ...] | None = None,
    target_layers: List[str] | None = None,
    exclude_patterns: tuple[str, ...] = (),
)
```
When `device` is omitted, `ModularObserver` infers it from `model.device`, then `next(model.parameters()).device`, falling back to `torch.device("cpu")`.

- `map(dummy_data: torch.Tensor = None, trace_samples: int = 1)`: Delegates to `self.mapper.map(self.model, dummy_data, trace_samples=trace_samples)` and triggers `_on_graph_discovered` -> `_sync_observers_with_mapper`.
- `graph() -> ModelGraph`: Returns `self.mapper.graph`.
- `layer_names() -> List[str]`: Names of observable layers.
- `by_name(name: str) -> Optional[ModelNode]` / `by_layer(layer: nn.Module) -> Optional[ModelNode]`: Look up an observable `ModelNode`.
- `on_graph_update(callback)`: Registers a callback invoked with `ModelGraph` whenever `ModelMapper` reports a graph change.
- `on_pre_forward_layer(callback=None)`: Registers or removes (`callback=None`) `register_forward_pre_hook` on every observable layer (automatically removes any existing handles before re-registering).
- `on_forward_layer(callback=None)`: Registers or removes `register_forward_hook` on every observable layer (removes prior handles first).
- `on_backward_layer(callback=None)`: Registers or removes `register_full_backward_hook` on every observable layer (removes prior handles first).
- `on_grad_weights(callback=None)`: Registers or removes `Tensor.register_hook` on parameters of observable layers (removes prior handles first; captures `(module_ref, param)` via default arguments).
- `clear()`: Removes pre-forward, forward, backward, and weight-grad hooks and clears internal node registries.

---

### `layerobserver.py`

#### `LayerObserver`
Lightweight per-layer hook wrapper initialized with `(observer, layer_id, node)` exposing `on_pre_forward_layer`, `on_forward_layer`, `on_backward_layer`, and `on_grad_weights`. Currently unused by `ModularObserver`.

---

## Known Limitations & Caveats

See [`reports/modularobserver_analysis.md`](file:///home/cesare/Projects/PositronixLab/reports/modularobserver_analysis.md) for the historical audit. Remaining caveats:

1. **Execution-order edges**: Dynamic edges reflect forward-hook invocation order (`sequence[-1] -> current`), not tensor dataflow, so parallel branches produce sequential edges and skip connections are omitted.
2. **Probabilistic `trace_samples` (float)**: Currently does not perform a random draw and decrements to `0` after the first pass. Use integer budgets (`0`, `1`, `N`) or `TRACE_ALWAYS`.
3. **Import-time dependencies**: `modelmapper.py` has an unused `graphviz` import, `layerobserver.py` has an unused `pymoo` import, and `modularobserver.py` imports `PipelineConfig` (pulling in `parallelazy`) at module load time.
4. **Orphaned `layerobserver.py`**: Still contains the old late-binding closure in `on_grad_weights` (fixed in `ModularObserver`).
