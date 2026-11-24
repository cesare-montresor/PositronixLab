# `positronixlab.modularobserver` — Code Review & Architecture Analysis

## 1. Overview & Refactor Status

`positronixlab/modularobserver` provides model topology discovery and PyTorch hook management for observing neural network activations and gradients without modifying model code.

| File | Role | Status | Primary Consumers |
| :--- | :--- | :--- | :--- |
| `__init__.py` | Package entry point | Empty (0 bytes; no public exports) | — |
| `modelmapper.py` | Static + `TorchFunctionMode` dynamic graph topology engine (`ModelNode`, `ModelEdge`, `ModelGraph`, `GraphFormatter`, `_MapperInterceptor`, `ModelMapper`, `TRACE_ALWAYS`, `extract_layer_io_metrics`) | Active (v2 core) | `modularobserver.py`, `layerobserver.py` |
| `modularobserver.py` | High-level observation manager (`ModularObserver`) wrapping `ModelMapper` and attaching module/parameter hooks | Active (mid-refactor) | `tasks/compositionality/atlas_check.py`, external `AtlasCompose` plugins |
| `layerobserver.py` | Per-layer hook wrapper (`LayerObserver`) | Orphaned / unused by `ModularObserver` | Only `layerobserver_old.py` |
| `layerobserver_old.py` | Archived PubSub observer (`LayerObserverPubSub`) | Legacy / Do not touch | None (pre-`parallelazy` split) |

**Refactor status**: `modelmapper.py` has been rewritten cleanly around `ModelGraph` / `GraphFormatter` / `TorchFunctionMode`, and `ModularObserver` was refactored to manage hooks directly on `ModelNode.module_ref` instead of instantiating `LayerObserver` objects. However, the transition is incomplete:
- `LayerObserver` (`layerobserver.py`) is still present but unused by `ModularObserver`, while `ModularObserver`'s docstrings still claim to manage and return `LayerObserver` instances.
- Several hook lifecycle bugs (closure capture, leaked handles, missing removal in `clear()`) and unfinished `ModelMapper` features (float `trace_samples`, `.device` assumption) remain in active code.

---

## 2. Bugs & Defects

### Critical / High Severity

1. **Late-binding closure bug in `on_grad_weights` (`ModularObserver` and `LayerObserver`)**
   - **Files**: `modularobserver.py:125`, `layerobserver.py:42`
   - **Root cause**:
     - In `modularobserver.py:125`:
       ```python
       hooks = [param.register_hook(lambda grad: callback(module_ref, param, grad)) for param in module_ref.parameters()]
       ```
       Both `module_ref` (outer `for node in self.nodes_list:` variable) and `param` (list comprehension variable) are captured by reference.
     - In `layerobserver.py:42`:
       ```python
       [param.register_hook(lambda grad: callback(self.layer, param, grad)) for param in self.layer.parameters()]
       ```
       `param` is captured by reference.
   - **Impact**: During backward, every parameter hook in `ModularObserver` passes the **last layer's `module_ref` and last `param`** regardless of which layer or parameter's gradient is being computed. In `LayerObserver`, every parameter hook passes the layer's last parameter (e.g., `bias` when invoked for `weight`). Gradient-to-parameter attribution is completely corrupted.

2. **`LayerObserver.on_grad_weights(None)` crashes on backward and never tracks or removes handles**
   - **File**: `layerobserver.py:41-42`
   - **Root cause**: Unlike `on_pre_forward_layer`, `on_forward_layer`, and `on_backward_layer`, `on_grad_weights` has no `if callback is None:` guard and discards the list of `RemovableHandle`s instead of storing them on `self.backward_weights_hook`.
   - **Impact**: Passing `None` to remove hooks actually registers a broken lambda that raises `TypeError: 'NoneType' object is not callable` on the next backward pass. Repeated registrations leak parameter hooks permanently.

3. **Hook leaks when re-registering callbacks without clearing first**
   - **Files**: `modularobserver.py:124-126, 136-138, 148-150, 160-162`; `layerobserver.py:48-49, 55-56, 62-63`
   - **Root cause**: Registering a new callback overwrites `self.hooks[hook_name]` (or `self.<kind>_layer_hook`) without calling `.remove()` on the previously stored handle(s).
   - **Impact**: Old hooks remain permanently attached to `nn.Module` / `nn.Parameter`, firing both old and new callbacks on every pass, retaining stale closures in memory, and becoming unreachable to later `on_*(None)` or `clear()` calls.

4. **`ModularObserver.clear()` fails to remove pre-forward hooks**
   - **File**: `modularobserver.py:165-172`
   - **Root cause**: `clear()` calls `self.on_forward_layer(None)`, `self.on_backward_layer(None)`, and `self.on_grad_weights(None)`, omitting `self.on_pre_forward_layer(None)`. It then resets `self.nodes_list = []`.
   - **Impact**: Any registered pre-forward hooks remain attached to the model after `clear()`, leave orphaned entries in `self.hooks`, and can no longer be removed via `on_pre_forward_layer(None)` because `self.nodes_list` is now empty.

5. **`ModelMapper` float `trace_samples` (probabilistic tracing) is non-functional**
   - **Files**: `modelmapper.py:294-296, 354-359, 465-474, 523-524`
   - **Root cause**: Docstrings and `_validate_trace_budget()` document float `trace_samples` in `(0, 1]` as a standing per-call probability ("that draw happens inside `trace_context()` ... not consumed/decremented"), and `random` is imported at line 4. However:
     - `trace_context()` (lines 465-474) never performs a `random.random()` draw — it treats any non-zero float as always active.
     - `_ActiveTraceContext.__exit__()` (lines 523-524) unconditionally executes `self.mapper.trace_budget = max(0, self.mapper.trace_budget - 1)` for all values `!= TRACE_ALWAYS`, immediately reducing any float in `(0, 1]` to `0` after the first pass.
   - **Impact**: Passing e.g. `trace_samples=0.5` deterministically traces once and disables tracing forever. Negative ints other than `-1` (`TRACE_ALWAYS`) are also unvalidated: they trace once and clamp to `0`.

6. **`ModelMapper.map()` crashes on standard `nn.Module` instances lacking `.device`**
   - **File**: `modelmapper.py:365`
   - **Root cause**: `_ = model(data.to(device=model.device))` assumes `model` defines a `.device` attribute (true for `MetaModel`/`BaseModel`, false for standard PyTorch `nn.Module`, `nn.Sequential`, `torchvision` models, etc.).
   - **Impact**: Raises `AttributeError: '...' object has no attribute 'device'` on any standard PyTorch module. Notably, `ModularObserver.__init__` (line 32) already uses `getattr(model, 'device', torch.device("cpu"))`, but `ModelMapper.map()` does not.

### Medium Severity

7. **`ModularObserver.map()` returns `None` despite signature and docstring promising `Dict[str, List]`**
   - **File**: `modularobserver.py:94-103` (and `modelmapper.py:316`)
   - **Root cause**: `ModularObserver.map()` is annotated `-> Dict[str, List]` and its docstring states it "returns the ready Cytoscape dictionary", yet line 103 omits `return`. Additionally, `ModelMapper.map()` is annotated `-> Dict[str, List]` (line 316) while actually returning `self.graph: ModelGraph`.
   - **Impact**: Callers relying on the return value of `ModularObserver.map()` receive `None`; static type checkers are misled by both signatures.

8. **`nn.Identity` and `nn.Bilinear` misclassified as linear layers via `spacemath.common.is_linear` / `is_layer`**
   - **Files**: `modularobserver.py:52-62`, `modelmapper.py:403` (delegating to `spacemath/common.py:61-75` and `spacemath/spacemath.py:15-31`)
   - **Root cause**: `is_linear` checks `module.__class__.__module__ == 'torch.nn.modules.linear'`, which includes `nn.Identity`, `nn.Bilinear`, and `nn.LazyLinear`, while excluding any custom `nn.Linear` subclass.
   - **Impact**:
     - `ModelMapper` (under `ModularObserver._is_suitable_layer`) includes `nn.Identity` in the graph with `layer_type='linear'` and `unit_type='unknown'`.
     - `ModularObserver._is_observable_layer` excludes `nn.Identity` only incidentally via `len(list(module.parameters())) == 0` (line 58). `nn.Bilinear` passes `_is_observable_layer` despite having unmapped I/O metrics (`in_units=None, out_units=None`), whereas custom `nn.Linear` subclasses are silently ignored.

9. **`ModelNode.to_dict()` crashes if `activation_fn` lacks `__name__`**
   - **File**: `modelmapper.py:115`
   - **Root cause**: `str(self.activation_fn.__name__)` fails when `activation_fn` is a callable object without `__name__` (e.g., `nn.ReLU()`, `functools.partial`, custom callable class), even though `activation_fn` is typed `Optional[Callable]`.
   - **Impact**: Raises `AttributeError` during `to_dict()` and `GraphFormatter.to_cytoscape()`.

10. **`_MapperInterceptor` misses common activations (`F.leaky_relu`, `F.elu`, `F.prelu`, `F.selu`, `F.mish`, `F.hardswish`, `F.hardswish`, etc.)**
    - **File**: `modelmapper.py:244`
    - **Root cause**: `self.target_ops` is a hardcoded 8-element set `{F.relu, F.gelu, F.silu, F.softmax, F.log_softmax, torch.relu, torch.sigmoid, torch.tanh}`.
    - **Impact**: `nn.LeakyReLU()` (which calls `F.leaky_relu`) — despite being one of the 4 official activations in `spacemath.ActivationType` — is never discovered by dynamic tracing (`activation_fn` stays `None`).

11. **Dynamic tracing never updates node depths after static mapping**
    - **File**: `modelmapper.py:347, 391, 408, 426-464`
    - **Root cause**: `map()` always runs `_static_mapping()` first, assigning `depth = 0, 1, 2, ...` to every leaf node in `named_modules()` order. During dynamic tracing, `_add_node(..., default_depth=None)` preserves existing depths in `ModelNode.update_from()`, and `_assign_dynamic_depths()` only writes depths for nodes where `node.depth is None`.
    - **Impact**: If a model's `forward()` executes layers in an order different from `named_modules()` registration order, dynamic tracing leaves the stale static depths untouched.

12. **Re-mapping a new or mutated model never re-runs static mapping or prunes removed layers**
    - **File**: `modelmapper.py:346-347, 371-379, 92-96`; `modularobserver.py:71-80`
    - **Root cause**: `self._has_run_static = True` is irreversible. Passing a different `model` to `ModelMapper.map(other_model)` skips `_static_mapping()` and accumulates nodes/edges from both models in `self.graph`. `ModularObserver._sync_observers_with_mapper()` also never removes deleted nodes or updates `nodes_by_layer` if a module was swapped in-place. Additionally, `ModelNode.update_from()` (line 94) ignores `new_val is None`, so structural attributes cannot reset to `None`.
    - **Impact**: Stale nodes/edges and stale `nodes_by_layer` references persist across model mutations or re-mapping.

13. **`extract_layer_io_metrics` fails on `nn.GroupNorm` despite explicitly checking for it**
    - **File**: `modelmapper.py:43-48`
    - **Root cause**: `nn.GroupNorm` is listed in `isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm))`, but `GroupNorm` exposes `num_channels` rather than `num_features` or `normalized_shape`.
    - **Impact**: `nn.GroupNorm` falls through both branches and returns `in_units=None, out_units=None, unit_type='unknown'`. `nn.ConvTranspose3d`, `nn.BatchNorm3d`, `nn.InstanceNorm*`, `nn.LazyLinear`, and `nn.Bilinear` are also unhandled.

### Low Severity / Hygiene

14. **Unused heavyweight / external imports at module load time**
    - `modelmapper.py:1`: `from graphviz.dot import node` (completely unused; requires `graphviz` installed).
    - `layerobserver.py:2`: `from pymoo.core import callback` (completely unused; requires `pymoo` installed).
    - `modularobserver.py:8`: `from positronixlab.pipeline import PipelineConfig` (only used as a type hint for unused `self.config`; forces importing `positronixlab.pipeline` and `parallelazy` at runtime).
    - `layerobserver.py:7, 11`: unused imports `is_activation, is_conv, is_linear, is_layer, ActivationType, LayerType, SpaceMath`.
    - `modularobserver.py:6, 7`: unused imports `ModelEdge`, `is_conv`.

15. **Cytoscape label emits blank `activation_fn: ` line** (`modelmapper.py:115, 202-204`): `ModelNode.to_dict()` serializes missing `activation_fn` as `''`, and `node_to_cytoscape()` filters with `is not None`, producing a trailing `"activation_fn: "` line on every node without a discovered activation.

16. **Redundant dunder methods on frozen dataclass `ModelEdge`** (`modelmapper.py:123-144`): `@dataclass(frozen=True)` already generates `__hash__` and `__eq__`.

---

## 3. Inconsistencies & Design Issues

- **`_is_observable_layer()` restricts observation to linear layers only (`modularobserver.py:56-62`)**: `ModelMapper` maps Linear, Conv, RNN, and Transformer modules (via `is_layer`), and `extract_layer_io_metrics` extracts units for all of them, yet `ModularObserver._is_observable_layer()` hardcodes `return is_linear(module) # todo: support all architectures`. Conv/RNN/Transformer layers appear in `observer.graph()` but are silently excluded from `layer_names()` and all hooks.
- **Execution-order edge wiring vs. true dataflow DAG (`modelmapper.py:486-498`)**: `ModelMapper` wires edges strictly between consecutive forward pre-hook calls (`sequence[-1] -> mod_name`). Parallel branches (`a = f(x); b = g(x); c = h(a + b)`) produce a phantom edge `f -> g` and drop `f -> h`; residual skip connections (`x + f(x)`) are invisible; functional activations are always attributed to `sequence[-1]` even across branches or when `is_suitable_layer` admits `nn.ReLU` itself.
- **Orphaned `LayerObserver` and stale `ModularObserver` docstrings**: `ModularObserver` docstrings (lines 24, 72, 87, 91) claim it manages and returns `LayerObserver` objects, whereas `by_name()` / `by_layer()` actually return `ModelNode` objects and `LayerObserver` is never used.
- **Empty `__init__.py`**: Callers must reach into internal submodules (`positronixlab.modularobserver.modularobserver`, `positronixlab.modularobserver.modelmapper`) instead of importing from `positronixlab.modularobserver`.
- **Inaccurate type annotations and docstrings**:
  - `ModelMapper.nodes()` annotated `-> Dict[str, ModelNode]` (returns `Dict[str, Dict[str, Any]]`).
  - `ModelMapper.edges()` annotated `-> List[ModelEdge]` (returns `List[Dict[str, Any]]`).
  - `ModelMapper.map()` annotated `-> Dict[str, List]` (returns `ModelGraph`).
  - `GraphFormatter.to_dict()` docstring says `'nodes'` is a list (it is a `dict`); `nodes_list()` docstring says "keyed by node id" (it is a `list`).
  - `ModularObserver.hooks` annotated `Dict[str, RemovableHandle]` (actually stores `RemovableHandle | List[RemovableHandle]`).
  - `ModularObserver.config`, `ModularObserver.device`, `ModularObserver.latest_graph_dict`, `LayerObserver.forward_weights_hook`, `LayerObserver.backward_weights_hook` are stored and never read.
- **Coupling to legacy `spacemath.py`**: `modelmapper.py:12` imports `LayerType` from legacy `positronixlab.spacemath.spacemath` rather than a dedicated enum or `isinstance` checks.

---

## 4. Opportunities & Recommendations

1. **Fix hook safety & lifecycle first (highest priority)**:
   - Bind loop variables explicitly in `on_grad_weights` (`lambda grad, m=module_ref, p=param: callback(m, p, grad)`).
   - Remove any existing hook handle(s) before overwriting in `on_pre_forward_layer`, `on_forward_layer`, `on_backward_layer`, and `on_grad_weights`.
   - Add `self.on_pre_forward_layer(None)` to `ModularObserver.clear()` and clear `self.hooks`.
   - Fix `LayerObserver.on_grad_weights` to store handles and support `callback=None` — or deprecate/remove `LayerObserver` if `ModularObserver` owns hooks directly.
2. **Make `ModelMapper` work on any PyTorch `nn.Module`**:
   - Infer device safely from `next(model.parameters()).device` (with `getattr(model, "device", None)` and CPU fallback) instead of `model.device`.
   - Implement the documented `random.random() < self.trace_budget` check in `trace_context()` and only decrement integer budgets (`isinstance(budget, int) and budget > 0`).
   - Expand `_MapperInterceptor.target_ops` to include `F.leaky_relu`, `F.elu`, `F.selu`, `F.prelu`, `F.mish`, `F.hardswish`, `F.hardsigmoid`, and guard `ModelNode.to_dict()` with `getattr(fn, "__name__", type(fn).__name__)`.
   - Handle `nn.GroupNorm.num_channels` in `extract_layer_io_metrics` and replace `spacemath.common.is_linear` with `isinstance` checks so `nn.Identity` is excluded and `nn.Linear` subclasses work.
3. **Decouple runtime imports & expose a clean public API**:
   - Remove stray `graphviz` and `pymoo` imports.
   - Put `PipelineConfig` under `if TYPE_CHECKING:` (or drop `config` if unused) so `ModularObserver` does not require `parallelazy` at import time.
   - Re-export `ModularObserver`, `ModelMapper`, `ModelGraph`, `ModelNode`, `ModelEdge`, `GraphFormatter`, `TRACE_ALWAYS` from `positronixlab/modularobserver/__init__.py`.
4. **Upgrade graph topology & multi-architecture observation**:
   - Track tensor `id()`s (via forward hooks returning output tensor IDs matched against downstream input tensor IDs) to discover true DAG edges (branches, skip connections) instead of linear execution order.
   - Allow configurable observable layer predicates (e.g., `Linear` + `Conv` + `RNN` + `Transformer`) in `ModularObserver` once downstream `ModularSpace`/`LayerSpace` reducers are wired in.

---

## 5. Docs / Tests / Examples Summary

- **Docs created**:
  - `docs/MODULAROBSERVER_SYSTEM.md` — architecture, dataflow, module layout, API reference, and known limitations.
  - `docs/MODULAROBSERVER_USAGE.md` — practical guide for `ModelMapper`, `GraphFormatter`, `ModularObserver`, hook lifecycle, and workarounds for current caveats.
- **Tests created**:
  - `tests/test_modularobserver.py` — unit tests covering `extract_layer_io_metrics`, `ModelNode`/`ModelEdge`/`ModelGraph`, `GraphFormatter`, `ModelMapper`, `ModularObserver`, and `LayerObserver`, including strict `@pytest.mark.xfail` tests pinning each confirmed bug.
- **Example created**:
  - `examples/example_modularobserver.py` — self-contained CPU example demonstrating static/dynamic mapping, Cytoscape export, and `ModularObserver` hook collection & cleanup.
