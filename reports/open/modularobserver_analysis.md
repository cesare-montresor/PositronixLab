# `positronixlab.modularobserver` — Open Issues Analysis

> **Status (2026-11-27 — `v0.1.3` / `5740a0c` + working tree):** **27 passed, 5 xfailed, 0 failed** (`tests/modularobserver/`).
> **Solved issues archive:** Resolved hook lifecycle, closure, `.device`, and import bugs (#1, #2, #3-`ModularObserver`, #4, #6, #8-`nn.Identity`, v0.1.3 A5) moved to [solved/modularobserver_solved.md](file:///home/cesare/Projects/PositronixLab/reports/solved/modularobserver_solved.md).

---

## 1. Open Bugs & Defects

### High / Medium Severity

1. **`ModelMapper` float `trace_samples` (probabilistic tracing) is non-functional (#5)**
   - **Location:** `modelmapper.py:467-476, 525-526` (xfail `test_modularobserver.py:341`)
   - **Details:** `trace_context()` never draws `random.random()`, and `_ActiveTraceContext.__exit__()` decrements any float in `(0, 1]` to `0` after the first pass.

2. **`LayerObserver.on_{pre_forward,forward,backward}_layer` leaks handles on re-registration (#3 remaining)**
   - **Location:** `layerobserver.py:55, 62, 69`
   - **Details:** Overwrites `self.<kind>_layer_hook` without calling `.remove()` on the existing handle first.

3. **`ModularObserver.map()` returns `None` despite `-> Dict[str, List]` annotation (#7)**
   - **Location:** `modularobserver.py:114-123`, `modelmapper.py:314` (xfail `test_modularobserver.py:432`)
   - **Details:** `ModularObserver.map()` omits `return`; `ModelMapper.map()` is annotated `-> Dict[str, List]` but returns `ModelGraph`.

4. **`ModelNode.to_dict()` crashes if `activation_fn` lacks `__name__` (#9)**
   - **Location:** `modelmapper.py:113` (xfail `test_modularobserver.py:235`)
   - **Details:** Fails on callable objects (`nn.ReLU()`, `functools.partial`, custom callables); should use `getattr(fn, "__name__", type(fn).__name__)`.

5. **`_MapperInterceptor` misses common activations (#10)**
   - **Location:** `modelmapper.py:242` (xfail `test_modularobserver.py:353`)
   - **Details:** Hardcoded 8-op set omits `F.leaky_relu`, `F.elu`, `F.prelu`, `F.selu`, `F.mish`, `F.hardswish`, etc.

6. **Dynamic tracing never updates node depths & re-mapping never prunes removed layers (#11–#12)**
   - **Location:** `modelmapper.py:345, 378-381, 393, 441-458`; `modularobserver.py:91-100`
   - **Details:** Static depths are never overwritten during dynamic tracing, and `self._has_run_static = True` is irreversible so re-mapping a mutated/new model accumulates stale nodes/edges.

7. **`extract_layer_io_metrics` fails on `nn.GroupNorm` (#13)**
   - **Location:** `modelmapper.py:41-46` (xfail `test_modularobserver.py:144`)
   - **Details:** `GroupNorm` exposes `num_channels`, not `num_features`/`normalized_shape`. `ConvTranspose3d`, `BatchNorm3d`, `InstanceNorm*`, `LazyLinear`, `Bilinear` are also unhandled.

8. **Shared-module collisions in `nodes_by_layer` (v0.1.3 B6)**
   - **Location:** `modularobserver.py:91-101`
   - **Details:** Keying `self.nodes_by_layer[node.module_ref] = node` overwrites earlier nodes when a module is reused across multiple graph positions. Line 97 (`module_ref = node.module_ref`) is dead code (`F841`).

9. **Top-level `PipelineConfig` import couples `ModularObserver` to `parallelazy` (#14 remaining)**
   - **Location:** `modularobserver.py:8`
   - **Details:** `from positronixlab.pipeline import PipelineConfig` is only used as a type hint for unused `self.config`; move under `if TYPE_CHECKING:`.

### Low Severity & Hygiene

10. **Cytoscape label emits blank `activation_fn: ` line (#15):** `modelmapper.py:113, 200-202`.
11. **Redundant `__hash__`/`__eq__` on frozen dataclass `ModelEdge` (#16):** `modelmapper.py:133-142`.
12. **Stale test callback in `test_modular_observer_on_grad_weights_closure_bug`:** `tests/modularobserver/test_modularobserver.py:443` — library closure bug is already fixed, but the test's callback returns a tuple (`TypeError: expected Variable, but hook returned 'tuple'`); fix the test and drop `@pytest.mark.xfail`.
13. **Minor hygiene:** `TRACE_ALWAYS` omitted from `__init__.py`; one-line `if` (`E701`) at `modularobserver.py:88`; unused `forward_weights_hook`/`backward_weights_hook` attributes in `layerobserver.py:34-35`; orphaned `LayerObserver` & stale `ModularObserver` docstrings; coupling to legacy `spacemath.LayerType`.

---

## 2. Open Design Notes

- **Execution-order edge wiring vs true dataflow DAG (`modelmapper.py:486-498`):** Consecutive pre-hook wiring creates phantom edges across parallel branches and misses residual skip connections; tensor-ID tracking would yield a true DAG.
