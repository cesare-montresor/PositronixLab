# `positronixlab.modularobserver` — Open Issues Analysis

> **Status (2026-11-27):** **32 passed, 0 xfailed, 0 failed** (`tests/modularobserver/`).
> **Solved issues archive:** Resolved hook lifecycle, closure, `.device`, import, and P1 bugs (#1–#7, #8-`nn.Identity`, #9, #10, #13, v0.1.3 A5) moved to [solved/modularobserver_solved.md](file:///home/cesare/Projects/PositronixLab/reports/solved/modularobserver_solved.md).

---

## 1. Open Bugs & Defects

### Medium Severity

1. **`LayerObserver.on_{pre_forward,forward,backward}_layer` leaks handles on re-registration (#3 remaining)**
   - **Location:** `layerobserver.py:55, 62, 69`
   - **Details:** Overwrites `self.<kind>_layer_hook` without calling `.remove()` on the existing handle first.

2. **Dynamic tracing never updates node depths & re-mapping never prunes removed layers (#11–#12)**
   - **Location:** `modelmapper.py:345, 378-381, 393, 441-458`; `modularobserver.py:91-100`
   - **Details:** Static depths are never overwritten during dynamic tracing, and `self._has_run_static = True` is irreversible so re-mapping a mutated/new model accumulates stale nodes/edges.

3. **Shared-module collisions in `nodes_by_layer` (v0.1.3 B6)**
   - **Location:** `modularobserver.py:91-101`
   - **Details:** Keying `self.nodes_by_layer[node.module_ref] = node` overwrites earlier nodes when a module is reused across multiple graph positions.

4. **Top-level `PipelineConfig` import couples `ModularObserver` to `parallelazy` (#14 remaining)**
   - **Location:** `modularobserver.py:8`
   - **Details:** `from positronixlab.pipeline import PipelineConfig` is only used as a type hint for unused `self.config`; move under `if TYPE_CHECKING:`.

### Low Severity & Hygiene

5. **Cytoscape label emits blank `activation_fn: ` line (#15):** `modelmapper.py:113, 200-202`.
6. **Redundant `__hash__`/`__eq__` on frozen dataclass `ModelEdge` (#16):** `modelmapper.py:133-142`.
7. **Additional layer types in `extract_layer_io_metrics`:** `ConvTranspose3d`, `BatchNorm3d`, `InstanceNorm*`, `LazyLinear`, `Bilinear` are unhandled.
8. **Minor hygiene:** one-line `if` (`E701`) at `modularobserver.py:88`; unused `forward_weights_hook`/`backward_weights_hook` attributes in `layerobserver.py:34-35`; orphaned `LayerObserver` & stale `ModularObserver` docstrings; coupling to legacy `spacemath.LayerType`.

---

## 2. Open Design Notes

- **Execution-order edge wiring vs true dataflow DAG (`modelmapper.py:486-498`):** Consecutive pre-hook wiring creates phantom edges across parallel branches and misses residual skip connections; tensor-ID tracking would yield a true DAG.
