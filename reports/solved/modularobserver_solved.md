# `positronixlab.modularobserver` — Solved Issues Archive

> **Archived:** 2026-11-27 (`v0.1.3` / `5740a0c` + working tree).
> **Active open issues:** See [modularobserver_analysis.md](file:///home/cesare/Projects/PositronixLab/reports/modularobserver_analysis.md).

---

## Resolved Bugs & Improvements

| ID | Summary | Resolution Evidence |
| :--- | :--- | :--- |
| **#1 / v0.1.3 A5** | Late-binding closure bug in `on_grad_weights` (`ModularObserver` & `LayerObserver`) | Fixed default-arg capture in `modularobserver.py:145` (`m=module_ref, p=param`) and `layerobserver.py:47` (`p=param`). *(Note: `test_modular_observer_on_grad_weights_closure_bug` still xfails only due to a test-callback tuple return bug.)* |
| **#2 / v0.1.3 A5** | `LayerObserver.on_grad_weights(None)` crashed & leaked handles | Tracks `_weight_grad_handles` and handles `callback is None` (`layerobserver.py:36, 39-48`); passing tests `test_modularobserver.py:530, 554`. |
| **#3 (ModularObserver)** | Re-registering callbacks leaked old hook handles | `ModularObserver.on_*` (`modularobserver.py:139-181`) removes old handles before overwriting; stale xfail removed. *(Still open in `LayerObserver.on_{pre_forward,forward,backward}_layer`.)* |
| **#4** | `ModularObserver.clear()` leaked pre-forward hooks | `modularobserver.py:189` now calls `self.on_pre_forward_layer(None)`; stale xfail removed. |
| **#6** | `ModelMapper.map()` crashed on standard `nn.Module` lacking `.device` | Falls back to `next(model.parameters(), None)` → CPU (`modelmapper.py:360-363`); stale xfail removed. |
| **#8 (`nn.Identity`)** | `nn.Identity` misclassified as linear layer | Fixed via `spacemath/common.py:64-65, 78-79` (`test_modularobserver.py:393`). *(`nn.Bilinear` still open.)* |
| **#14 (partial)** | Unused heavyweight imports (`graphviz`, `pymoo`, `ModelEdge`, `is_conv`, `spacemath`) | Removed; archived `layerobserver_old.py` deleted. *(Top-level `PipelineConfig` import still open.)* |
| **Design** | Linear-only `_is_observable_layer` & empty `__init__.py` | `ModularObserver.__init__` generalized with `observable_types`, `target_layers`, `exclude_patterns` (`modularobserver.py:25-45, 64-82`); `__init__.py` populated (`__init__.py:1-41`) with lazy `ModularObserver` export. |
| **Cleanup (2026-11-27)** | Dead `module_ref` (`F841`), missing `TRACE_ALWAYS` export, broken test callback in `test_modular_observer_on_grad_weights_closure_bug` | Removed unused `module_ref` (`modularobserver.py:97`); exported `TRACE_ALWAYS` in `__init__.py`; fixed test callback and removed stale `xfail` (`test_modularobserver.py:443`). |
| **#5 (P1, 2026-11-27)** | `ModelMapper` float `trace_samples` (probabilistic tracing) was never sampled and was decremented to `0` | Added probabilistic `random.random() >= self.trace_budget` check in `trace_context()` and guarded budget decrement to `int` only (`modelmapper.py`); xfail removed and passing. |
| **#7 (P1, 2026-11-27)** | `ModularObserver.map()` returned `None` instead of the mapped graph | Added `return self.graph()` and updated return type annotation (`modularobserver.py`); xfail removed and passing. |
| **#9 (P1, 2026-11-27)** | `ModelNode.to_dict()` crashed when `activation_fn` lacked `__name__` | Used `getattr(self.activation_fn, '__name__', type(self.activation_fn).__name__)` (`modelmapper.py`); xfail removed and passing. |
| **#10 (P1, 2026-11-27)** | `_MapperInterceptor.target_ops` omitted `F.leaky_relu`, `F.elu`, `F.selu`, `F.prelu`, `F.mish`, `F.hardswish`, `F.hardsigmoid` | Added all 7 activation ops to `target_ops` (`modelmapper.py`); xfail removed and passing. |
| **#13 (P1, 2026-11-27)** | `extract_layer_io_metrics()` missed `nn.GroupNorm` (`num_channels`) | Added `elif hasattr(module, 'num_channels'):` branch (`modelmapper.py`); xfail removed and passing. |
