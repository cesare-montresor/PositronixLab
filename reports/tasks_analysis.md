# `positronixlab/tasks` — Module Analysis

## 1. Overview & Refactor Status

`positronixlab/tasks` is intended to house **downstream evaluation tasks** that validate whether the emergent functional modularity discovered by `ModularSpace` / `ModularCluster` translates into causally separable, composable subnetworks.

Currently, the module contains a single task family, `compositionality/`, with three active source files and one fixture-dependent example:

| File | Role | Status |
|---|---|---|
| `positronixlab/tasks/__init__.py` | Package init | Empty (0 bytes) |
| `positronixlab/tasks/README.md` | Module overview | Empty (0 bytes) |
| `positronixlab/tasks/compositionality/__init__.py` | Subpackage init | Empty (0 bytes) |
| `positronixlab/tasks/compositionality/README.md` | Subpackage overview | Empty (0 bytes) |
| `positronixlab/tasks/compositionality/atlas_check.py` | Cluster-to-class attribution (`AtlasCheck`), modularity metrics (purity, entropy, KL divergence), and cluster-based binary mask extraction | Partial WIP; recent KL fix, several integration & lifecycle bugs |
| `positronixlab/tasks/compositionality/weightmask_check.py` | Csordás-style differentiable Gumbel-Sigmoid per-class weight mask optimization (`WeightmaskCheck`, `MultiClassWeightmaskManager`, `WeightMaskCriterion`) and mask metrics (`WeightMasksMetrics`) | Partial WIP; core loop bugs (`num_epochs`, empty class slices, `requires_grad` leak) |
| `positronixlab/tasks/compositionality/compositionality.py` | Inference-time mask application (`StaticMaskApplier`), confusion-matrix evaluation (`CompositionalityCheck`), and end-to-end orchestrator (`CompositionalityPipeline`) | Partial WIP; broken handoff from `AtlasCheck` to `StaticMaskApplier` |
| `positronixlab/tasks/compositionality/examples/example_atlas.py` | Script loading `trained_model.pth` & `all_vars.pkl` to run `CompositionalityPipeline` | Stale / CWD-dependent |

**Refactor status**: Imports were updated during the `trial` -> `pipeline` rename (`a5b9b30`, `2e947b8`), and `kl_divergence` in `atlas_check.py` was replaced with manual epsilon-smoothed KL. However, the three files were developed somewhat independently and have **not yet been integration-tested together**: end-to-end execution of `CompositionalityPipeline` fails or silently produces invalid results due to mask shape/key mismatches and leaked forward hooks.

---

## 2. Bugs & Defects

### Critical / High Impact

1. **Atlas mask key & shape mismatch with `StaticMaskApplier` (breaks Atlas evaluation in `CompositionalityCheck` / `CompositionalityPipeline`)**
   - **Locations**:
     - `positronixlab/tasks/compositionality/atlas_check.py:229-242` (`AtlasCheck.extract_weightmasks`)
     - `positronixlab/tasks/compositionality/compositionality.py:27-37` (`StaticMaskApplier.__enter__`)
     - `positronixlab/tasks/compositionality/compositionality.py:216-232` (`CompositionalityPipeline.run_full_evaluation`)
   - **Root cause**:
     1. *Key mismatch*: `AtlasCheck.extract_weightmasks()` stores masks under raw dotted module names (e.g. `"nn_latent.0"`), whereas `StaticMaskApplier.__enter__()` checks `safe_name = name.replace('.', '_')` (`if safe_name in self.masks:`). For any nested module, Atlas masks are **silently skipped** and the unmasked model is evaluated instead.
     2. *Shape / broadcast bug*: `AtlasCheck.extract_weightmasks()` produces 1D neuron-level boolean masks of shape `(out_features,)`, whereas `StaticMaskApplier` computes `orig_w * m_mask` where `orig_w` has shape `(out_features, in_features)`. When `out_features != in_features`, PyTorch raises `RuntimeError` (shape mismatch). Worse, when `out_features == in_features`, PyTorch **silently broadcasts along columns (input features) instead of rows (output neurons)** — masking the wrong axis.
   - **Impact**: `CompositionalityPipeline.run_full_evaluation()` either crashes on top-level non-square layers, silently corrupts masking on top-level square layers, or silently applies no mask at all on nested layers.

2. **`WeightmaskCheck.run()` ignores `num_epochs` completely**
   - **Location**: `positronixlab/tasks/compositionality/weightmask_check.py:18, 33`
   - **Root cause**: `num_epochs: int = 5` is accepted in the signature (and forwarded as `csordas_epochs` by `CompositionalityPipeline` and `example_atlas.py`), but there is no `for epoch in range(num_epochs):` loop — only a single pass over `self.dataprovider.train`.
   - **Impact**: Mask optimization always runs for exactly 1 epoch regardless of configuration.

3. **Empty class slice in `WeightmaskCheck.run()` produces `NaN` loss (and crashes if `AtlasCheck` hooks are attached)**
   - **Location**: `positronixlab/tasks/compositionality/weightmask_check.py:41-56`
   - **Root cause**: Inside `for c in range(num_classes):`, `mask = (y == c)` is never checked for emptiness (`if not mask.any(): continue`). When any batch lacks samples of class `c`, `x_c` has shape `(0, ...)` and `nn.CrossEntropyLoss()(logits, y_c)` returns `NaN` (making `task_loss` and `total_loss` `NaN`). While empty-batch matmul gradients stay zero for `Linear` weights, the L1 penalty still pushes mask logits toward zero on an uninformative step, and if `AtlasCheck`'s leaked forward hook (Bug #4) is still attached, `SpaceMath.activations_to_scores` crashes with `RuntimeError: quantile() input tensor must be non-empty`.
   - **Impact**: `NaN` losses, spurious L1-only mask decay on absent classes, and hard crashes when composed with `AtlasCheck`.

4. **`AtlasCheck` leaks forward hooks onto `self.model` permanently**
   - **Location**: `positronixlab/tasks/compositionality/atlas_check.py:117` (and `compositionality.py:209-225`)
   - **Root cause**: `AtlasCheck.__init__()` registers `self.observer.on_forward_layer(self._process_activations)` and provides no `remove_hooks()` / `close()` cleanup.
   - **Impact**: After `atlas.run()` finishes in `CompositionalityPipeline`, the hook remains attached during `WeightmaskCheck` and `CompositionalityCheck`, running `SpaceMath.activations_to_scores` on every forward pass. Combined with Bug #3 (empty `x_c`), `torch.quantile` inside `SpaceMath.activations_to_scores` crashes with `RuntimeError: quantile() input tensor must be non-empty`.

5. **`AtlasCheck._on_prediction()` can hang forever in an infinite busy-wait loop**
   - **Location**: `positronixlab/tasks/compositionality/atlas_check.py:146-147`
   - **Root cause**: `while any(score is None for score in self.latest_scores.values()): time.sleep(0.01)` assumes asynchronous hook execution. PyTorch forward hooks run synchronously during `self.model(x)` before `_on_prediction` is invoked. If any mapped layer is not executed during a forward pass (e.g. conditional branch or unused submodule), `self.latest_scores[name]` stays `None` and the process deadlocks permanently.
   - **Impact**: Unrecoverable hang instead of a clear error or skip.

### Medium Impact

6. **`MultiClassWeightmaskManager.remove_hooks()` does not restore `requires_grad` and is not exception-safe**
   - **Locations**: `positronixlab/tasks/compositionality/weightmask_check.py:28-68, 95-96, 144-159`
   - **Root cause**: `_setup_hooks_and_freezing()` sets `param.requires_grad = False` on all model parameters without saving their prior state, and `remove_hooks()` never restores `requires_grad`. Furthermore, `WeightmaskCheck.run()` does not wrap training in `try ... finally: mask_mgr.remove_hooks()`, so any exception leaves `module.weight` deleted/replaced by a non-`Parameter` tensor and hooks attached.
   - **Impact**: Caller's model is permanently mutated (frozen, or broken on exception).

7. **`MultiClassWeightmaskManager` crashes with `KeyError` on target layers lacking a `weight` attribute**
   - **Location**: `positronixlab/tasks/compositionality/weightmask_check.py:82-88, 98-117`
   - **Root cause**: `__init__` creates `self.masks[safe_name]` only `if hasattr(module, 'weight')`, but `_setup_hooks_and_freezing()` registers `pre_hook` unconditionally for every module in `target_layers`. When `pre_hook` executes, `self.masks[s_name]` raises `KeyError`.
   - **Impact**: Passing any parameterless layer (e.g. `nn.ReLU`, `nn.Dropout`) in `target_layers` crashes on forward.

8. **`AtlasCheck.modularity_score()` raises `ZeroDivisionError` when `cluster_class_stats` is empty**
   - **Location**: `positronixlab/tasks/compositionality/atlas_check.py:310-315`
   - **Root cause**: Computes `sum(...) / len(stats)` without checking `if stats:`.
   - **Impact**: Crashes if called before `run()` or on a model with no observable layers / zero activations.

9. **`CompositionalityCheck.run()` raises `ZeroDivisionError` on empty dataloader**
   - **Location**: `positronixlab/tasks/compositionality/compositionality.py:151`
   - **Root cause**: `untainted_cm.diag().sum().item() / untainted_cm.sum().item()` lacks the `if ... > 0 else 0.0` guard present at line 137.
   - **Impact**: Crashes on an empty evaluation split.

10. **`CompositionalityPipeline.run_full_evaluation()` crashes when `dataprovider.test` is `None`**
    - **Location**: `positronixlab/tasks/compositionality/compositionality.py:229`
    - **Root cause**: `DataProvider` defaults to a 2-way split (`[0.7, 0.3]`, `train` and `val` only), leaving `dataprovider.test` as `None`. `CompositionalityPipeline` passes `self.dataprovider.test` directly to `CompositionalityCheck` without falling back to `val` or raising a clear `ValueError`.
    - **Impact**: `TypeError: 'NoneType' object is not iterable` on default `DataProvider` instances.

11. **`AtlasCheck._cluster_class_attribution()` device mismatch when `clusters` and `model` are on different devices**
    - **Location**: `positronixlab/tasks/compositionality/atlas_check.py:183-189`
    - **Root cause**: `batch_ys` is moved to `self.model.device`, while `clusters` is never moved to `self.model.device`. Indexing `batch_clustering[idxs_on]` or concatenating `(clusters_on, ys_on)` raises `RuntimeError` if `clusters` is on CPU and `model` is on CUDA (or vice versa).
    - **Impact**: Fails in GPU setups unless caller manually moves every cluster tensor to `model.device`.

12. **Broken dead/WIP method `AtlasCheck._cluster_output_attribution()`**
    - **Location**: `positronixlab/tasks/compositionality/atlas_check.py:201-227`
    - **Root cause**: Line 212 multiplies `y_hats` (shape `[bs, 1, C]`) by `idx_on` (shape `[num_features]`) and line 220 casts vector `y_hats` to `int(y_hats)`. Never called.
    - **Impact**: Crashes immediately if invoked.

### Low Impact / Minor

13. **`WeightMasksMetrics.compute_sparsity()` computes density, not sparsity**
    - **Location**: `positronixlab/tasks/compositionality/weightmask_check.py:202-205`
    - **Root cause**: Returns `active_params / total_params` (proportion of *active* weights, i.e., density) rather than `1.0 - active_params / total_params` (proportion of pruned/zero weights).
    - **Impact**: Inverted metric interpretation if consumers expect standard ML sparsity (higher = more pruned).

14. **`DiscreteDistribution.update()` mutates `total_count` type from `int` to `Tensor`**
    - **Location**: `positronixlab/tasks/compositionality/atlas_check.py:39, 51-53`
    - **Root cause**: `self.total_count += count` where `count` is a 0D `Tensor` turns `self.total_count` into a `Tensor` (and fails if `class_labels` is on GPU while `self.counts` is on CPU).

15. **Unused imports & dead code**
    - `atlas_check.py:3`: `from importlib.metadata import distribution` (unused).
    - `weightmask_check.py:1`: `from math import log` (unused).
    - `weightmask_check.py:133-138`: `MultiClassWeightmaskManager.compute_spatial_loss()` duplicates `WeightMaskCriterion._compute_overall_weightmaskloss()` and is unused.
    - `weightmask_check.py:168`: `device=layer_masks[0].device if layer_masks else 'cpu'` inside `if not layer_masks:` (dead branch).
    - `example_atlas.py:2, 8, 11, 12, 22, 25`: unused imports (`time`, `CompositionalityCheck`, `AtlasCheck`, `WeightmaskCheck`) and unused attributes (`self.methods`, `self.dataprovider`).

16. **Hardcoded CWD-relative paths and wrong config class in `example_atlas.py`**
    - **Location**: `positronixlab/tasks/compositionality/examples/example_atlas.py:29, 35, 39`
    - **Root cause**: Line 29 instantiates base `PipelineConfig()` instead of `PipelineConfigV1()`, which fails at runtime with `AttributeError: 'PipelineConfig' object has no attribute 'task'` inside `DeeplearningBase` (and lacks `modular_space_active_q`). Defaults `'trained_model.pth'` and `'all_vars.pkl'` are also resolved relative to CWD instead of `Path(__file__).parent`.

---

## 3. Inconsistencies & Design Issues

1. **Empty `__init__.py` and `README.md` files**: Neither `positronixlab/tasks/` nor `positronixlab/tasks/compositionality/` exports any symbols, forcing callers to import deep internal module paths (`positronixlab.tasks.compositionality.atlas_check`).
2. **No shared `Task` / `Evaluator` abstraction**: Every class in `compositionality/` has a different constructor signature and argument order:
   - `AtlasCheck(config, model, dataprovider, clusters)`
   - `WeightmaskCheck(model, dataprovider)`
   - `CompositionalityCheck(model, dataloader, num_classes, device=None)`
   - `CompositionalityPipeline(model, dataprovider, config)`
3. **Inconsistent `model.device` contract**: `AtlasCheck` requires `model.device` (via `DeeplearningBase`), whereas `WeightmaskCheck` and `CompositionalityCheck` derive the device via `next(model.parameters()).device` and work with any standard `nn.Module`.
4. **Inconsistent mask conventions**:
   - `AtlasCheck.extract_weightmasks()` -> `Dict[int, Dict[str, Tensor]]`, dotted keys, 1D `(out_features,)` `torch.bool`.
   - `WeightmaskCheck.run()` -> `Dict[int, Dict[str, Tensor]]`, underscore-sanitized keys (`name.replace('.', '_')`), 2D `(out_features, in_features)` `torch.float32`.
   - `StaticMaskApplier` -> expects underscore-sanitized keys and weight-broadcastable tensors.
5. **Reproducibility gaps**: `MultiClassWeightmaskManager._get_computed_mask()` calls `torch.rand_like(mask_scores)` without a configurable `torch.Generator` or seed.
6. **Type annotation & comment hygiene**:
   - `atlas_check.py:103` annotates `cluster_class_stats: Dict[str, Dict[str, DiscreteDistribution]]`, but inner keys are `int` cluster IDs.
   - `weightmask_check.py:5` imports `DataProvider` indirectly from `dataprovider_factory` instead of `dataprovider`.
   - `compositionality.py:165` contains an Italian inline comment (`# 1. Normalizza per riga...`), and several docstrings/comments contain typos (`cordas`, `downstram`, `efficent`, `latente sapace`).
7. **Coupling to `ModularObserver` linear-only filter**: Because `AtlasCheck` relies on `ModularObserver`, it currently only observes `nn.Linear` layers; Conv/RNN/Transformer layers are ignored even though `SpaceMath.activations_to_scores` has a `LayerType.CONV` branch.

---

## 4. Opportunities & Recommendations

1. **Unify the mask contract (`MaskSpec` or canonical `(layer_name, tensor)` format)**:
   - Keep canonical PyTorch `named_modules()` dotted keys everywhere (only sanitize internally inside `nn.ParameterDict`, mapping back to dotted names on export).
   - Have `AtlasCheck.extract_weightmasks()` return broadcast-ready masks (e.g. shape `(out_features, 1)` for `nn.Linear`, or expand to `module.weight.shape`) or let `StaticMaskApplier` explicitly handle neuron masks vs. weight masks.
2. **Fix `WeightmaskCheck.run()` training loop & lifecycle**:
   - Add the missing `for epoch in range(num_epochs):` outer loop.
   - Guard empty class slices with `if not mask.any(): continue`.
   - Save and restore `param.requires_grad` inside `MultiClassWeightmaskManager.remove_hooks()`, and make it a context manager (`__enter__`/`__exit__`) used via `with` in `WeightmaskCheck.run()`.
3. **Make `AtlasCheck` hook lifecycle explicit**:
   - Add `close()` / `__enter__` / `__exit__` calling `self.observer.on_forward_layer(None)` so hooks are detached immediately after `run()`.
   - Remove the `while any(score is None ...): time.sleep(0.01)` busy-wait in `_on_prediction` and replace with an explicit check/skip.
   - Move `clusters` to `self.model.device` once in `_cluster_class_attribution` (or `__init__`).
4. **Define a lightweight `BaseTask` / `EvaluationResult` contract in `positronixlab/tasks/`**:
   - Standardize `run() -> Dict[str, Any]`, JSON/YAML-serializable metric summaries, and optional `seed` parameter so downstream tasks (compositionality, continual learning, pruning/ablation, OOD generalization) can be plugged into NAS (`NasExperiment`) or `AtlasCompose` uniformly.
5. **Re-export public classes in `__init__.py`**:
   - Export `AtlasCheck`, `WeightmaskCheck`, `WeightMasksMetrics`, `StaticMaskApplier`, `CompositionalityCheck`, `CompositionalityPipeline` from `positronixlab/tasks/compositionality/__init__.py` and `positronixlab/tasks/__init__.py`.
6. **Clarify `compute_sparsity` vs `compute_density`**:
   - Rename or add `compute_density()` and fix `compute_sparsity()` to return `1.0 - active / total` (with a deprecation note if external callers rely on it).
7. **Support `val` fallback & multi-metric reporting in `CompositionalityPipeline`**:
   - Allow passing an explicit `eval_dataloader` (defaulting to `dataprovider.test or dataprovider.val`).
   - Include `AtlasCheck.modularity_score()` and `WeightMasksMetrics` (IoU/IoMin/sparsity between Atlas and Csordás masks) in `CompositionalityPipeline.run_full_evaluation()` output.

---

## 5. Docs / Tests / Examples Summary

| Artifact | Path | Purpose |
|---|---|---|
| Module Analysis | `positronixlab/tasks/analysis.md` | This report |
| System Reference | `positronixlab/tasks/docs/TASKS_SYSTEM.md` | Architecture, dataflow, API reference, known caveats |
| Usage Guide | `positronixlab/tasks/docs/TASKS_USAGE.md` | Practical recipes, mask alignment workarounds, metrics guide |
| Unit Tests | `positronixlab/tasks/tests/test_tasks_compositionality.py` | Synthetic CPU tests covering distributions, `AtlasCheck`, `WeightmaskCheck`, `WeightMasksMetrics`, `StaticMaskApplier`, `CompositionalityCheck`, `CompositionalityPipeline`, plus strict `xfail` tests pinning confirmed bugs |
| Runnable CPU Example | `positronixlab/tasks/examples/example_compositionality.py` | Self-contained synthetic MLP example running without `.pkl`/`.pth` fixtures |
