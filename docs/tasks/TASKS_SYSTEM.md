# PositronixLab Tasks System

Downstream evaluation suite for validating whether emergent functional modularity
discovered in a model's embedding space (`ModularSpace` + `ModularCluster`) yields
causally separable, composable subnetworks.

Currently implements the **Compositionality** evaluation task, which compares
cluster-derived subnetwork masks (**Atlas**) against gradient-optimized
differentiable binary weight masks (**Csordás** baseline) via per-class
confusion-matrix analysis.

## Architecture

```
                        ┌──────────────────────┐
                        │ Trained Model + Data │
                        └──────────┬───────────┘
                                   │
             ┌─────────────────────┴─────────────────────┐
             ▼                                           ▼
┌─────────────────────────────┐             ┌─────────────────────────────┐
│ AtlasCheck                  │             │ WeightmaskCheck (Csordás)   │
│  • ModularObserver hooks    │             │  • MultiClassWeightmaskMgr  │
│  • SpaceMath scores (> 0)   │             │  • Gumbel-Sigmoid STE masks │
│  • Cluster->Class counts    │             │  • CE + L1(sigmoid(logits)) │
│  • modularity_score()       │             │  • Binary threshold (> 0)   │
│  • extract_weightmasks()    │             └──────────────┬──────────────┘
└──────────────┬──────────────┘                            │
               │ atlas_masks                               │ csordas_masks
               │ {c: {layer: [out] bool}}                  │ {c: {safe_layer: [out,in] float}}
               └─────────────────────┬─────────────────────┘
                                     ▼
                       ┌───────────────────────────┐
                       │ CompositionalityCheck     │
                       │  • Untainted baseline CM  │
                       │  • StaticMaskApplier      │
                       │  • Per-class masked CMs   │
                       │  • Recall, Leakage, Diff  │
                       └───────────────────────────┘
```

`CompositionalityPipeline` orchestrates all three stages end-to-end.

## Module Layout

| File | Contents |
|---|---|
| `positronixlab/tasks/compositionality/atlas_check.py` | `Distribution` ABC, `DiscreteDistribution`, `TensorDistribution`, `AtlasCheck` |
| `positronixlab/tasks/compositionality/weightmask_check.py` | `WeightmaskCheck`, `MultiClassWeightmaskManager`, `WeightMaskCriterion`, `WeightMasksMetrics` |
| `positronixlab/tasks/compositionality/compositionality.py` | `StaticMaskApplier`, `CompositionalityCheck`, `CompositionalityPipeline` |
| `positronixlab/tasks/examples/example_compositionality.py` | Self-contained synthetic CPU example |
| `positronixlab/tasks/compositionality/examples/example_atlas.py` | Legacy fixture-based example (`trained_model.pth`, `all_vars.pkl`) |

---

# API Reference

## `atlas_check.py`

### `Distribution` (ABC)
Abstract accumulator interface with `update(data)`, `get_distribution()`, and `add(other)`.

### `DiscreteDistribution(num_classes: int)`
Counts class occurrences in a float32 tensor of shape `(num_classes,)`.
- `update(class_labels, counts=None)`: accepts a scalar `int`/0D tensor or iterable of labels with optional weights.
- `get_distribution() -> Tensor`: returns normalized class probability vector (zeros if `total_count == 0`).
- `add(other: DiscreteDistribution)`: merges counts in-place (validates type and `num_classes`).

### `TensorDistribution(shape: torch.Size)`
Running sum/mean accumulator for arbitrary fixed-shape tensors.
- `update(y_hat: Tensor)`: adds `y_hat` and increments `total_count` by 1.
- `get_distribution() -> Tensor`: returns element-wise mean (zeros if `total_count == 0`).
- `add(other: TensorDistribution)`: merges sums and counts (validates type and `shape`).

### `AtlasCheck(config, model, dataprovider, clusters)`
Attributes active CUs (neurons with `SpaceMath.activations_to_scores(...) > 0.0`) per cluster to ground-truth class labels over `dataprovider.train`.

**Requirements**:
- `model` must expose a `.device` attribute (required by `DeeplearningBase`).
- `clusters` is `Dict[str, Tensor]` mapping each observable layer name (`observer.layer_names()`) to a 1D integer tensor of cluster IDs of length `out_features`, placed on `model.device`.

**Key methods**:
- `run()`: resets accumulators and runs one pass over `dataprovider.train` via `DeeplearningBase.process_epoch`.
- `extract_weightmasks(threshold: float = 0.0) -> Dict[int, Dict[str, Tensor]]`: for each class `c` and layer, selects all neurons belonging to clusters where `P(c | cluster) - P(c) > threshold`. Returns 1D `torch.bool` masks keyed by raw dotted layer name.
- `modularity_score(kld_target: str = 'y') -> Dict[str, Dict[str, float]]`: computes cluster-size-weighted per-layer and `'model'` averages of:
  - `modularity_purity`: `max_c P(c | cluster)`
  - `modularity_entropy`: `-sum_c P(c | cluster) log(P(c | cluster) + 1e-9)`
  - `modularity_kld`: `KL(P(. | cluster) || target)` where `target` is `'y'` (empirical class prior) or `'uniform'`
  - `modularity_kld_exp`: `1 - exp(-kld)`

---

## `weightmask_check.py`

### `WeightmaskCheck(model, dataprovider)`
Optimizes per-class differentiable binary weight masks (Csordás et al. style) using Gumbel-Sigmoid straight-through estimation.
- `run(target_layers: Dict[str, nn.Module], num_epochs: int = 5, lr: float = 1e-2) -> Dict[int, Dict[str, Tensor]]`: optimizes mask logits over `dataprovider.train`, thresholds `(param[c] > 0).float()`, removes hooks, and returns `{class_id: {safe_layer_name: binary_mask}}` where `safe_layer_name = name.replace('.', '_')`.

### `MultiClassWeightmaskManager(model, target_layers, num_classes, temp=1.0)`
`nn.Module` wrapper that freezes `model` parameters, stashes `module.weight` as `module._original_weight_param`, and installs forward pre-hooks applying `orig_weight * mask[current_class]`.
- `_get_computed_mask(mask_scores, training)`: Gumbel-Sigmoid STE in train mode; hard threshold `(mask_scores > 0).float()` in eval mode.
- `remove_hooks()`: removes pre-hooks and restores original `nn.Parameter` weights.

### `WeightMaskCriterion(lambda_l1: float = 1e-4)`
Computes `total_loss = CrossEntropyLoss(logits, targets) + lambda_l1 * mean_l(||sigmoid(m_l)||_1)`.
Returns `{"total_loss", "task_loss", "overall_weightmaskloss"}`.

### `WeightMasksMetrics`
Static utility methods for comparing binary masks and predictions:
- `compute_iou(mask_a, mask_b) -> float`
- `compute_iomin(mask_a, mask_b) -> float` (intersection over smaller mask size)
- `compute_sparsity(mask) -> float` *(Note: currently returns active-parameter fraction `active / total`, i.e. density)*
- `evaluate_network_modularity(masks_a, masks_b) -> {"network_iou", "network_iomin"}`
- `compute_accuracy(logits, targets) -> float`
- `compute_sequence_accuracy(predictions, targets, pad_token_id=0) -> float`

---

## `compositionality.py`

### `StaticMaskApplier(model, masks)`
Context manager that applies static masks `{safe_name: tensor}` via forward pre/post hooks and guarantees restoration of original `nn.Parameter` weights on exit (including on exception).

### `CompositionalityCheck(model, dataloader, num_classes, device=None)`
Evaluates masked subnetworks via `(C, C)` confusion matrices:
- `compute_untainted_cm() -> Tensor`
- `evaluate_masked_runs(masks_by_class, method_name="Atlas") -> Dict[int, Dict[str, Any]]`: returns per-class `confusion_matrix`, `target_recall`, `off_target_leakage`, and `total_accuracy`.
- `run(atlas_masks, csordas_masks) -> Dict[str, Any]`: runs untainted, Atlas, and Csordás evaluations and computes per-class row-normalized Frobenius confusion-matrix distance (`cm_frobenius_diff`).

### `CompositionalityPipeline(model, dataprovider, config)`
High-level orchestrator:
- `run_full_evaluation(clusters, atlas_threshold=0.0, csordas_epochs=5, csordas_lr=1e-2) -> Dict[str, Any]`

---

## Known Caveats & Current Limitations

See `positronixlab/tasks/analysis.md` for the full defect list. Key operational caveats:
1. **Mask shape & key mismatch**: `AtlasCheck.extract_weightmasks()` returns dotted keys and 1D `(out_features,)` bool masks, while `StaticMaskApplier` expects underscore-sanitized keys (`name.replace('.', '_')`) and weight-broadcastable shapes (e.g. `(out_features, 1)` or `(out_features, in_features)`).
2. **Hook cleanup**: `AtlasCheck` leaves `ModularObserver` forward hooks attached after `run()`. Call `atlas.observer.on_forward_layer(None)` before reusing the model.
3. **`WeightmaskCheck` limitations**: `num_epochs` is currently ignored (1 pass only), empty class slices inside a batch produce `NaN` gradients, and `remove_hooks()` does not restore `requires_grad=True` on model parameters.
4. **Test split requirement**: `CompositionalityPipeline` passes `dataprovider.test` to `CompositionalityCheck`; ensure `dataprovider.test` is populated (not `None`).
