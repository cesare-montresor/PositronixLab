# Tasks Usage Guide

How to run downstream modularity and compositionality evaluations on a trained
PyTorch model using `positronixlab.tasks.compositionality`.

## Prerequisites

- A trained PyTorch `nn.Module` (must expose `.device` if used with `AtlasCheck`;
  `MetaModel` provides this automatically).
- A `DataProvider` (or compatible object) exposing `.num_classes`, `.train`, and
  `.test` (or `.val`) `DataLoader`s yielding `(x, y)` or `(x, y, idx)`.
- For `AtlasCheck`: a `PipelineConfig` and per-layer 1D cluster assignment
  tensors `clusters: Dict[str, torch.Tensor]` (one integer cluster ID per output
  neuron of each observable `nn.Linear` layer, placed on `model.device`).

---

## 1. Evaluating Cluster Modularity (`AtlasCheck`)

```python
import torch
from positronixlab.pipeline import PipelineConfigV1
from positronixlab.tasks.compositionality.atlas_check import AtlasCheck

config = PipelineConfigV1()
atlas = AtlasCheck(
    config=config,
    model=model,
    dataprovider=dp,
    clusters=clusters,  # {layer_name: LongTensor[out_features]} on model.device
)
atlas.run()

# Per-layer and model-wide purity, entropy, KL divergence
scores = atlas.modularity_score(kld_target="y")  # or "uniform"
print(scores["model"])

# Extract per-class neuron masks (excess probability > threshold)
raw_atlas_masks = atlas.extract_weightmasks(threshold=0.05)

# IMPORTANT: Detach observer forward hooks before running other tasks on `model`
atlas.observer.on_forward_layer(None)
```

### Adapting Atlas Masks for `StaticMaskApplier`

`AtlasCheck.extract_weightmasks()` returns 1D neuron masks keyed by dotted module
names (e.g. `"net.0"`), whereas `StaticMaskApplier` expects underscore-sanitized
keys (`"net_0"`) and tensors broadcastable to `module.weight.shape`
(`(out_features, in_features)`):

```python
def adapt_atlas_masks_for_applier(raw_masks):
    adapted = {}
    for cls_id, layer_masks in raw_masks.items():
        adapted[cls_id] = {
            layer_name.replace(".", "_"): mask.float().unsqueeze(1)
            for layer_name, mask in layer_masks.items()
        }
    return adapted

atlas_masks = adapt_atlas_masks_for_applier(raw_atlas_masks)
```

---

## 2. Optimizing Csordás Weight Masks (`WeightmaskCheck`)

```python
from positronixlab.tasks.compositionality.weightmask_check import (
    WeightmaskCheck,
    WeightMasksMetrics,
)

# Select target layers (e.g. Linear layers with a `weight` attribute)
target_layers = {
    name: mod for name, mod in model.named_modules()
    if isinstance(mod, torch.nn.Linear)
}

# Save requires_grad state (WeightmaskCheck currently leaves params frozen)
orig_requires_grad = {p: p.requires_grad for p in model.parameters()}

# Ensure every batch contains at least one sample per class to avoid NaN loss
csordas = WeightmaskCheck(model=model, dataprovider=dp)
csordas_masks = csordas.run(target_layers=target_layers, num_epochs=5, lr=1e-2)

# Restore requires_grad
for p, req in orig_requires_grad.items():
    p.requires_grad = req

# Inspect mask overlap between two classes
overlap = WeightMasksMetrics.evaluate_network_modularity(
    csordas_masks[0], csordas_masks[1]
)
print("Class 0 vs Class 1 overlap:", overlap)
```

---

## 3. Running Compositionality & Interference Checks (`CompositionalityCheck`)

```python
from positronixlab.tasks.compositionality.compositionality import (
    CompositionalityCheck,
    StaticMaskApplier,
)

checker = CompositionalityCheck(
    model=model,
    dataloader=dp.test,
    num_classes=dp.num_classes,
)

results = checker.run(
    atlas_masks=atlas_masks,
    csordas_masks=csordas_masks,
)

print(f"Untainted accuracy: {results['untainted']['accuracy']:.4f}")
for c, comp in results["comparison"].items():
    print(
        f"Class {c}: Atlas recall={comp['atlas_target_recall']:.3f}, "
        f"Csordas recall={comp['csordas_target_recall']:.3f}, "
        f"CM Fro diff={comp['cm_frobenius_diff']:.3f}"
    )
```

### Applying a Single Mask Manually

```python
with StaticMaskApplier(model, atlas_masks[0]):
    logits = model(x_batch)
# Original nn.Parameter weights are automatically restored on exit
```

---

## 4. End-to-End Example (CPU, No Fixtures Required)

Run the synthetic CPU example from the repository root:

```bash
.venv/bin/python positronixlab/tasks/examples/example_compositionality.py
```

And run the unit test suite:

```bash
.venv/bin/pytest positronixlab/tasks/tests -v
```
