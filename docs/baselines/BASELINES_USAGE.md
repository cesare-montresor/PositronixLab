# PositronixLab Baselines Usage Guide

Practical recipes for loading SOTA models and compositional datasets, saving/reloading bundles, and running Csordás and NeuroSurgeon subnetwork discoverers via `positronixlab.baselines`.

---

## 1. Creating Models and Compositional Datasets

```python
from positronixlab.baselines import ModelLoader, DataProviderLoader

print("Models:", ModelLoader.list_models())
print("Datasets:", DataProviderLoader.list_datasets())

# Create a Csordás compositional dataset and MLP
dp = DataProviderLoader.create("csordas:add_mul", batch_size=32, num_workers=0)
model = ModelLoader.create(
    "csordas:mlp",
    n_inputs=dp.num_features,
    n_outputs=dp.num_classes,
    layer_sizes=[64, 64],
)
print("Maskable layers:", model.maskable_layers())
```

For Lepori CVR or SynEval benchmarks:

```python
cvr_dp = DataProviderLoader.create("lepori:cvr_inside_contact", batch_size=16, num_workers=0)
cvr_model = ModelLoader.create("lepori:cvr_resnet18", pretrained=False)
```

---

## 2. Saving & Reloading Self-Describing Bundles

`ModelLoader.save()` writes a `positronix_sota_bundle_v1` checkpoint containing CPU weights, constructor kwargs, dataset name, metrics, and metadata so the model can be reconstructed with a single call:

```python
from positronixlab.baselines import ModelLoader

ModelLoader.save(
    model,
    "/tmp/add_mul_mlp.pt",
    model_kwargs={"n_inputs": dp.num_features, "n_outputs": dp.num_classes, "layer_sizes": [64, 64]},
    dataset_name="csordas:add_mul",
    metrics={"val_acc": 0.96},
)

reloaded_model, info = ModelLoader.load_bundle("/tmp/add_mul_mlp.pt", device="cpu")
print(info["arch"], info["metrics"])
```

### Loading External Checkpoints (With Mask Stripping)

`ModelLoader.load()` auto-detects `"neurosurgeon"`, `"lepori"`, `"csordas"`, and `"plain"` checkpoints and strips mask keys automatically. Pass `return_mask=True` to also extract embedded masks as a `SubnetworkMask`:

```python
model, extracted_mask = ModelLoader.load(
    "csordas:mlp",
    checkpoint="/path/to/checkpoint.pt",
    n_inputs=dp.num_features,
    n_outputs=dp.num_classes,
    return_mask=True,
)
```

---

## 3. Discovering Subnetworks with SOTA Baselines

Every discoverer implements `SubnetworkDiscoverer.discover(...) -> SubnetworkMask` without mutating the caller's model:

```python
from positronixlab.baselines import CsordasGumbelDiscoverer, NeuroSurgeonDiscoverer

target_layers = model.maskable_layers()
add_loader = dp.get_subtask("add").train_loader()

# 1. Csordás Gumbel-Sigmoid STE (weight-level)
csordas = CsordasGumbelDiscoverer(mask_loss_weight=1e-4, lr=1e-2, epochs=5)
csordas_mask = csordas.discover(model, add_loader, target_layers=target_layers, mask_name="csordas_add")

# 2. NeuroSurgeon Continuous Sparsification (weight- or neuron-level)
ns = NeuroSurgeonDiscoverer(
    mask_method="continuous_sparsification",  # or "hard_concrete", "magnitude_pruning"
    mask_unit="weight",                       # or "neuron"
    l0_lambda=1e-4,
    lr=1e-2,
    epochs=5,
)
ns_mask = ns.discover(model, add_loader, target_layers=target_layers, mask_name="ns_add")

print("Csordás sparsity:", csordas_mask.sparsity(), "NS sparsity:", ns_mask.sparsity())
print("Cross-method IoU:", csordas_mask.iou(ns_mask))
```

### Direct NeuroSurgeon Wrapping

For custom training loops over `CircuitModel`, use `wrap_with_neurosurgeon()` (or `model.to_neurosurgeon()`):

```python
from positronixlab.baselines import wrap_with_neurosurgeon, is_neurosurgeon_available

if is_neurosurgeon_available():
    wrapped = wrap_with_neurosurgeon(
        model,
        mask_method="continuous_sparsification",
        mask_unit="weight",
        l0_lambda=1e-4,
        deepcopy_model=True,
    )
    # Optimize wrapped.mask_parameters() using task_loss + l0_lambda * wrapped.l0_loss()
    mask = wrapped.extract_subnetwork_mask(name="custom_ns")
```

---

## 4. Running Tests & Head-to-Head Benchmarks

```bash
# Baselines & unified discoverer unit tests
TRITON_CACHE_DIR=/tmp/triton_cache TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache \
  .venv/bin/pytest tests/baselines tests/tasks/test_masks_and_discoverers.py tests/tasks/test_atlas_discoverer.py -v

# Head-to-head Atlas vs Csordás vs NeuroSurgeon CPU benchmark
.venv/bin/python examples/tasks/example_atlas_vs_sota.py
```
