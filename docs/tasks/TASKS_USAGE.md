# Tasks Usage Guide

Practical recipes for discovering subnetworks (`AtlasDiscoverer`, `RandomMaskDiscoverer`), applying and manipulating `SubnetworkMask`s (`MaskApplier`), and running multi-method comparisons (`ModularityBenchmark`), plus the earlier `compositionality` prototype.

---

## 1. Discovering Subnetworks with `AtlasDiscoverer`

```python
from positronixlab.baselines import ModelLoader, DataProviderLoader
from positronixlab.tasks import AtlasDiscoverer, MaskApplier, default_eval_accuracy

dp = DataProviderLoader.create("csordas:add_mul", batch_size=32, num_workers=0)
model = ModelLoader.create(
    "csordas:mlp",
    n_inputs=dp.num_features,
    n_outputs=dp.num_classes,
    layer_sizes=[64, 64],
)
target_layers = model.maskable_layers()

atlas = AtlasDiscoverer(atlas_epochs=3, space_dim=8, top_cluster_fraction=0.5)

# Discover masks for every subtask using one shared atlas fit
subtask_loaders = {name: dp.get_subtask(name).train_loader() for name in dp.list_subtasks()}
atlas_masks = atlas.discover_subtasks(
    model,
    reference_dataloader=dp.train,
    subtask_dataloaders=subtask_loaders,
    target_layers=target_layers,
)

# Evaluate with task-aware accuracy (supports both integer and multi-digit modes)
from positronixlab.tasks import make_accuracy_fn, resolve_subtask_criterion

acc_fn = make_accuracy_fn(dp, subtask="add")
criterion = resolve_subtask_criterion(dp, subtask="add")
with MaskApplier(model, add_mask, mode="retain"):
    retained_acc = acc_fn(model, dp.get_subtask("add").test_loader())

with MaskApplier(model, add_mask, mode="zero_ablate"):
    ablated_acc = default_eval_accuracy(model, dp.get_subtask("add").test_loader())
```

### Decoupled Attribution & Sparsity Sweeps

Fit the atlas once, then sweep `top_cluster_fraction` cheaply without re-fitting:

```python
positions, clusters = atlas.fit_atlas(model, dp.train, target_layers=target_layers)
masks = atlas.attribute_from_clusters(model, clusters, subtask_loaders, top_cluster_fraction=0.4)
```

---

## 2. Manipulating & Comparing `SubnetworkMask`s

```python
from positronixlab.tasks import SubnetworkMask

m_add = atlas_masks["add"]
m_mul = atlas_masks["mul"]

# Set algebra
shared = m_add & m_mul
union = m_add | m_mul
add_only = m_add - m_mul
complement = m_add.invert()

# Projection & metrics
weight_mask = m_add.to_weight_mask(model)
print("Sparsity:", m_add.sparsity(), "IoU:", m_add.iou(m_mul), "IoMin:", m_add.iomin(m_mul))

# Persistence
m_add.save("/tmp/add_mask.pt")
reloaded = SubnetworkMask.load("/tmp/add_mask.pt")
```

---

## 3. Multi-Method Benchmarking (`ModularityBenchmark`)

Compare `AtlasDiscoverer` against `CsordasGumbelDiscoverer`, `NeuroSurgeonDiscoverer`, and `RandomMaskDiscoverer` on the exact same model and subtasks:

```python
from positronixlab.baselines import CsordasGumbelDiscoverer, NeuroSurgeonDiscoverer
from positronixlab.tasks import ModularityBenchmark, RandomMaskDiscoverer

csordas = CsordasGumbelDiscoverer(epochs=5)
rand = RandomMaskDiscoverer(density=0.5, seed=42)

masks_by_method = {
    "atlas": atlas_masks,
    "csordas": {
        name: csordas.discover(model, loader, target_layers=target_layers, mask_name=f"csordas_{name}")
        for name, loader in subtask_loaders.items()
    },
    "random": {
        name: rand.discover(model, loader, target_layers=target_layers, mask_name=f"rand_{name}")
        for name, loader in subtask_loaders.items()
    },
}

test_loaders = {name: dp.get_subtask(name).test_loader() for name in dp.list_subtasks()}
bench = ModularityBenchmark(modes=("retain", "zero_ablate"), seed=42)
report = bench.compare_methods(
    model,
    masks_by_method,
    test_loaders,
    dataset_name="csordas:add_mul",
    model_arch="csordas:mlp",
    reference_method="atlas",
)

print(report.summary_metrics())
report.save_json("/tmp/benchmark_report.json")
report.save_csv("/tmp/benchmark_report.csv")
```

---

## 4. Earlier Compositionality Prototype (`tasks.compositionality`)

For confusion-matrix analysis with pre-computed cluster assignments:

```python
from positronixlab.pipeline import PipelineConfigV1
from positronixlab.tasks.compositionality.atlas_check import AtlasCheck

atlas = AtlasCheck(config=PipelineConfigV1(), model=model, dataprovider=dp, clusters=clusters)
atlas.run()
scores = atlas.modularity_score(kld_target="y")
raw_masks = atlas.extract_weightmasks(threshold=0.05)
atlas.observer.on_forward_layer(None)  # detach observer hooks
```

---

## 5. Running Examples & Tests

```bash
# Head-to-head Atlas vs SOTA CPU benchmark
.venv/bin/python examples/tasks/example_atlas_vs_sota.py

# ModularityBenchmark comparison example
.venv/bin/python examples/tasks/example_sota_benchmark.py

# Earlier compositionality prototype example
.venv/bin/python examples/tasks/example_compositionality.py

# Run all tasks unit tests
TRITON_CACHE_DIR=/tmp/triton_cache TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache \
  .venv/bin/pytest tests/tasks -v
```
