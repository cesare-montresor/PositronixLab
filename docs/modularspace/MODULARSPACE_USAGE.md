# ModularSpace Usage Guide

## 1. Quickstart: `LayerSpace` with Presets

```python
import torch
from positronixlab.modularspace import LayerSpace, LayerSpaceConfig, get_preset, list_presets

print("Available presets:", list_presets())

cfg = LayerSpaceConfig.from_dict(
    get_preset("softmax_ce", modular_space_size=8, modular_space_lr=0.02)
)

observed_layers = {
    "fc1": {"out_units": 32, "layer_type": "linear", "activation_type": "relu"},
}

space = LayerSpace(
    cfg,
    observed_layers,
    device=torch.device("cpu"),
    stats_enabled=True,  # recommended for inv_freq weighting (used by softmax_ce)
)

activations = torch.randn(16, 32)
cluster_labels = torch.arange(32) % 4  # optional cluster feedback from t-1

losses, positions, stats = space.process_activations(
    "fc1", activations, clusters_labels=cluster_labels
)
print("Total loss:", float(losses["loss"]), "Positions:", positions.shape)
```

## 2. Keyword Overrides & Clean Ablations

Any stage selector or loss weight can be overridden directly in `LayerSpace(...)`. Setting any loss weight to `0.0` skips that term completely (it is neither computed nor added to the returned `losses` dict):

```python
space = LayerSpace(
    LayerSpaceConfig(),
    observed_layers,
    device=torch.device("cpu"),
    score_method="top_k",
    weight_method="uniform",
    contrast_method="distance_linear",
    loss_global=1.0,
    loss_cluster=0.0,
    loss_sigreg=0.0,
    loss_centers=0.5,
    loss_batch=0.25,
)
```

> **Note:** At least one enabled loss term must produce a gradient-bearing scalar (for example, `loss_global > 0.0` or `loss_sigreg > 0.0` with `sigreg_target="positions"`). Calling `process_scores()` with all effective loss terms disabled raises a `RuntimeError` during `loss.backward()`.

## 3. Convolutional Layers & Dynamic `unit_mask`

For 4D conv feature maps `(B, C, H, W)`, set `layer_type="conv"` and choose a conv reducer (e.g. `"conv_l2"`):

```python
cfg = LayerSpaceConfig(modular_space_scoring="top_k", modular_space_reducer="conv_l2")
space = LayerSpace(
    cfg,
    {"conv1": {"out_units": 16, "layer_type": "conv", "activation_type": "relu"}},
    device=torch.device("cpu"),
)
unit_mask = torch.zeros(16)
unit_mask[:12] = 1.0  # mask out inactive unit slots
losses, positions, _ = space.process_activations("conv1", torch.randn(8, 16, 5, 5), unit_mask=unit_mask)
```

## 4. Using `SpaceStep` Standalone

`SpaceStep` is stateless and can be reused on custom unit tensors (e.g., an overspace of clusters):

```python
from positronixlab.modularspace import SpaceStep, get_preset_spec

spec = get_preset_spec("distance_sigreg_positions", active_q=0.75)
step = SpaceStep(spec)

positions = torch.randn(20, 6, requires_grad=True)
scores = step.score_activations(torch.randn(16, 20), layer_type="linear")
loss, metrics = step.compute_loss(positions, scores)
loss.backward()
```

## 5. Tracking Cluster Identities (`ClusterTracker` & `ModularOverSpace`)

```python
import torch
from positronixlab.modularspace import ClusterTracker, LayerSpaceConfig, ModularOverSpace

# Single-layer tracker
tracker = ClusterTracker(d_out=2, seed=7)
labels_t0 = torch.tensor([0, 0, 1, 1])
centroids_t0 = torch.tensor([[0.0, 0.0], [3.0, 3.0]])
state0, events0 = tracker.step(labels_t0, centroids_t0)

# Multi-layer manager
overspace = ModularOverSpace(
    config=LayerSpaceConfig(modular_overspace_size=4),
    observed_layers={},
    device=torch.device("cpu"),
    stats_enabled=True,
)
state, events = overspace.update("fc1", labels_t0, centroids_t0)
outer_coords = overspace.get_positions("fc1")  # shape: (K, 4)
```

## 6. Legacy `ModularSpace` (V1)

`ModularSpace` is lazy-loaded from `positronixlab.modularspace`. Until the `get_space()` `None`-check bug is fixed, pass `stats_enabled=True` and a `PipelineConfigV1`-compatible config:

```python
from positronixlab.modularspace import ModularSpace
from positronixlab.pipeline import PipelineConfigV1

space = ModularSpace(
    PipelineConfigV1(),
    {"fc1": {"out_units": 16, "layer_type": "linear", "activation_type": "relu"}},
    device=torch.device("cpu"),
    stats_enabled=True,
)
losses, positions, stats = space.process_activations("fc1", torch.randn(8, 16))
```

## 7. Running Tests & Examples on CPU

`spacemath.contrast` uses `torch.compile`. In environments with a read-only home directory, set writable cache directories:

```bash
TRITON_CACHE_DIR=/tmp/triton_cache TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache \
  .venv/bin/pytest tests/modularspace -v

TRITON_CACHE_DIR=/tmp/triton_cache TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache \
  .venv/bin/python examples/modularspace/example_layerspace.py
```
