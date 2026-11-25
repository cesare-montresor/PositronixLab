# SpaceMath Usage Guide

This guide shows how to use `positronixlab.spacemath` primitives to score neural activations, compute scoped contrastive updates, and apply geometric regularizers.

A complete runnable script is available at [example_spacemath.py](../../examples/spacemath/example_spacemath.py).

---

## 1. Quick Start: Single Layer Space Step

```python
import torch
from positronixlab.spacemath import (
    SpaceType,
    activations_to_scores,
    get_contrast,
    get_weighter,
    loss_centers_repulsion,
    loss_sigreg_centers,
    mask_global_scope,
    stats_update_,
)

bs, num_units, emb_dim = 32, 64, 8
positions = torch.randn(num_units, emb_dim, requires_grad=True)
stats = torch.zeros(num_units, 4)

# 1. Score detached forward activations -> [-1, 1]
activations = torch.relu(torch.randn(bs, num_units))
scores = activations_to_scores(
    activations,
    layer_type="linear",
    score_method="top_k",
    active_q=0.8,
)

# 2. Update running per-unit stats and compute inverse-frequency weights
stats_update_(stats, scores)
w_on, w_off = get_weighter("inv_freq")(scores, stats=stats, freq_alpha=0.5)

# 3. Build a global (1, U) scope mask
scope_mask = mask_global_scope(num_units, positions.device, positions.dtype)

# 4. Run contrastive step
res = get_contrast("softmax_ce")(
    positions,
    w_on,
    w_off,
    scope_mask,
    space_type=SpaceType.COSINE,
    tau=0.1,
)

# 5. Add optional auxiliary regularizers (kept outside torch.compile)
loss = (
    res.loss
    + 0.1 * loss_centers_repulsion(res.centers_on, res.centers_off, res.mixed, space_type=SpaceType.COSINE)
    + 0.1 * loss_sigreg_centers(res.centers_on, res.mixed, space_type=SpaceType.COSINE, method="weak")
)
loss.backward()
```

---

## 2. Scoring Convolutional & Sequence Activations

Pass 4D `(bs, C, H, W)` feature maps with `layer_type="conv"` and a spatial reducer (`"conv_locality"`, `"conv_l2"`, `"conv_mean_abs"`, or `"conv_max"`), or 3D `(bs, T, U)` sequence activations with `layer_type="transformer"`/`"lstm"`/`"rnn"` and a sequence reducer (`"seq_mean"`, `"seq_max"`, `"seq_cls"`, or `"seq_energy"`):

```python
from torch import nn
from positronixlab.spacemath import activations_to_scores

fmap = torch.randn(16, 32, 7, 7)
scores = activations_to_scores(
    fmap,
    layer_type="conv",
    reducer="conv_l2",
    score_method="quantile",
    activation_fn=nn.ReLU(),
    apply_activation=True,
    active_q=0.75,
)
# scores.shape == (16, 32)

seq = torch.randn(16, 12, 64)
seq_scores = activations_to_scores(seq, layer_type="transformer", reducer="seq_mean", active_q=0.75)
# seq_scores.shape == (16, 64)
```

---

## 3. Cluster-Scoped Contrast

To refine unit positions *within* existing clusters (e.g., from FAISS K-Means or HDBSCAN), build a `(K, U)` membership mask. Negative labels (e.g., `-1` for noise) and optional `unit_mask` slots are automatically excluded:

```python
from positronixlab.spacemath import contrast_distance_log, mask_cluster_membership

labels = torch.tensor([0, 0, 1, 1, 2, 2, -1, 0])  # -1 is excluded
cluster_mask = mask_cluster_membership(
    labels,
    num_units=len(labels),
    device=positions.device,
    dtype=positions.dtype,
)

if cluster_mask is not None:
    res_cluster = contrast_distance_log(
        positions[:8],
        w_on[:, :8],
        w_off[:, :8],
        cluster_mask,
        space_type=SpaceType.EUCLIDEAN,
        weight_on=1.0,
        weight_off=0.5,
    )
```

---

## 4. Direct Geometry, Losses & SIGReg

```python
from positronixlab.spacemath import (
    SpaceType,
    distance_elementwise,
    distance_pairwise,
    loss_batch_repulsion,
    loss_layer_alignment,
    loss_sigreg,
    space_normalize,
)

x = torch.randn(16, 8)
y = torch.randn(16, 8)

d_el = distance_elementwise(x, y, space_type=SpaceType.COSINE)  # (16,)
d_pw = distance_pairwise(x, y, space_type=SpaceType.EUCLIDEAN)  # (16, 16)

# Isotropic Gaussian regularization (auto-scales unit-sphere vectors by sqrt(D))
l_weak = loss_sigreg(x, space_type=SpaceType.COSINE, method="weak", sketch_dim=64)
l_strong = loss_sigreg(x, space_type=SpaceType.EUCLIDEAN, method="strong", sketch_dim=32)

# Cross-layer alignment across L layers of (bs, D) active centers
l_align = loss_layer_alignment([x, y], space_type=SpaceType.COSINE)
```

---

## 5. Registry Discovery

All v2 primitive families expose a dictionary registry and a case-insensitive lookup helper (except `LOSSES`, which is accessed directly by key):

```python
from positronixlab.spacemath import (
    REDUCERS, get_reducer,
    SCORERS, get_scorer,
    WEIGHTERS, get_weighter,
    CONTRASTS, get_contrast,
    LOSSES,
)

print("Reducers :", sorted(REDUCERS))
print("Scorers  :", sorted(SCORERS))
print("Weighters:", sorted(WEIGHTERS))
print("Contrasts:", sorted(CONTRASTS))
print("Losses   :", sorted(LOSSES))
```

---

## 6. Environment Tip for `torch.compile`

Because `contrast_*` functions compile on first invocation, set writable cache directories if running in a sandboxed or read-only-home environment:

```bash
TRITON_CACHE_DIR=/tmp/triton_cache TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache \
  .venv/bin/python examples/spacemath/example_spacemath.py
```
