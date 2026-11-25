# ModularSpace System Architecture

`positronixlab.modularspace` builds a decoupled vector-space "brain atlas" for mapping emergent functional modularity in neural networks. Each observed layer gets an independent embedding space where every computation unit (CU / neuron / conv kernel) has a learnable coordinate vector updated via contrastive learning on forward activations.

## Core Invariant: Decoupled Space Graph

`LayerSpace` and `ModularSpace` **never** backpropagate into the observed neural network:
1. Raw forward activations are converted to detached `[-1, 1]` unit scores (active `>= 0`, inactive `< 0`).
2. Each layer owns its own `positions: Tensor[num_cus, modular_space_size]` (`requires_grad=True`), its own `torch.optim.Adam` optimizer, and an optional running `stats: Tensor[num_cus, 4]` buffer (`[count_on, count_off, sum_score_on, sum_score_off]`).
3. Contrastive + regularization losses optimize only `positions`.

## Package Layout

```text
positronixlab/modularspace/
├── __init__.py            # Public exports (ModularSpace & ModularOverSpace lazy-loaded)
├── config.py              # LayerSpaceConfig (BaseConfig subclass, Ax-friendly flat fields)
├── presets.py             # Named presets (v1_centers, distance_sigreg_centers, distance_sigreg_positions, softmax_ce)
├── space_step.py          # SpaceStepSpec & stateless SpaceStep runner over spacemath primitives
├── layerspace.py          # Current per-layer CU embedding space (LayerSpace)
├── modularspace.py        # Legacy V1 per-layer space (ModularSpace)
├── modularoverspace.py    # Per-layer wrapper around ClusterTracker (ModularOverSpace)
└── custertracker.py       # Cluster identity tracker (ClusterTracker / ClusterIdentityTracker)

# Companion top-level directories:
# docs/modularspace/       # System/usage docs + design notes (layerloss.md, cluster_identity_tracking.md)
# tests/modularspace/      # Unit tests for LayerSpace, SpaceStep, config/presets, ModularSpace, ClusterTracker
# examples/modularspace/   # Runnable CPU examples & Dash demos
```

## Components

### 1. `LayerSpaceConfig` (`config.py`)
A framework-independent `BaseConfig` dataclass containing flat, Ax-sweepable hyperparameters:
- **Geometry & optimization:** `modular_space_size`, `modular_space_lr`, `modular_space_active_q`, `modular_space_distance` (`"cosine"` | `"euclidean"`), `modular_overspace_size`, `modular_modularize_decay`, `space_epochs`.
- **Stage selectors:** `modular_space_scoring`, `modular_space_reducer`, `modular_space_apply_activation`, `modular_space_weighting`, `modular_space_contrast`.
- **Conditional loss weights (`0.0` skips the term):** `modular_space_loss_global`, `modular_space_loss_on`, `modular_space_loss_off`, `modular_space_loss_cluster`, `modular_space_loss_centers`, `modular_space_loss_batch`, `modular_space_loss_sigreg`.
- **Method-specific knobs:** `modular_space_tau`, `modular_space_freq_alpha`, `modular_space_sigreg_method` (`"weak"` | `"strong"`), `modular_space_sigreg_target` (`"centers"` | `"positions"`), `modular_space_sigreg_dim`, `modular_cnn_locality`.

### 2. `PRESETS` (`presets.py`)
Named configurations reproducing the four main space formulations:
- `v1_centers`: Cosine `distance_log` global contrast + `loss_centers` repulsion (reproduces `ModularSpace` v1).
- `distance_sigreg_centers`: Global + cluster `distance_log` + weak SIGReg on active centers.
- `distance_sigreg_positions`: Global + cluster `distance_log` + weak SIGReg directly on CU positions.
- `softmax_ce`: Inverse-frequency weighted `softmax_ce` (global + cluster) + weak SIGReg on positions.

### 3. `SpaceStep` & `SpaceStepSpec` (`space_step.py`)
`SpaceStep` is a stateless, composable pipeline over generic unit tensors `positions: (U, D)` and `scores: (bs, U)` wired to `positronixlab.spacemath`:
1. **Scoring (`spacemath.scoring.activations_to_scores`):** Reduces architecture-specific activations (e.g., `linear`, `conv_l2`) and maps them to `[-1, 1]` scores (`quantile`, `top_k`, `sign_preserving_row`, `sign_preserving_batch`, `softmax`).
2. **Weighting (`spacemath.weighting`):** Produces `(w_on, w_off)` via `raw`, `inv_freq` (using running `stats`), or `uniform`.
3. **Global contrast (`loss_global > 0`):** Runs registered contrast (`distance_log`, `distance_linear`, `softmax_ce`) over all active unit slots (`mask_global_scope`).
4. **Cluster contrast (`loss_cluster > 0`):** When `clusters_labels` has `K >= 2` clusters, runs contrast within each cluster (`mask_cluster_membership`), splitting mixed clusters.
5. **Auxiliary losses:**
   - `loss_centers > 0`: Per-sample repulsion between `global_res.centers_on` and `global_res.centers_off`.
   - `loss_batch > 0`: Inter-sample repulsion between active centers (`primary.centers_on`).
   - `loss_sigreg > 0`: Isotropic Gaussian regularization (`sigreg_weak` / `sigreg_strong`) on either `positions` or `primary.centers_on`.

`SpaceStepSpec.from_dict()` and `SpaceStepSpec.from_config()` translate both short field names and flat `modular_space_*` config keys.

### 4. `LayerSpace` (`layerspace.py`)
Manages per-layer `positions`, `optimizers`, and optional `stats` tensors (marked with `torch._dynamo.mark_static_address` for `torch.compile` compatibility in `spacemath.contrast`).
- Resolves parameters from `LayerSpaceConfig` (or compatible config) with optional keyword overrides in `__init__`.
- `get_space(layer_id)` returns the per-layer state bundle (`positions`, `optimizer`, `stats`, metadata).
- `score_activations(layer_id, activations)` and `compute_loss(positions, scores, *, stats=None, clusters_labels=None, unit_mask=None)` expose the decoupled scoring and loss stages directly.
- `process_activations(layer_id, activations, clusters_labels=None, unit_mask=None)` scores activations and calls `process_scores(...)`.
- `process_scores(...)` updates `stats` in-place (if enabled), computes the composite loss, steps Adam, and returns `(losses_dict, positions.detach(), stats_copy)`.

### 5. `ModularSpace` (`modularspace.py`)
Legacy V1 implementation using `SpaceMath` and `@staticmethod_compile`. Kept as a lazy export in `__init__.py` for backwards compatibility with `PipelineConfigV1` workflows.

### 6. `ClusterTracker` & `ModularOverSpace` (`custertracker.py`, `modularoverspace.py`)
Tracks cluster identities across independent re-clusterings of `N` fixed items and maintains temporally stable outer-space coordinates `O_t: [K_t, d_out]`:
- Computes the `[K_new, K_old]` overlap matrix via vectorized one-hot matmul (`O(N * K_old * K_new)`).
- Classifies transitions: `init`, `match`, `split`, `merge`, `switch`, `add`, `remove` (see [cluster_identity_tracking.md](cluster_identity_tracking.md) for the mathematical specification).
- `ModularOverSpace` owns one `ClusterTracker` per `layer_id` and records states, outer positions, events, and optional event-count stats.
