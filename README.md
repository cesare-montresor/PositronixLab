# PositronixLab

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MPL 2.0](https://img.shields.io/badge/License-MPL_2.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)

**PositronixLab** is a PyTorch research framework for mapping, isolating, and evaluating emergent functional modularity in neural networks.

It builds a decoupled, vector-space **"brain atlas"** of a model's computation units (neurons, channels, or attention heads): forward activations are observed without altering the base model's gradient graph, projected into a contrastive embedding space per layer, clustered into functional computational groups (CGs), tracked across training, and benchmarked head-to-head against weight-masking SOTA baselines.

---

## Key Features

* **Decoupled Modular Space (`modularspace`, `spacemath`)** — Maps detached layer activations to `[-1, 1]` functional scores and optimizes per-layer unit embeddings (`[num_cus, space_dim]`) via composable contrastive objectives (`SpaceStep`, presets, `torch.compile`-accelerated primitives) without backpropagating into the base model.
* **Non-Invasive Model Observation (`modularobserver`)** — Static and dynamic (`TorchFunctionMode`) computation-graph tracing (`ModelMapper`) with configurable hook registration (`ModularObserver`, `LayerObserver`) and Cytoscape graph export.
* **Adaptive Clustering & Identity Tracking (`modularcluster`)** — FAISS-backed adaptive K-Means, streaming clustering, and persistent cluster identity tracking (`ClusterIdentityTracker`, `StreamingClusterTracker`) across training epochs.
* **Unified Subnetwork Evaluation Contract (`tasks`)** — Canonical `SubnetworkMask`, non-destructive `MaskApplier` (zero, mean, resample, retain, and inverse ablation modes), `AtlasDiscoverer`, and `ModularityBenchmark` for apples-to-apples comparisons.
* **SOTA Baselines Included (`baselines`)** — Unified `ModelLoader`, `DataProviderLoader`, and discoverers for **Csordás et al.** (Gumbel-Sigmoid weight masking, Add/Mul, Tuple Arithmetic, Permuted MNIST), **Lepori et al.** (CVR / SynEval odd-one-out), and **NeuroSurgeon** (continuous sparsification, hard concrete, magnitude pruning).
* **Multi-Objective NAS & Live Visualization (`nas`, `modularui`, `stats`)** — Ax 1.3+ multi-objective experiment orchestration, streaming UMAP/EMA reducers, and a thread-safe Dash/Plotly/Cytoscape dashboard.

---

## Installation

Requires **Python 3.12+**.

```bash
pip install positronixlab
```

### Optional Extras

PositronixLab keeps its core lightweight and exposes optional capabilities as extras:

| Extra | Command | Includes |
| :--- | :--- | :--- |
| **UI Dashboard** | `pip install "positronixlab[ui]"` | Dash, Plotly, Bootstrap components, Cytoscape |
| **NAS** | `pip install "positronixlab[nas]"` | Ax 1.3+ multi-objective optimization (`ax-platform`) |
| **UMAP** | `pip install "positronixlab[umap]"` | `umap-learn` (CPU manifold projection; `cuML` auto-detected if installed) |
| **SOTA Baselines** | `pip install "positronixlab[baselines]"` | `NeuroSurgeon`, `transformers`, `opencv-python-headless`, `pillow` |
| **Everything** | `pip install "positronixlab[all]"` | All runtime extras above |
| **Development** | `pip install "positronixlab[dev]"` | `pytest`, `pytest-cov`, `pytest-xdist`, `build`, `twine` |

To install from source in editable mode:

```bash
git clone https://github.com/cesare-montresor/PositronixLab.git
cd PositronixLab
pip install -e ".[dev]"
```

---

## Quickstart

### Unsupervised Atlas Discovery vs. SOTA Weight Masking

Discover subtask-specific subnetworks in **one unsupervised Atlas pass** and compare directly against supervised Csordás Gumbel-Sigmoid and NeuroSurgeon masks on the exact same model and dataset:

```python
import torch

from positronixlab.baselines import (
    CsordasGumbelDiscoverer,
    DataProviderLoader,
    ModelLoader,
    NeuroSurgeonDiscoverer,
    is_neurosurgeon_available,
)
from positronixlab.tasks import AtlasDiscoverer, MaskApplier

# 1. Load a compositional benchmark and model
dp = DataProviderLoader.create(
    "csordas:add_mul", n_train=512, n_val=128, n_test=128, n_digits=1, batch_size=64
)
model = ModelLoader.create(
    "csordas:mlp",
    n_inputs=dp.num_features,
    n_outputs=dp.num_classes,
    layer_sizes=[48, 32],
    device="cpu",
)

# 2. Train the base model on the full (add + mul) task
opt = torch.optim.Adam(model.parameters(), lr=2e-2)
for _ in range(12):
    for x, y in dp.train_loader():
        opt.zero_grad()
        loss = model.loss(x, y, model(x))
        loss.backward()
        opt.step()

val_add = dp.get_subtask("add").val_loader()
val_mul = dp.get_subtask("mul").val_loader()

# 3. PositronixLab Atlas: ONE unsupervised pass -> masks for BOTH subtasks
atlas = AtlasDiscoverer(atlas_epochs=2, space_dim=8, k_min=2, k_max=4)
atlas_masks = atlas.discover_subtasks(
    model,
    reference_dataloader=dp.val_loader(),
    subtask_dataloaders={"add": val_add, "mul": val_mul},
)

# 4. Csordás Gumbel-Sigmoid baseline: supervised per-subtask mask training
csordas = CsordasGumbelDiscoverer(mask_loss_weight=1e-2, lr=5e-2, epochs=4)
cs_add = csordas.discover(model, val_add, mask_name="csordas:add")

# 5. Evaluate any mask non-destructively via MaskApplier
with MaskApplier(model, atlas_masks["add"], mode="retain"):
    # Forward passes here retain only the discovered "add" subnetwork
    pass
```

See [examples/tasks/example_atlas_vs_sota.py](examples/tasks/example_atlas_vs_sota.py) for the complete runnable script.

---

## Package Architecture

| Module | Responsibility |
| :--- | :--- |
| `positronixlab.spacemath` | Functional scoring, contrastive kernels, weighting, losses, reducers, and `torch.compile` utilities |
| `positronixlab.modularspace` | `LayerSpace`, `LayerSpaceConfig`, `SpaceStep`, presets, `ModularSpace`, and `ClusterIdentityTracker` |
| `positronixlab.modularobserver` | `ModelMapper` (static + dynamic graph tracing), `ModularObserver`, and `LayerObserver` |
| `positronixlab.modularcluster` | Adaptive FAISS K-Means (`ModularClusterKmeans`) and `StreamingClusterTracker` |
| `positronixlab.tasks` | `SubnetworkMask`, `MaskApplier`, `SubnetworkDiscoverer`, `AtlasDiscoverer`, and `ModularityBenchmark` |
| `positronixlab.baselines` | Csordás, Lepori, and NeuroSurgeon adapters, models, datasets, `ModelLoader`, and `DataProviderLoader` |
| `positronixlab.dataprovider` | `DataConfig`, `DataProvider`, `CompositionalDataProvider`, and 16 built-in datasets |
| `positronixlab.deeplearning` | Training and evaluation loops (`DeeplearningTrain`, `DeeplearningEval`, `EarlyStopping`) |
| `positronixlab.metamodel` | `MetaModel`, `MetaModelConfig`, `MetaCriterion`, and `MetaTransform` |
| `positronixlab.pipeline` | `BaseConfig` (composable YAML/JSON dataclass configs) and `parallelazy`-backed `Pipeline` |
| `positronixlab.nas` | Ax 1.3+ `Experiment`, `TrialConfig`, and `TrialPipeline` |
| `positronixlab.stats` | Streaming `Reducer`s (`Ema`, `Window`, `Displacement`), `DisplacementFilter`, and `UmapProcessor` |
| `positronixlab.modularui` | Thread-safe Dash/Plotly/Cytoscape live visualization components and layouts |

---

## Ecosystem

* **[parallelazy](https://github.com/cesare-montresor/parallelazy)** — Lightweight multiprocess worker/plugin runtime (spawn/pipes, topic pub/sub, CUDA-ready) powering `positronixlab.pipeline`.
* **AtlasCompose** — Application-level experiment orchestration combining `positronixlab` and `parallelazy`.

---

## Documentation & Examples

* Per-module architecture guides (`*_SYSTEM.md`) and usage tutorials (`*_USAGE.md`) live under [`docs/`](docs/).
* Standalone runnable examples for each subsystem live under [`examples/`](examples/).

---

## Development & Testing

```bash
# Run the full test suite
pytest -q

# Run baselines and Atlas subnetwork discovery tests
pytest tests/baselines tests/tasks/test_masks_and_discoverers.py tests/tasks/test_atlas_discoverer.py tests/tasks/test_sota_benchmarks.py -v
```

> **Note:** In read-only home environments where `torch.compile` cannot write to `~/.cache`, set `TRITON_CACHE_DIR=/tmp/triton_cache TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache`.

---

## License

Licensed under the [Mozilla Public License 2.0 (MPL-2.0)](LICENSE).
