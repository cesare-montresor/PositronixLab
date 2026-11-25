# ModularUI Usage Guide

How to instantiate `ModularUI`, feed aggregated training/space/cluster data into it, and inspect or customize its Dash components.

---

## Quick Start

> **Note:** `positronixlab/modularui/__init__.py` is currently empty, so import directly from `positronixlab.modularui.modularui` (and sibling submodules).

```python
import numpy as np
from positronixlab.modularui.modularui import ModularUI

# Create the UI store + Dash app (set open_browser=False in headless/server runs)
ui = ModularUI(
    host="127.0.0.1",
    port=8050,
    poll_interval_ms=3000,
    positions_maxlen=1000,
    metrics_maxlen=2000,
    cluster_maxlen=500,
    open_browser=False,
)

# Start background Werkzeug server thread
ui.run()

# Push metadata and metrics from your experiment / StatsPlugin subscriber
ui.set_config({"lr": 1e-3, "batch_size": 64, "modular_space_size": 8})
ui.set_var("epoch", (1, 10))
ui.set_observed_layers({"fc1": {"class_name": "Linear", "in_units": 784, "out_units": 64}})
ui.set_model_graph({
    "nodes": {"fc1": {"class_name": "Linear"}, "fc2": {"class_name": "Linear"}},
    "edges": [{"source": "fc1", "target": "fc2"}],
})

# Record training metrics (train.loss and val.loss are plotted on the same chart)
ui.add_training_point("train.loss", 0.42)
ui.add_training_point("val.loss", 0.48)

# Record per-layer and overall space loss
ui.add_spaceloss_point("fc1", ["loss", "loss_on"], [0.31, 0.12])
ui.add_overall_spaceloss_point(["loss", "loss_on"], [0.31, 0.12])

# Record 2D PCA-projected CU positions (shape: [num_cus, 2]) and cluster snapshots
pts = np.random.randn(64, 2)
ui.add_position_step("fc1", pts, index=0)
ui.add_cluster(
    layer_id="fc1",
    position_index=0,
    labels=np.zeros(64, dtype=int),
    best_k=1,
    best_score=0.5,
    centroids=pts.mean(axis=0, keepdims=True),
)

# Shut down cleanly when finished
ui.stop()
```

---

## Data Ingestion API

All ingestion methods are thread-safe and accept Python scalars, NumPy arrays, or CPU `torch.Tensor`s (tensors are detached and moved to CPU automatically via `_to_numpy`).

### Dynamic Variables & Config
- `ui.set_var(name: str, value: Any)` — stores arbitrary experiment state. Special keys recognized by `HomeLayout`:
  - `"epoch"`: `(current_epoch, total_epochs)` tuple displayed in the top bar.
  - `"metrics_history"`: `{family: {label: [[timestamp, value], ...]}}` used as a fallback for training charts.
- `ui.set_config(config: Any)` — accepts a `dict` or any object implementing `.to_dict()` (e.g., `PipelineConfig`).
- `ui.set_model_graph(graph: dict)` — `{"nodes": {id: {"class_name": ...}}, "edges": [{"source": ..., "target": ...}]}`.
- `ui.set_observed_layers(observed_layers)` — `{layer_id: {"class_name": ..., "unit_type": ..., "in_units": ..., "out_units": ..., "activation_fn": ...}}` or a list of layer IDs.
- `ui.set_space_stats(stats: dict)` — `{layer_id: array_of_shape_(num_cus, 2)}` with `[count_on, count_off]` per CU (controls marker size in `LayerBoardDisplay`).
- `ui.set_position_progress(progress: Optional[Dict[str, Any]])` — updates UMAP collection progress (`{"required": n, "layers": {layer: count}}`) displayed before initial projection is ready.

### Scalar Metrics
- `ui.ingest_stat(family: str, msg: dict)` — primary entry point for `StatsPlugin` messages `{"layer_id", "label", "value", "seq", "t"}`.
- `ui.add_point(family: str, layer_id: str, label: str, value: Any)` — appends one scalar to `series[family][layer_id][label]`.
- Compatibility helpers:
  - `ui.add_training_point("train.loss", 0.5)` → `add_point("train", "global", "loss", 0.5)`.
  - `ui.add_spaceloss_point(layer_id, names, values)` → `add_point("spaceloss", layer_id, n, v)` (accepts `values` as a `dict` or sequence/tensor).
  - `ui.add_overall_spaceloss_point(names, values)` → `add_spaceloss_point("global", names, values)`.

### Positions & Clusters
- `ui.add_position_step(layer_id: str, points, index: Optional[int] = None)` — appends one `(N, 2)` projected snapshot. Pass monotonically increasing contiguous `index` values (`0, 1, 2, ...`) so cluster alignment stays exact.
- `ui.add_cluster(layer_id, position_index, labels, best_k, best_score, centroids)` — appends a cluster snapshot tagged with the absolute `position_index`.

---

## Read / Inspection Helpers (Headless & Debugging)

`ModularUI` can be inspected without running the web server:

```python
ui.get_training_groups()           # {"loss": {"train": [...], "val": [...]}}
ui.get_family_groups("spaceloss")  # {"loss": {"fc1": [...], "mean (all layers)": [...]}}
ui.get_known_layer_ids()           # sorted union of observed, positioned, and progressing layers
ui.get_position_steps("fc1")       # number of retained steps
ui.get_position_at("fc1", 0)       # (N, 2) ndarray at relative index 0
ui.get_cluster_for_index("fc1", 0) # most recent cluster snapshot <= absolute step
ui.debug_state()                   # compact summary dict for debugging
ui.reset()                         # clears all stores
```

---

## Headless Component & Figure Building

Widgets and layouts can be rendered directly to Dash component trees and Plotly `go.Figure` objects in tests or scripts without calling `ui.run()`:

```python
layout_tree = ui.home_layout.page_layout()
metrics_tree = ui.home_layout.training_metrics.render()
board_tree = ui.home_layout.layer_board.render(visible_layers=["fc1"])
```

See [example_modularui_headless.py](file:///home/cesare/Projects/PositronixLab/examples/modularui/example_modularui_headless.py) for a complete runnable example.

---

## Known Caveats

- Pass `open_browser=False` in headless/CI environments (the default is `True`).
- Pass a 2-tuple `(epoch, epochs)` to `ui.set_var("epoch", ...)` — scalar integers are not accepted by `HomeLayout._format_epoch`.
- Keep `add_position_step(..., index=...)` contiguous (`0, 1, 2, ...`) when pairing with `add_cluster` so `_position_offset_locked` aligns relative and absolute indices.
- Avoid passing a `dict` of CUDA tensors to `add_spaceloss_point`; move scalar values to CPU/Python floats first.
- See [`reports/modularui_analysis.md`](file:///home/cesare/Projects/PositronixLab/reports/modularui_analysis.md) for the full list of known defects and design notes.
