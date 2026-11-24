# Stats Usage Guide

How to use the `positronixlab/stats` building blocks (`Stream`, `Reducer`,
`Processor`, `DisplacementFilter`) directly or inside a plugin such as
`StatsPlugin` (which lives in `../AtlasCompose`).

## Choosing a Reducer

| Name (`REDUCERS`) | Class | Args | Use when |
|---|---|---|---|
| `last` | `LastReducer` | — | Pure decimation; you only care about the newest value. |
| `ema` | `EmaReducer` | `alpha=0.3` | Smooth noisy signals before decimating. |
| `mean` | `MeanReducer` | — | Block arithmetic mean since last `drain()` (typical for training/loss metrics). |
| `window` | `WindowReducer` | `maxlen=200` | Rolling window emitting shape `[n, ...]`. Suitable for scalar/small metrics, **not** positions. |
| `displacement` | `DisplacementReducer` | `threshold=0.5`, `method="max"` (`"max"` \| `"rms"`) | Event-driven spatial gating: emit only when L2 movement from the last keyframe exceeds `threshold`. |

## Choosing a Processor

| Name (`PROCESSORS`) | Class | Args | Output |
|---|---|---|---|
| `umap` | `UmapProcessor` | `n_components=2`, `n_samples=50`, `min_dist=0.1`, `min_neighbors=10`, `max_neighbors=50`, `seed=42` | Per-layer 2D points + progressive `index` via initial fit + warm-start `fit_transform`. Compute 2D cluster centroids with `centroids_2d(layer, labels)`. |
| `overall` | `OverallProcessor` | `topic`, `key="overall"` | Pass-through of all items + one extra message on `topic` with the mean across all keys seen so far. |

> **Note:** `PcaProcessor` was removed in favor of warm-start `UmapProcessor`.
> Note also that `processor.py` currently requires a working `cuml` import at
> module load time (see `analysis.md`).

## Using the Building Blocks Directly

All components are plain Python classes usable outside any plugin runtime.
See also the runnable script [example_stats.py](../examples/example_stats.py).

### Reducers & `DisplacementFilter`

```python
import torch
from positronixlab.stats.reducer import (
    LastReducer,
    EmaReducer,
    MeanReducer,
    WindowReducer,
    DisplacementReducer,
)
from positronixlab.stats.filter import DisplacementFilter

# Block mean (resets state on drain)
mean_r = MeanReducer()
mean_r.update("loss", torch.tensor(0.5))
mean_r.update("loss", torch.tensor(0.3))
assert abs(mean_r.drain()["loss"].item() - 0.4) < 1e-6

# Rolling window
win_r = WindowReducer(maxlen=3)
for v in (1.0, 2.0, 3.0, 4.0):
    win_r.update("x", torch.tensor(v))
assert win_r.drain()["x"].tolist() == [2.0, 3.0, 4.0]

# Spatial displacement gating
disp_r = DisplacementReducer(threshold=0.5, method="max")
p0 = torch.zeros(16, 4)
disp_r.update("fc1", p0)                       # initial keyframe -> dirty
assert "fc1" in disp_r.drain()
disp_r.update("fc1", p0 + 0.05)                # below threshold -> suppressed
assert disp_r.drain() == {}
disp_r.update("fc1", p0 + 1.0)                 # above threshold -> dirty
assert "fc1" in disp_r.drain()

# Standalone spatial gate
gate = DisplacementFilter(threshold=0.5, method="rms")
assert gate.check_and_update("fc1", p0) is True
assert gate.check_and_update("fc1", p0 + 0.01) is False
```

### `Stream` (Manually Driven)

`Stream` requires three arguments: `(name, cfg, extractors)`. `extract` may be
a string key in `extractors` (defaulting to `name`) or a direct callable.

```python
import time
import torch
from positronixlab.stats.stream import Stream

extractors = {
    "spaceloss": lambda msg: [(msg["layer_id"], msg["metric"])]
    if msg.get("name") == "spaceloss" else [],
}

s = Stream(
    "spaceloss",
    {
        "topic": "metrics",
        "extract": "spaceloss",
        "reducer": "mean",
        "processor": "overall",
        "processor_args": {"topic": "stats_spaceloss_overall"},
        "every_n": 2,
    },
    extractors,
)

s.ingest({"name": "spaceloss", "layer_id": "fc1", "metric": [0.4, 0.2]})
s.ingest({"name": "spaceloss", "layer_id": "fc2", "metric": [0.8, 0.6]})

if s.due(time.monotonic()):
    blocks, count = s.drain()
    cpu_blocks = {k: t.cpu() for k, t in blocks.items()}
    msgs = s.processor.process(cpu_blocks)
    # msgs includes per-layer items plus {"topic": "stats_spaceloss_overall", "key": "overall", "value": ...}
```

### Merging Stream Configurations (`Stream.merge_streams`)

```python
from positronixlab.stats.stream import Stream

defaults = {
    "positions": {
        "topic": "positions",
        "reducer": "ema",
        "reducer_args": {"alpha": 0.3},
        "processor": "umap",
        "processor_args": {"n_samples": 50, "min_dist": 0.1},
        "interval_s": 0.25,
    },
    "spaceloss": {"topic": "metrics", "reducer": "mean", "interval_s": 0.5},
}

merged = Stream.merge_streams(
    user={
        # Same processor kind: processor_args are merged one level deep
        "positions": {"interval_s": 1.0, "processor_args": {"n_samples": 20}},
        # Disable a stream with None or {"enabled": False}
        "spaceloss": None,
        # Add a new stream
        "training": {"topic": "metrics", "reducer": "mean", "every_n": 10},
    },
    streams=defaults,
)
```

Merge rules:
- `None` or `{"enabled": False}` removes the stream; `"enabled"` is stripped otherwise.
- Changing `reducer`/`processor` **without** providing new `*_args` drops the stale `*_args`.
- Keeping the same `reducer`/`processor` and providing `*_args` shallow-merges the args dict.

### `UmapProcessor` & 2D Cluster Centroids

```python
import numpy as np
import torch
from positronixlab.stats.processor import UmapProcessor

umap_proc = UmapProcessor(n_components=2, n_samples=10, min_neighbors=5, max_neighbors=25)
for _ in range(9):
    assert umap_proc.process({"fc1": torch.randn(32, 8)}) == []

# 10th snapshot triggers initial fit and emits all 10 buffered frames
frames = umap_proc.process({"fc1": torch.randn(32, 8)})
assert umap_proc.ready("fc1") and len(frames) == 10

# Subsequent frames use warm-start fit_transform anchored to the previous 2D frame
next_frames = umap_proc.process({"fc1": torch.randn(32, 8)})

# Compute 2D centroids directly from neuron cluster labels
labels = np.array([0] * 16 + [1] * 16)
centroids = umap_proc.centroids_2d("fc1", labels)  # shape (2, 2), float32
```

## Threading & Buffer Ownership Model

| Operation | Intended Thread | Locking Notes |
|---|---|---|
| `Stream.ingest` / `reducer.update` | Receive thread | Protected by `Stream._lock`. Keep O(1). (`MeanReducer` and `DisplacementReducer` also hold internal locks; `Last`/`Ema`/`Window` do not.) |
| `Stream.due` / `Stream.drain` | Main loop | Protected by `Stream._lock`. |
| `UmapProcessor.process` / `ready` / `centroids_2d` / `progress` | Main loop (or reader thread) | Internally locked via `UmapProcessor._lock`. |
| `OverallProcessor.process` | Main loop only | **Not** internally locked. |

Rules of thumb:
- **Never retain uncloned outputs of `_as_tensor`**: `_as_tensor` may return a
  view over a recycled IPC/cudagraph buffer or NumPy array. All built-in
  reducers `.clone()` on write; `_to_np()` always copies.
- **Convert drained tensors to CPU before `UmapProcessor.process()`**:
  `Stream.drain()` preserves the incoming device, while `UmapProcessor.process()`
  calls `t.numpy()`.

## Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `ImportError` / `OSError` importing `stats.processor` or `stats.stream` | `processor.py:14` unconditionally imports `cuml.manifold.umap.UMAP`. Requires a working cuML/CUDA environment until made optional. |
| No output from `UmapProcessor` while `progress()` grows | Expected during the initial `n_samples` collection phase; wait or lower `n_samples`. |
| `Stream.drain()` returns `({}, n)` with `DisplacementReducer` | Messages were ingested (incrementing `_count`), but none exceeded `threshold`. Skip calling `OverallProcessor.process({})` on empty blocks. |
| `RuntimeError` in `WindowReducer.drain()` after a shape change | `WindowReducer` does not reset its deque when tensor shape changes; recreate the reducer or use `EmaReducer`/`MeanReducer`. |
| `UmapProcessor` collection resets unexpectedly | Input feature dimensions (`shape[1:]`) changed during collection; buffer clears by design. |
