# Stats Usage Guide

How to configure and consume the stats aggregation pipeline (`StatsPlugin` +
`Stream`/`Reducer`/`Processor`).

## Quick Start (defaults)

With no options, `StatsPlugin` runs three streams out of the box:

```python
from positronixlab.positronixlab.plugin_stats import StatsPlugin

plugins = [
    # ... producers: ObservableTrainPlugin, ModularSpacePlugin, ClusterPlugin ...
    StatsPlugin.Descriptor(),   # default streams + cluster sync
]
plugin_manager.load_plugins(plugins, parallel=True)
```

Defaults (see `DEFAULT_STREAMS`):

| Stream | Raw topic | Reducer | Rate | Processor | Output topic |
|---|---|---|---|---|---|
| `positions` | `positions` | `ema` α=0.3 | 0.25 s | PCA (2D, fit on 100 snapshots) | `stats_positions` |
| `spaceloss` | `metrics` | `mean` | 0.5 s | overall mean → extra topic | `stats_spaceloss` (+ `stats_spaceloss_overall`) |
| `training` | `metrics` | `mean` | 0.5 s | — | `stats_training` |

Cluster snapshots from topic `clusters` are projected with each layer's PCA and
republished on `stats_clusters` tagged with the absolute `position_index`.

## Configuring Streams

All configuration lives in `options["streams"][<name>]`, merged over the defaults
by `Stream.merge_streams`:

```python
StatsPlugin.Descriptor({
    "tick_s": 0.02,
    "save_path": "/path/state.pkl",          # .json / .yaml / .pkl / .pt
    "streams": {
        # override a default stream (shallow merge on the base config)
        "positions": {"interval_s": 1.0, "processor_args": {"n_samples": 50}},

        # disable a default stream: None or {"enabled": false}
        "spaceloss": None,

        # add a brand-new stream (must define topic + reducer)
        "my_metric": {
            "topic": "metrics",
            "extract": "training",           # extractor name in EXTRACTORS
            "reducer": "window",
            "reducer_args": {"maxlen": 500},
            "interval_s": 2.0,               # or "every_n": 100 (takes precedence)
            "out_topic": "stats_my_metric",
            "out_env": "stats_my_metric",    # env var with the latest full state
            "keep_history": True,
        },
    },
    "clusters": None,                        # or {"topic": ..., "out_topic": ...} / None to disable
})
```

Merge rules:

- `None` or `{"enabled": false}` → stream removed; `"enabled"` is stripped otherwise.
- Changing `reducer`/`processor` **without** new args drops the stale `*_args`.
- A new stream must at least define `topic` and `reducer`; everything else has a
  default (`extract` defaults to the stream name, `keep_history` defaults to `True`).

### Choosing a reducer

| Reducer | Args | Use when |
|---|---|---|
| `last` | — | Pure decimation; you only care about the newest value. |
| `ema` | `alpha` (0..1, default 0.3) | Smooth noisy signals before decimating (default for positions). |
| `mean` | — | Block average since last emission (default for metrics). |
| `window` | `maxlen` (default 200) | Keep the recent history of a scalar/small metric; emits shape `[n, ...]`. Not for positions. |

### Choosing a processor

| Processor | Args | Output |
|---|---|---|
| `pca` | `n_components=2`, `n_samples=100` | Per-layer 2D points + progressive `index` (timeline reference for clusters). |
| `umap` | `n_components=2`, `n_samples=200`, `n_neighbors=15`, `min_dist=0.1` | Same pattern with UMAP (backend: cuML → parametric_umap → umap-learn). Heavier than PCA. |
| `overall` | `topic`, `key="overall"` | Pass-through + one extra message on `topic` with the mean across keys. |

## Consuming the Output

### Topics

Every emitted message is stamped by the plugin:

```python
{
    "key":   layer_id_or_metric_name,
    "value": torch.Tensor,          # aggregated (and projected, if PCA/UMAP)
    "stream": "positions",          # stream name
    "seq":   42,                    # per-stream monotonic emission counter
    "count": 17,                    # raw messages folded into this block
    "t":     1728960000.123,        # epoch seconds
    "index": 5,                     # PCA/UMAP only: timeline index of the step
    "names": [...],                 # only if configured on the stream
}
```

Subscribe like any other plugin topic:

```python
self.topic_subscribe("stats_positions", self._on_positions)   # 2D points per layer
self.topic_subscribe("stats_spaceloss", self._on_spaceloss)   # [loss, loss_on, loss_off, dict_center] per layer
self.topic_subscribe("stats_training", self._on_training)     # {metric: value}
self.topic_subscribe("stats_clusters", self._on_clusters)     # 2D centroids + position_index
```

### Env vars

- `stats_pca_progress` — `{"required": n_samples, "layers": {layer_id: collected}}`,
  updated only when it changes. Use it to show a "collecting PCA" progress bar.
- `<stream.out_env>` (if configured) — latest full state `{key: tensor}` of that stream.

### Saved state

With `save_path` set, `on_stop()` writes the full history (resolution = the
stream's `interval_s`, not the raw rate):

```python
{
  "header": {"timestamp": ...},
  "data": {
    "dynamic_vars", "model_graph", "stats", "observed_layers",   # from env plane
    "training_metrics":  {metric: [float, ...]},
    "layer_spaceloss":   {layer: {name: [float, ...]}},
    "positions_plot":    {layer: [2D points, ...]},
    "pca_history":       {layer: [snapshots still collecting]},
    "cluster_history":   {layer: [resolved cluster snapshots]},
  },
}
```

The payload shape matches the old `ModularUI.save_state`, so existing loaders keep working.

## Using the Building Blocks Directly

`Stream`, reducers and processors are plain classes — usable outside the plugin
(e.g. in a test or a standalone aggregation loop).

```python
import torch
from positronixlab.stats.stream import Stream
from positronixlab.stats.reducer import EmaReducer, MeanReducer, WindowReducer, LastReducer
from positronixlab.stats.processor import PcaProcessor, UmapProcessor, OverallProcessor

# --- one stream, manually driven -------------------------------------------
s = Stream("positions", {
    "topic": "positions",
    "reducer": "ema", "reducer_args": {"alpha": 0.3},
    "processor": "pca", "processor_args": {"n_samples": 10},
    "interval_s": 0.25,
})

# hot path: fold raw messages (O(1))
s.ingest({"layer_id": "fc1", "positions": torch.randn(64, 32)})

# main loop: emit when due
import time
if s.due(time.monotonic()):
    blocks, n = s.drain()          # {layer_id: ema tensor}, raw count
    msgs = s.processor.process({k: t.cpu() for k, t in blocks.items()})
    # msgs: [{"key": "fc1", "value": 2D tensor, "index": i}, ...]

# --- reducers alone ----------------------------------------------------------
r = MeanReducer()
r.update("loss", torch.tensor(0.5))
r.update("loss", torch.tensor(0.3))
assert r.drain()["loss"].item() == 0.4     # block mean, state reset after drain

w = WindowReducer(maxlen=3)
for v in (1.0, 2.0, 3.0, 4.0):
    w.update("x", torch.tensor(v))
assert w.drain()["x"].tolist() == [2.0, 3.0, 4.0]   # rolling window

# --- processors alone ---------------------------------------------------------
pca = PcaProcessor(n_components=2, n_samples=5)
for i in range(5):
    out = pca.process({"fc1": torch.randn(8, 16)})   # collecting: no output yet
out = pca.process({"fc1": torch.randn(8, 16)})       # fitted: emits 2D points + index
assert pca.ready("fc1") and pca.last_index("fc1") == 0
pts2d = pca.project("fc1", some_raw_points_np)       # None before fit
```

## Threading Model (what's safe where)

| Operation | Thread | Notes |
|---|---|---|
| `Stream.ingest` / reducer `update` | receive thread | O(1) fold under a short per-stream lock; never do heavy work here. |
| `Stream.due` / `drain`, processor `process`, publish, history append | main loop (`StatsPlugin.main`) | All heavy work (PCA/UMAP fit & transform, CPU conversion) happens here. |
| Cluster receive handler | receive thread | Only copies (`_to_np`) and queues; projection/publish happen in the main loop. |

Rules of thumb:

- Never keep references to received message buffers — clone/copy them
  (`_as_tensor` does not copy; `_to_np` does).
- Processors are internally locked, so `project()`/`progress()` can be called
  from other threads (e.g. an RPC handler), but the main loop is the intended
  consumer of `process()`.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| No output for a layer, `stats_pca_progress` shows it collecting | PCA/UMAP needs `n_samples` snapshots first; wait or lower `n_samples`. |
| Cluster snapshots missing from `stats_clusters` | They stay queued until the layer's PCA is fitted; also check `clusters` option isn't `None` and a PCA processor exists. |
| Sudden restart of collection for a layer | Layer shape changed (e.g. model resize) — collection buffer is cleared by design. |
| Stale `reducer_args` after switching reducer kind | Provide the new args explicitly; `merge_streams` only drops stale args when you change the kind without giving new ones. |
| UMAP slow | Expected: UMAP `transform` is heavier than PCA; consider `pca`, a GPU backend (cuML/parametric_umap), or a larger `interval_s`. |
