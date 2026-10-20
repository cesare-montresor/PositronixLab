# PositronixLab Stats System

Aggregation layer that sits between raw high-frequency producers (training, space
plugins) and consumers (UI, other plugins). Raw messages are folded into per-key
accumulators in O(1), reduced at a configurable rate, optionally transformed
(PCA/UMAP projection), synchronized with the cluster timeline, stored in history
and republished as low-rate ordered blocks.

Everything heavy (PCA/UMAP fit & transform, CPU conversion, publish) runs in the
`StatsPlugin.main()` loop — **never** in the receive thread.

## Architecture

```
 raw producers (high rate)                     StatsPlugin                          consumers (low rate)
┌──────────────────────────┐   topics    ┌─────────────────────────────────────┐   ┌──────────────────────┐
│ plugin_space             │  positions  │  Stream "positions"                 │   │                      │
│  → {layer_id, positions} ├────────────►│  extractor → reducer(ema)           │   │  UI / other plugins  │
├──────────────────────────┤             │        ↓ drain (interval_s)         │   │                      │
│ plugin_space             │   metrics   │  processor(pca) → 2D points + index ├─►│  topic: stats_*       │
│  → {name:"spaceloss",...}├────────────►│                                      │   │  env:  stats_*      │
│ plugin_train             │             │  Stream "spaceloss" / "training"    │   │                      │
│  → {name:"training", ...}┤             │  extractor → reducer(mean)          │   │  history + save_state│
└──────────────────────────┘             └───────────────▲─────────────────────┘   └──────────────────────┘
                                                         │ cluster sync (project centroids,
        plugin_cluster ──► topic "clusters" ─────────────┘ tag with position_index)
```

Per-stream pipeline:

```
msg ──► extractor ──► [(key, value)] ──► reducer.update()      [receive thread, O(1), locked]
                                              │
                              due? (interval_s / every_n)
                                              ▼
                        reducer.drain() → {key: aggregated tensor}   [main loop]
                                              ▼
                        processor.process(items) → [messages]        [main loop]
                                              ▼
              topic_publish(out_topic) · history append · env_set(out_env)
```

## Module Layout

| File | Contents |
|---|---|
| `positronixlab/stats/common.py` | Tensor/numpy conversion helpers (`_as_tensor`, `_to_np`) |
| `positronixlab/stats/reducer.py` | `Reducer` ABC + `Last/Ema/Mean/Window` reducers, `REDUCERS` registry |
| `positronixlab/stats/processor.py` | `Processor` ABC + `Pca/Umap/Overall` processors, `PROCESSORS` registry |
| `positronixlab/stats/stream.py` | `Stream`: per-stream config, ingest/due/drain, `merge_streams` |
| `positronixlab/positronixlab/plugin_stats.py` | `StatsPlugin`: lifecycle, cluster sync, history, persistence |

---

# API Reference

## `common.py`

### `_as_tensor(value: Any) -> torch.Tensor`

Converts any value to a **detached float32** tensor.
Accepts `torch.Tensor`, list/tuple of tensors (stacked), or anything numpy-convertible.

> **Warning:** does NOT guarantee a copy. Callers that keep the value must
> `.clone()` — never hold references to received message buffers (they may be
> recycled IPC/cudagraph buffers).

### `_to_np(value: Any) -> np.ndarray`

Returns an independent numpy copy of the value, decoupled from the message buffer.

## `reducer.py`

Reducers keep per-key state; `drain()` returns only "dirty" keys (touched since
the last drain) as `{key: aggregated tensor}` and clears the dirty set.

### `Reducer` (ABC)

| Member | Description |
|---|---|
| `update(key: str, value: torch.Tensor)` | Fold one raw value into the accumulator (abstract). |
| `_snapshot(key: str) -> torch.Tensor` | Aggregated value for `key`, independent of internal state (abstract). |
| `_after_drain(key: str)` | Hook called after a key is drained; used by block reducers to reset. |
| `drain() -> Dict[str, torch.Tensor]` | Snapshot all dirty keys, then clear the dirty set. |

### `LastReducer()`

Keeps only the last value per key (pure decimation).

### `EmaReducer(alpha: float = 0.3)`

Exponential moving average: filters before decimating to avoid aliasing.
State is reset when a key's shape changes.

### `MeanReducer()`

Arithmetic mean of all values received since the last drain (block semantics).
State is deleted in `_after_drain`.

### `WindowReducer(maxlen: int = 200)`

Rolling window of the last `maxlen` values; emits the whole window stacked as
shape `[n, ...]`. Suitable for scalar/small metrics, **not** for positions.

### Registry

```python
REDUCERS = {"last": LastReducer, "ema": EmaReducer, "mean": MeanReducer, "window": WindowReducer}
```

## `processor.py`

Processors run downstream of the reducer in the main loop.

### `Processor` (ABC)

| Member | Description |
|---|---|
| `process(items: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]` | Items are CPU aggregated tensors for one emission (`key → tensor`). Returns messages `{"key", "value", **extra}`. A `"topic"` field in a message routes it to another topic and excludes it from history and env. |

### `PcaProcessor(n_components: int = 2, n_samples: int = 100)`

Per-layer PCA. Collects `n_samples` snapshots, fits, then emits already-projected
positions `(n, n_components)` with a progressive `index` — the reference timeline
for cluster synchronization. At fit time each collected snapshot is projected and
emitted as a separate step (preserving timeline sync). If a layer's shape changes
during collection, collection restarts.

| Method | Description |
|---|---|
| `ready(layer) -> bool` | PCA fitted for the layer. |
| `last_index(layer) -> int` | Index of the last emitted step (0 if none). |
| `project(layer, arr) -> Optional[np.ndarray]` | Project raw points with the fitted PCA; `None` before fit. |
| `progress() -> Dict[str, Any]` | `{"required": n_samples, "layers": {layer: collected}}`. |
| `collecting() -> Dict[str, List[np.ndarray]]` | Snapshots still being collected (for persistence). |

### `UmapProcessor(n_components=2, n_samples=200, n_neighbors=15, min_dist=0.1)`

Same collect-then-transform pattern as PCA, using UMAP. Backend is selected at
import time: **cuML** (NVIDIA) → **parametric_umap** (Apple/AMD/Intel GPU) →
**umap-learn** (CPU). Note: UMAP `transform` is computationally heavier than PCA.

### `OverallProcessor(topic: str, key: str = "overall")`

Pass-through of all items plus one extra message on `topic` containing the mean of
the latest known value of each key (e.g. average spaceloss across layers).

### Registry

```python
PROCESSORS = {"pca": PcaProcessor, "umap": UmapProcessor, "overall": OverallProcessor}
```

## `stream.py`

### `Stream(name: str, cfg: Dict[str, Any])`

One configured stream. Config keys:

| Key | Type | Description |
|---|---|---|
| `topic` | `str` | Raw topic to ingest from (required). |
| `extract` | `str` | Extractor name in `EXTRACTORS` (defaults to the stream name). |
| `reducer` / `reducer_args` | `str` / `dict` | Reducer kind and constructor args. |
| `processor` / `processor_args` | `str` / `dict` | Optional processor kind and args. |
| `names` | `List[str]` | Optional field names attached to emitted messages (e.g. metric component names). |
| `interval_s` | `float` | Emit at most every N seconds. |
| `every_n` | `int` | Emit after N ingested messages (takes precedence over interval). |
| `out_topic` | `str` | Topic where aggregated blocks are published (`None` = no publish). |
| `out_env` | `str` | Env var name holding the latest full state (`None` = no env). |
| `keep_history` | `bool` | Append emitted values to in-memory history (default `True`). |

| Method | Description |
|---|---|
| `ingest(msg)` | **Hot path** (receive thread): extract items and fold into the reducer under a short lock. |
| `due(now) -> bool` | True when an emission is due (`every_n` reached or `interval_s` elapsed with pending data). |
| `drain() -> Tuple[Dict[str, torch.Tensor], int]` | Drain dirty keys from the reducer; returns `(blocks, n_ingested)` and resets the counter. |

### `Stream.merge_streams(user, streams) -> Dict[str, Dict[str, Any]]` (classmethod)

Merges user overrides onto default stream configs:

- entry `None` or `{"enabled": false}` → stream removed;
- `"enabled"` key is stripped from overrides;
- changing `reducer`/`processor` without providing new args drops the stale
  `*_args` of the previous kind;
- other keys are shallow-merged onto the base config.

## `plugin_stats.py` — `StatsPlugin`

### Options (`default_options()`)

| Option | Default | Description |
|---|---|---|
| `tick_s` | `0.02` | Main-loop tick period (seconds). |
| `save_path` | `None` | Path for `save_state()`; format by extension (`.json`, `.yaml`, `.pkl/.pickle`, `.pt`). |
| `streams` | `DEFAULT_STREAMS` | Stream configs, merged via `Stream.merge_streams`. |
| `clusters` | `{"topic": "clusters", "out_topic": "stats_clusters"}` | Cluster-sync config; `None` disables it. |

### Default streams (`DEFAULT_STREAMS`)

| Stream | topic | reducer | interval_s | processor | out_topic |
|---|---|---|---|---|---|
| `positions` | `positions` | `ema` (α=0.3) | 0.25 | `pca` (2D, n_samples=100) | `stats_positions` |
| `spaceloss` | `metrics` | `mean` | 0.5 | `overall` → topic `stats_spaceloss_overall` | `stats_spaceloss` |
| `training` | `metrics` | `mean` | 0.5 | — | `stats_training` |

Note: `spaceloss` and `training` share the raw topic `metrics`; the plugin opens
**one** subscription per topic and dispatches to all streams that read it.

### Extractors (`EXTRACTORS`)

Raw message → `[(key, value)]`:

| Name | Accepted message | Emits |
|---|---|---|
| `positions` | `{"layer_id": str, "positions": tensor}` | `[(layer_id, positions)]` |
| `spaceloss` | `{"name": "spaceloss", "layer_id": str, "metric": [loss, loss_on, loss_off, dict_center]}` | `[(layer_id, metric_vector)]` |
| `training` | `{"name": "training", "metric": {k: v, ...}}` | `[(k, v), ...]` |

### Lifecycle

- **`on_start()`** — subscribes to each raw topic (shared dispatcher per topic);
  subscribes to the clusters topic if cluster sync is enabled and a PCA processor
  exists; publishes initial PCA progress.
- **`main()`** — every `tick_s`: emit due streams, resolve pending clusters,
  publish PCA progress when changed, sleep.
- **`on_stop()`** — final flush of all streams, resolve remaining clusters,
  `save_state()`.

### Emission (`_emit`)

1. `drain()` the stream → aggregated CPU tensors;
2. run the processor (if any) to get messages;
3. each message is stamped with `stream`, `seq` (per-stream monotonic), `count`
   (raw messages folded into this block) and `t` (epoch seconds); `names` is
   attached when configured;
4. messages carrying a `"topic"` field are published there only (no history, no env);
5. otherwise: publish to `out_topic`, append to history if `keep_history`, and
   update the `out_env` latest-state dict (single `env_set` per emission).

### Cluster synchronization

Cluster snapshots arriving on the clusters topic are copied (`_to_np`) and queued
in the receive thread. In the main loop, for every layer whose PCA is ready, each
queued snapshot has its centroids projected to 2D and is tagged with
`position_index = pca.last_index(layer)` (the absolute position timeline index).
Snapshots arriving before the PCA fit stay queued until the layer is ready.
Resolved snapshots are appended to `cluster_history` and published on
`clusters.out_topic` as:

```python
{"layer_id", "position_index", "labels", "best_k", "best_score", "centroids"}  # centroids: 2D tensor
```

### Env vars published

| Name | Value |
|---|---|
| `stats_pca_progress` | `{"required": n_samples, "layers": {layer_id: collected_count}}` — updated only on change. |
| `<stream.out_env>` (if set) | Latest full state `{key: tensor}` of that stream. |

### `save_state() -> bool`

Persists the same payload shape as the old `ModularUI.save_state` (existing
loaders keep working). History resolution equals the stream's `interval_s`, not
the raw rate. Payload:

```python
{
  "header": {"timestamp": ...},
  "data": {
    "dynamic_vars":     {...},   # env vars minus dedicated/stats_* names; trial_config normalized via to_dict()
    "model_graph":      {...},   # from env "model_graph"
    "stats":            {...},   # from env "space_stats"
    "observed_layers":  {...},   # from env "observed_layers"
    "training_metrics": {metric: [float, ...]},
    "layer_spaceloss":  {layer: {name: [float, ...]}},   # names = SPACELOSS_NAMES
    "positions_plot":   {layer: [2D points, ...]},       # positions stream history
    "pca_history":      {layer: [snapshots still collecting]},
    "cluster_history":  {layer: [resolved cluster snapshots]},
  },
}
```
