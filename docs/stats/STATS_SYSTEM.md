# PositronixLab Stats System

Aggregation layer that sits between raw high-frequency producers (training, space
observers) and low-rate consumers (UI, logging, downstream plugins). Raw messages
are folded into per-key accumulators in O(1), drained at a configurable cadence,
and optionally transformed by a downstream `Processor` (e.g. per-layer UMAP 2D
projection or cross-key averaging).

Everything heavy (UMAP fit & warm-start `fit_transform`, CPU conversion, publish)
is designed to run in the consumer/main loop — **never** in the receive thread.

> **Note:** The concrete `StatsPlugin` wrapper moved to the sibling application
> repo `../AtlasCompose`. `positronixlab/stats` provides the reusable, plugin-agnostic
> building blocks (`common`, `filter`, `reducer`, `processor`, `stream`).

## Architecture

Per-stream pipeline:

```
msg ──► extractor ──► [(key, value)] ──► _as_tensor() ──► reducer.update()   [receive thread, O(1), locked]
                                                                │
                                            due(now)? (every_n / interval_s)
                                                                ▼
                                    reducer.drain() → {key: aggregated tensor}   [main loop]
                                                                ▼
                                    processor.process(items) → [messages]        [main loop]
```

## Module Layout

| File | Contents |
|---|---|
| `positronixlab/stats/__init__.py` | Empty package marker (import from submodules directly) |
| `positronixlab/stats/common.py` | Tensor/numpy conversion helpers (`_as_tensor`, `_to_np`) |
| `positronixlab/stats/filter.py` | `DisplacementFilter`: standalone spatial displacement gate |
| `positronixlab/stats/reducer.py` | `Reducer` ABC + `Last/Ema/Mean/Window/Displacement` reducers, `REDUCERS` registry |
| `positronixlab/stats/processor.py` | `Processor` ABC + `UmapProcessor`, `OverallProcessor`, `PROCESSORS` registry |
| `positronixlab/stats/stream.py` | `Stream`: per-stream config, `ingest`/`due`/`drain`, `merge_streams` |

---

# API Reference

## `common.py`

### `_as_tensor(value: Any) -> torch.Tensor`

Converts `value` to a **detached float32** `torch.Tensor`:
- `torch.Tensor` → `value.detach().to(torch.float32)`
- non-empty `list`/`tuple` whose first element is a `torch.Tensor` → stacks elements reshaped to `()` (0-D scalars)
- anything else → `torch.as_tensor(np.asarray(value, dtype=np.float32))`

> **Warning:** `_as_tensor` does **not** guarantee a copy. If `value` is already
> a `float32` tensor or `float32` NumPy array, the returned tensor shares memory
> with the input buffer. Any caller retaining the value **must `.clone()`**
> (never hold references to recycled IPC/cudagraph buffers).

### `_to_np(value: Any) -> np.ndarray`

Returns an independent NumPy copy of `value` (`value.detach().cpu().numpy().copy()`
for tensors; `np.array(value)` otherwise), decoupled from the input buffer.

---

## `filter.py`

### `DisplacementFilter(threshold: float = 0.5, method: str = "max")`

Standalone event-driven spatial gate. Keeps a per-key reference keyframe tensor
and signals emission when L2 displacement crosses `threshold`.

| Method | Description |
|---|---|
| `check_and_update(key: str, value: torch.Tensor) -> bool` | Returns `True` (and updates keyframe) on first call, shape change, or when displacement `> threshold`; `False` otherwise. `method="max"` uses max per-row L2 norm (`dim=-1`); any other value uses RMS across rows. |

> **Caveats:** Not wired into `Stream` (duplicates `DisplacementReducer`). Does
> not validate `method` in `__init__`, does not `.detach()` stored keyframes,
> and is not internally locked.

---

## `reducer.py`

Reducers keep per-key state; `drain()` returns only "dirty" keys (touched since
the last drain) as `{key: aggregated tensor}` and clears the dirty set. All
built-in reducers clone input tensors before storing them.

### `Reducer` (ABC)

| Member | Description |
|---|---|
| `update(key: str, value: torch.Tensor)` | Fold one raw value into the accumulator (abstract). |
| `_snapshot(key: str) -> torch.Tensor` | Aggregated value for `key`, independent of internal state (abstract). |
| `_after_drain(key: str)` | Hook called after a key is drained; used by block/pending reducers to reset. |
| `drain() -> Dict[str, torch.Tensor]` | Snapshot all dirty keys, call `_after_drain(key)`, and clear `_dirty`. |

### Concrete Reducers

| Class | Args | Semantics | Internal Lock |
|---|---|---|---|
| `LastReducer` | — | Pure decimation: keeps only the latest cloned tensor per key. | No (rely on `Stream._lock`) |
| `EmaReducer` | `alpha: float = 0.3` | Exponential moving average (`s = (1 - alpha) * s + alpha * value`). Resets on shape change. | No (rely on `Stream._lock`) |
| `MeanReducer` | — | Block arithmetic mean of all values received since the last `drain()`. Resets on shape change and clears state in `_after_drain`. | Yes (`self._lock`) |
| `WindowReducer` | `maxlen: int = 200` | Rolling `deque(maxlen=maxlen)` of cloned values; emits `torch.stack(...)` of shape `[n, ...]`. Suitable for scalar/small metrics, **not** positions. | No (rely on `Stream._lock`) |
| `DisplacementReducer` | `threshold: float = 0.5`, `method: str = "max"` | Spatial gating: emits on init, shape change, or when L2 displacement (`"max"` or `"rms"`, validated in `__init__`) from last keyframe exceeds `threshold`. Drops `_pending[key]` in `_after_drain`. | Yes (`self._lock`) |

### Registry

```python
REDUCERS = {
    "last": LastReducer,
    "ema": EmaReducer,
    "mean": MeanReducer,
    "window": WindowReducer,
    "displacement": DisplacementReducer,
}
```

---

## `processor.py`

Processors run downstream of `reducer.drain()` in the main loop.

> **UMAP backend fallback:** `processor.py` tries `cuml.manifold.UMAP` first, falls back to CPU `umap.UMAP` (`umap-learn`), and sets `UMAP = None` if neither is installed (raising only if `UmapProcessor` is actually instantiated).

### `Processor` (ABC)

| Member | Description |
|---|---|
| `process(items: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]` | `items` maps `key → aggregated CPU tensor`. Returns messages `{"key", "value", **extra}`. A `"topic"` field routes the message to an alternate topic (excluding it from stream history/env in plugin consumers). |

### `UmapProcessor`

```python
UmapProcessor(
    n_components: int = 2,
    n_samples: int = 50,
    min_dist: float = 0.1,
    min_neighbors: int = 10,
    max_neighbors: int = 50,
    seed: Optional[int] = 42,
)
```

Per-layer UMAP 2D projection with warm-start updates (internally locked via `self._lock`):
1. **Collection phase**: Buffers `n_samples` snapshots per layer. If feature dimensions (`shape[1:]`) change during collection, the buffer is cleared.
2. **Initial fit**: Concatenates buffered snapshots, fits a `UMAP` mapper with `init="spectral"` and `n_neighbors = clamp(int(sqrt(num_neurons)), min_neighbors, max_neighbors)`, transforms all buffered snapshots, stores the last frame's 2D coordinates in `_last_2d[layer]`, and emits one message `{"key": layer, "value": tensor_2d, "index": i}` per buffered snapshot.
3. **Streaming warm-start phase**: For each subsequent frame, creates a new `UMAP` instance anchored with `init=_last_2d[layer]`, runs `fit_transform(pts)`, updates `_last_2d[layer]`, and emits `{"key": layer, "value": tensor_2d, "index": i}`.

Cluster centroids are **not** projected via `UMAP.transform()`; instead use `centroids_2d(layer, labels)`.

| Method | Description |
|---|---|
| `process(items) -> List[Dict[str, Any]]` | Ingest CPU tensors `{layer: [N, D]}` and return emitted 2D frames. |
| `ready(layer: str) -> bool` | Whether the layer has completed initial UMAP fit. |
| `last_index(layer: str) -> int` | Last emitted step index (`max(steps - 1, 0)`). |
| `centroids_2d(layer: str, labels: np.ndarray) -> Optional[np.ndarray]` | 2D cluster centroids computed as the mean of member neurons' latest 2D positions, ordered by `np.unique(labels)`. Returns `None` if not ready or if `labels` length mismatches. |
| `progress() -> Dict[str, Any]` | `{"required": n_samples, "layers": {layer: collected_count}}`. |
| `collecting() -> Dict[str, List[np.ndarray]]` | Shallow copy of snapshots currently buffered per layer. |

### `OverallProcessor(topic: str, key: str = "overall")`

Pass-through of all `items` plus one extra message:
`{"topic": self.topic, "key": self.key, "value": torch.stack(list(self._latest.values())).mean(0)}`
representing the mean across the latest known value of every key seen so far (e.g. average spaceloss across layers). Not internally locked; requires non-empty `_latest` and matching tensor shapes/devices across keys.

### Registry

```python
PROCESSORS = {
    "umap": UmapProcessor,
    "overall": OverallProcessor,
}
```

> Note: `PcaProcessor` was removed in the UMAP warm-start refactor (`ef5872f`/`672a4c0`).

---

## `stream.py`

### `Extractor`

```python
Extractor = Callable[[Dict[str, Any]], Iterable[Tuple[str, Any]]]
```

### `Stream(name: str, cfg: Dict[str, Any], extractors: Dict[str, Extractor])`

Config keys (`cfg`):

| Key | Type | Description |
|---|---|---|
| `topic` | `str` | Raw topic to ingest from (required). |
| `extract` | `str \| Extractor` | Key into `extractors` (defaults to `name`) or a direct `Extractor` callable. |
| `reducer` / `reducer_args` | `str` / `dict` | Reducer key in `REDUCERS` (required) and optional constructor kwargs. |
| `processor` / `processor_args` | `str` / `dict` | Optional processor key in `PROCESSORS` and constructor kwargs. |
| `names` | `List[str]` | Optional metadata field names (consumed by external plugin wrappers). |
| `interval_s` | `float` | Minimum seconds between emissions. |
| `every_n` | `int` | Emit after `every_n` ingested messages (checked before `interval_s`). |
| `out_topic` / `out_env` / `keep_history` | `str \| None` / `bool` | Plugin-facing routing/persistence metadata (`keep_history` defaults to `True`). |

Methods:

| Method | Description |
|---|---|
| `ingest(msg: Dict[str, Any]) -> None` | **Hot path** (receive thread): runs `extract(msg)`; if truthy, folds each `(key, _as_tensor(value))` into `self.reducer` and increments `_count` under `self._lock`. |
| `due(now: float) -> bool` | Returns `True` if `_count > 0` and either `every_n` is reached or `interval_s` has elapsed since `_last_emit`. |
| `drain() -> Tuple[Dict[str, torch.Tensor], int]` | Under `self._lock`, drains dirty keys from `self.reducer`, resets `_count` to 0, updates `_last_emit = time.monotonic()`, and returns `(blocks, n_ingested)`. |

### `Stream.merge_streams(user, streams) -> Dict[str, Dict[str, Any]]` (classmethod)

Deep-copies `streams` and applies per-stream overrides from `user`:
- `None` or `{"enabled": False}` → removes the stream.
- `"enabled"` key is stripped from remaining overrides.
- For `reducer` and `processor`:
  - If the kind changed and no new `*_args` is given → drops stale `*_args`.
  - If the kind did **not** change and `*_args` is given → merges `{**base[args], **ov[args]}` one level deep.
- Remaining keys are updated via `base.update(ov)`.
