# `positronixlab/stats` — Module Analysis

## 1. Overview & Refactor Status

`positronixlab/stats` is the reusable high-rate → low-rate aggregation layer for PositronixLab. It folds high-frequency tensor streams in $O(1)$ per message on a receive thread, drains dirty keys on a slower cadence, and runs optional downstream projections/aggregations.

| File | Role | Status |
|---|---|---|
| `__init__.py` | Package init | Empty (0 bytes); no top-level re-exports. |
| `common.py` | `_as_tensor`, `_to_np` conversion helpers | Stable; relies on callers cloning views returned by `_as_tensor`. |
| `filter.py` | `DisplacementFilter` spatial gate | Added in `672a4c0`; orphaned (unused across repo, duplicates `DisplacementReducer`). |
| `reducer.py` | `Reducer` ABC + `Last`, `Ema`, `Mean`, `Window`, `Displacement` reducers + `REDUCERS` | Active; `DisplacementReducer` and partial locking (`MeanReducer`, `DisplacementReducer`) added in `672a4c0`. |
| `processor.py` | `Processor` ABC + `UmapProcessor`, `OverallProcessor` + `PROCESSORS` | **Broken on CPU/non-cuML environments** due to unconditional top-level `cuml` import (`672a4c0`). `PcaProcessor` was removed in `ef5872f`/`672a4c0`. |
| `stream.py` | `Stream` (`ingest`, `due`, `drain`, `merge_streams`) | Active; transitively broken on non-cuML environments via `from positronixlab.stats.processor import ...`. |

### Refactor & Ecosystem Notes
- **`StatsPlugin` moved out**: The plugin runtime wrapper (`plugin_stats.py`) moved to sibling repo `../AtlasCompose` in commit `4e70649`. `Stream` still carries plugin-facing metadata attributes (`names`, `out_topic`, `out_env`, `keep_history`, `seq`, `latest`) that are not consumed inside `positronixlab/stats`.
- **`PcaProcessor` removed**: Replaced by warm-start `UmapProcessor` in `ef5872f`/`672a4c0`. Cluster centroid projection changed from `project(layer, centroids)` to `UmapProcessor.centroids_2d(layer, labels)`.
- **`_as_tensor` buffer-view semantics**: `_as_tensor(value)` returns a detached `float32` view without copying when `value` is already a `float32` `torch.Tensor` or `float32` `np.ndarray`. All five reducers properly `.clone()` on write so retained state does not alias recycled IPC/cudagraph buffers. `_to_np` always returns an independent NumPy copy.

---

## 2. Bugs & Defects

### High Severity

1. **Unconditional `cuml` import breaks `processor.py` and `stream.py` without GPU/cuML**
   - **Location**: `positronixlab/stats/processor.py:14` (and transitively `positronixlab/stats/stream.py:13`)
   - **Root cause**: Line 14 executes `from cuml.manifold.umap import UMAP` unconditionally at import time, *before* the `try / except ImportError` fallback at lines 36–39. Furthermore, even lines 36–39 raise `ModuleNotFoundError` if neither `cuml` nor `umap-learn` is installed (e.g., when a user only needs `OverallProcessor` or `Stream` without UMAP).
   - **Impact**: Importing `positronixlab.stats.processor` or `positronixlab.stats.stream` crashes immediately on CPU-only hosts or environments where `libcuml.so` / `~/.cupy` is unavailable, blocking all `Stream` and `OverallProcessor` usage.

2. **`WindowReducer.drain()` crashes on tensor shape change and on `maxlen=0`**
   - **Location**: `positronixlab/stats/reducer.py:110-115`
   - **Root cause**: Unlike `EmaReducer`, `MeanReducer`, and `DisplacementReducer`, `WindowReducer.update()` has no `shape != value.shape` reset guard. Appending tensors of differing shapes causes `torch.stack(list(self._w[key]))` at line 115 to raise `RuntimeError: stack expects each tensor to be equal size`. Similarly, `maxlen=0` discards all items while still adding `key` to `_dirty`, causing `torch.stack([])` to raise `RuntimeError`.
   - **Impact**: Any layer resize or `maxlen=0` config crashes the drain loop.

3. **`OverallProcessor.process()` crashes on empty input and on mismatched shapes/devices**
   - **Location**: `positronixlab/stats/processor.py:210-215`
   - **Root cause**: `torch.stack(list(self._latest.values())).mean(0)` (line 213) raises `RuntimeError: stack expects a non-empty TensorList` when called with `{}` before any key has been seen (e.g., when paired with `DisplacementReducer` gating). It also crashes if keys have differing shapes, differing devices, or integer dtypes.
   - **Impact**: Crash in the main emission loop on empty drains or heterogeneous metric shapes.

4. **`UmapProcessor.process()` crashes on CUDA tensors and on post-fit neuron count changes**
   - **Location**: `positronixlab/stats/processor.py:112`, `119-124`
   - **Root cause**: Line 112 calls `t.numpy()` directly without `.detach().cpu()`. Although `Processor.process` documents CPU input, `Stream.drain()` does not move tensors to CPU. Additionally, after initial fit, warm-start passes `init=self._last_2d[layer]` to `UMAP.fit_transform(pts)` without checking `pts.shape[0] == self._last_2d[layer].shape[0]`; a neuron-count change crashes `fit_transform`.
   - **Impact**: `TypeError` on GPU tensors passed from `Stream.drain()`; shape crash if a layer's neuron count changes after initial fit.

5. **`UmapProcessor._create_mapper()` can set `n_neighbors >= num_neurons` for small layers**
   - **Location**: `positronixlab/stats/processor.py:92-106`
   - **Root cause**: `dynamic_n_neighbors = max(self.min_neighbors, min(calculated_k, self.max_neighbors))` clamps *up* to `min_neighbors=10` even when `num_neurons <= 10` (e.g., MNIST 10-class output layer, Covertype 7-class, Iris 3–4 units).
   - **Impact**: Warm-start `fit_transform(pts)` receives `n_neighbors >= pts.shape[0]`, causing `ValueError` or degenerate UMAP graphs on small layers.

### Medium Severity

6. **`_as_tensor()` crashes on list/tuple of non-scalar tensors**
   - **Location**: `positronixlab/stats/common.py:16-17`
   - **Root cause**: `torch.stack([_as_tensor(v).reshape(()) for v in value])` forces every element to scalar shape `()`, raising `RuntimeError` if elements are 1-D or higher-D tensors. Also, if `value[0]` is a Python float and `value[1]` is a tensor, it falls through to `np.asarray(value, dtype=np.float32)`.
   - **Impact**: Contradicts documented "list/tuple of tensors (stacked)" behavior.

7. **`DisplacementFilter` lacks `method` validation and crashes on empty tensors (`method='max'`)**
   - **Location**: `positronixlab/stats/filter.py:9-11`, `25-29`
   - **Root cause**: Unlike `DisplacementReducer` (`reducer.py:126-127`), `DisplacementFilter.__init__` does not validate `method`, silently routing any typo (e.g. `'MAX'`, `'l2'`) into the `else` (`'rms'`) branch. Both `DisplacementFilter` (line 26) and `DisplacementReducer` (line 153) call `.max()` on an empty tensor when `numel() == 0`, raising `RuntimeError`.
   - **Impact**: Silent wrong metric on typos; crash on empty tensors.

8. **`Stream.ingest()` counts empty generator extractors and gated-out `DisplacementReducer` updates**
   - **Location**: `positronixlab/stats/stream.py:49-64`
   - **Root cause**:
     - `if not items: return` (line 50) is always `False` for lazy generators/iterators (which satisfy `Extractor`'s `Iterable` return type), so an empty generator still increments `self._count`.
     - `self._count += 1` (line 55) runs even when `DisplacementReducer.update()` suppressed the update (leaving `_dirty` empty), causing `Stream.due()` to fire and `Stream.drain()` to return `({}, n)` with `n > 0`.
   - **Impact**: Spurious empty emissions (`blocks == {}`), which in turn trigger Bug #3 if `OverallProcessor` is attached.

9. **`Stream.merge_streams()` crashes when `reducer_args` or `processor_args` is `None`**
   - **Location**: `positronixlab/stats/stream.py:88-89`
   - **Root cause**: `{**base.get(args, {}), **ov[args]}` raises `TypeError: 'NoneType' object is not a mapping` if either `base[args]` or `ov[args]` is explicitly `None`.
   - **Impact**: Crash on valid-looking YAML/JSON overrides like `{"reducer_args": None}`.

### Low Severity

10. **`UmapProcessor.last_index()` is ambiguous between "no emissions" and "first emission (`index=0`)"**
    - **Location**: `positronixlab/stats/processor.py:168-170`
    - **Root cause**: `max(self._steps.get(layer, 0) - 1, 0)` returns `0` both before any frame is emitted (`_steps == 0`) and after the first frame is emitted (`_steps == 1`).
    - **Impact**: Callers cannot distinguish uninitialized from index 0 without also calling `ready(layer)`.

11. **`UmapProcessor.centroids_2d()` crashes on empty `pts`/`labels`**
    - **Location**: `positronixlab/stats/processor.py:189-190`
    - **Root cause**: `np.stack([])` raises `ValueError` when `labels` is empty (0 rows).
    - **Impact**: Edge-case crash on 0-neuron frames.

12. **`DisplacementFilter.check_and_update()` retains autograd graphs on `requires_grad=True` tensors**
    - **Location**: `positronixlab/stats/filter.py:19, 33`
    - **Root cause**: Calls `value.clone()` without `.detach()`.
    - **Impact**: Memory leak if called directly with live autograd tensors outside `_as_tensor`.

---

## 3. Inconsistencies & Design Issues

- **Severe documentation drift (fixed in this review)**:
  - Prior `STATS_SYSTEM.md` and `STATS_USAGE.md` still documented the deleted `PcaProcessor`, old `UmapProcessor(n_neighbors=15, n_samples=200)` signature and 3-tier `cuML → parametric_umap → umap-learn` fallback, old 2-arg `Stream(name, cfg)`, and pre-split `positronixlab.positronixlab.plugin_stats.StatsPlugin`. Neither `DisplacementFilter` nor `DisplacementReducer` was documented.
- **Copy-paste artifacts in `processor.py`**:
  - Duplicate imports of `torch`, `numpy`, `math`, `threading`, and `typing` (`processor.py:2-7` vs `28-33`).
  - Redundant self-import `from positronixlab.stats.processor import Processor` at `processor.py:42`.
  - Unused `logger` in both `processor.py:11` and `stream.py:16`.
- **Inconsistent thread safety**:
  - `MeanReducer` and `DisplacementReducer` hold their own `threading.Lock`, while `LastReducer`, `EmaReducer`, and `WindowReducer` do not (relying on `Stream._lock`). Through `Stream`, `MeanReducer` and `DisplacementReducer` double-lock; standalone, `Last`/`Ema`/`Window` are not thread-safe.
  - `UmapProcessor` holds `self._lock`, while `OverallProcessor` has no lock (contradicting previous docs that claimed all processors are internally locked).
  - `DisplacementFilter` has no lock.
- **`DisplacementFilter` vs `DisplacementReducer` duplication**:
  - `filter.py` and `DisplacementReducer` (`reducer.py:117-178`) duplicate the exact same L2 `max`/`rms` keyframe logic. `DisplacementFilter` is never imported or wired into `Stream`, though `processor.py:59` comments `# Lowered from 200 due to upstream DisplacementFilter`.
- **Dependency mismatch (`pyproject.toml` vs `processor.py`)**:
  - `pyproject.toml` lists `cuml-cu12` and `parametric_umap`, but NOT `umap-learn`. `processor.py` imports `cuml` and falls back to `umap` (`umap-learn`), never `parametric_umap`.
- **Empty `__init__.py`**:
  - `positronixlab/stats/__init__.py` exports nothing; callers must import from internal submodules (`common`, `filter`, `reducer`, `processor`, `stream`).
- **Buffer aliasing risk in `OverallProcessor`**:
  - `OverallProcessor.process()` stores `self._latest.update(items)` without `.clone()`. Safe when fed from `Reducer.drain()` (which clones), but unsafe if called directly with aliased/recycled tensors.

---

## 4. Opportunities & Recommendations

1. **Lazy / optional UMAP import in `processor.py`** (highest priority):
   - Remove the unconditional `from cuml.manifold.umap import UMAP` at line 14, the duplicate imports at lines 28–33, and the self-import at line 42.
   - Defer UMAP backend resolution (`cuml` → `umap-learn`, or inject a custom factory) to `UmapProcessor.__init__` / `_create_mapper`, raising a clear `ImportError` only when `UmapProcessor` is actually instantiated without a backend. This immediately unblocks `Stream` and `OverallProcessor` on CPU-only environments.
2. **Unify `DisplacementFilter` and `DisplacementReducer`**:
   - Have `DisplacementReducer` delegate gating to `DisplacementFilter` (or drop `filter.py` if unused), add `method` validation + `.detach()` + empty-tensor handling in one place.
3. **Harden edge cases**:
   - `WindowReducer`: reset deque on shape/device change; reject `maxlen <= 0` in `__init__`.
   - `OverallProcessor`: guard empty `_latest` (return `[]` when `items` is empty or `_latest` is empty), add `self._lock`, clone stored tensors, and guard shape mismatches.
   - `UmapProcessor`: convert `t.detach().cpu().numpy()`, clamp `dynamic_n_neighbors` strictly `< num_neurons` (e.g. `min(dynamic_n_neighbors, max(2, num_neurons - 1))`), and reset/re-fit if `pts.shape[0]` changes after fit.
   - `Stream`: materialize/count `items` via `any_item` flag (fixing generators), consider tracking whether `reducer` actually marked keys dirty before incrementing `_count`, and guard `None` in `merge_streams` (`base.get(args) or {}`).
   - `_as_tensor`: drop `.reshape(())` when stacking list/tuple of tensors (or only reshape 0-D elements) so lists of N-D tensors stack cleanly.
4. **Clarify locking contract**:
   - Either make `Stream._lock` the sole synchronization boundary (removing redundant locks in `MeanReducer` / `DisplacementReducer`) or put a lock on `Reducer` base class consistently.
5. **Re-export public API in `positronixlab/stats/__init__.py`**:
   - Export `Stream`, `Reducer`, `REDUCERS`, concrete reducers, `DisplacementFilter`, `Processor`, `PROCESSORS`, `UmapProcessor`, `OverallProcessor`.

---

## 5. Docs / Tests / Examples Summary

- **Docs updated**:
  - `positronixlab/stats/docs/STATS_SYSTEM.md` — synced with current codebase (`DisplacementFilter`, `DisplacementReducer`, updated `UmapProcessor` API, `Stream(name, cfg, extractors)`, removal of `PcaProcessor`, move of `StatsPlugin` to `../AtlasCompose`, and known caveats).
  - `positronixlab/stats/docs/STATS_USAGE.md` — updated direct-usage examples, reducer/processor tables, threading notes, and troubleshooting to match actual code.
- **Tests created**:
  - `positronixlab/stats/tests/test_stats.py` — unit tests covering `common`, `filter`, `reducer`, `processor`, and `stream`, with `@pytest.mark.xfail(strict=True)` tests pinning every confirmed bug.
- **Example created**:
  - `positronixlab/stats/examples/example_stats.py` — concise, runnable CPU example exercising `_as_tensor`/`_to_np`, `DisplacementFilter`, all 5 reducers, `OverallProcessor`, `UmapProcessor` (with a lightweight CPU stub when `cuml` is unavailable), and `Stream` (`ingest`/`due`/`drain`/`merge_streams`).
