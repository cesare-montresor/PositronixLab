# `positronixlab/stats` — Solved Issues Archive

> **Archived:** 2026-11-27 (`v0.1.3` / `5740a0c` + working tree).
> **Active open issues:** See [stats_analysis.md](file:///home/cesare/Projects/PositronixLab/reports/stats_analysis.md).

---

## Resolved Bugs & Improvements

| ID | Summary | Resolution Evidence |
| :--- | :--- | :--- |
| **#1 (High)** | Unconditional top-level `from cuml.manifold.umap import UMAP` crashed CPU/non-cuML imports; duplicate imports & self-import in `processor.py` | Removed unconditional import, duplicates, and self-import (`processor.py:26-32`); falls back `cuml.manifold.UMAP` → `umap.UMAP` → `None`. `tests/stats/test_stats.py:398-410` passes without xfail. |
| **Exports** | Empty `stats/__init__.py` | Populated (`stats/__init__.py:1-25`) exporting all classes + `REDUCERS`/`PROCESSORS` (2026-11-27). |
| **UMAP `None` guard (2026-11-27)** | `UmapProcessor._create_mapper()` raised `TypeError: 'NoneType' object is not callable` when `UMAP is None` | Added explicit `ImportError` recommending `pip install 'positronixlab[umap]'` (`processor.py:90-94`). |
| **#2 (P1, 2026-11-27)** | `WindowReducer.drain()` crashed on tensor shape change & `maxlen=0` | Guarded `self.maxlen <= 0` and cleared deque on shape change (`reducer.py:109-116`); both xfails removed and passing. |
| **#3 (P1, 2026-11-27)** | `OverallProcessor.process({})` crashed when `_latest` was empty | Added `if not self._latest: return []` (`processor.py:219-221`); xfail removed and passing. |
| **#4 (P1, 2026-11-27)** | `UmapProcessor.process()` warm-start crashed when neuron count changed after fit | Falls back to `init="spectral"` when `prev_2d.shape[0] != num_neurons` (`processor.py:121-129`); xfail removed and passing. |
| **#5 (P1, 2026-11-27)** | `UmapProcessor._create_mapper()` clamped `n_neighbors` above `num_neurons - 1` on small layers | Clamped `dynamic_n_neighbors` to `min(..., num_neurons - 1)` when `num_neurons > 1` (`processor.py:89-93`); xfail removed and passing. |
| **#6 (P1, 2026-11-27)** | `_as_tensor()` crashed on list/tuple of non-scalar tensors | Only calls `.reshape(())` when all elements have `numel() == 1`, otherwise stacks directly (`common.py:16-20`); xfail removed and passing. |
| **#7 (P1, 2026-11-27)** | `DisplacementFilter` lacked `method` validation & both `DisplacementFilter`/`DisplacementReducer` crashed on empty tensors | Validated `method in ('max', 'rms')` in `DisplacementFilter.__init__` (`filter.py:11-12`) and guarded `value.numel() == 0` in both (`filter.py:24-25`, `reducer.py:151-152`); all 3 xfails removed and passing. |
| **#8 (P1, 2026-11-27)** | `Stream.ingest()` incremented `_count` on empty lazy generators & gated-out `DisplacementReducer` updates | Tracked `had_items` and required `had_items and self.reducer._dirty` before incrementing `_count` (`stream.py:52-57`); both xfails removed and passing. |
| **#9 (P1, 2026-11-27)** | `Stream.merge_streams()` crashed when `reducer_args`/`processor_args` was `None` | Handled `None` args in `merge_streams()` and `Stream.__init__` (`stream.py:28-32, 90-92`); xfail removed and passing. |
| **#10 (P1, 2026-11-27)** | `UmapProcessor.last_index()` returned `0` before any emission | Changed to `self._steps.get(layer, 0) - 1` (`processor.py:176`); xfail removed and passing. |
| **#11 (P1, 2026-11-27)** | `UmapProcessor.centroids_2d()` crashed on empty labels/points | Returned `np.empty((0, pts.shape[1]), dtype=np.float32)` when `ks.size == 0` (`processor.py:197-198`); xfail removed and passing. |
