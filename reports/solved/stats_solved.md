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
