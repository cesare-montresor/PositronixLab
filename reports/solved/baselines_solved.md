# `positronixlab/baselines` — Solved Issues Archive

> **Archived:** 2026-11-27 (`v0.1.3` / `5740a0c` + working tree).
> **Active open issues:** See [baselines_analysis.md](file:///home/cesare/Projects/PositronixLab/reports/baselines_analysis.md).

---

## Resolved Minor Issues (2026-11-27)

- **Parameterless model `StopIteration` in NeuroSurgeon adapter & discoverer:** Replaced `next(...parameters()).device` with `next(...parameters(), None)` + CPU fallback in `baselines/neurosurgeon/discoverer.py:76` and `baselines/neurosurgeon/adapter.py:159`.
- **CVR renderer availability check:** `is_cvr_renderer_available()` (`baselines/lepori/_cvr_gen/__init__.py:20-28`) now checks both `cv2` and `PIL.Image`.
- **Documentation drift:** Updated `docs/baselines/BASELINES_SYSTEM.md:24, 119` (`6 datasets` for `csordas/`, `8 datasets` for `lepori/`, and added `"pillow"` to the `baselines` extra).
