# `positronixlab/modularspace` — Solved Issues Archive

> **Archived:** 2026-11-27.
> **Active open issues:** See [open/modularspace_analysis.md](file:///home/cesare/Projects/PositronixLab/reports/open/modularspace_analysis.md).

---

## Resolved Bugs & Cleanup

| ID | Summary | Resolution Evidence |
| :--- | :--- | :--- |
| **B1 (P1, 2026-11-27)** | `ModularSpace.get_space()` failed when `stats_enabled=False` | Guarded `self.stats_enabled and stats is None` (`modularspace.py:61`); xfail removed and passing. |
| **B3 (P1, 2026-11-27)** | `LayerSpace.process_scores()` called `loss.backward()` on detached zero tensor | Guarded optimizer step with `if loss.requires_grad:` (`layerspace.py:337-340`); xfail removed and passing. |
| **B4 (P1, 2026-11-27)** | `ClusterTracker._match_and_project()` crashed on `torch.argmax` when `k_old == 0` | Added `if k_old == 0: no_lineage.append(j); continue` (`custertracker.py:173-175`); xfail removed and passing. |
| **B5 (P1, 2026-11-27)** | `ClusterTracker._split_positions()` returned `parent_outer` unchanged when `m == 1` | Offsets single child along seeded unit direction (`custertracker.py:288-293`); xfail removed and passing. |
| **Cleanup** | Stale `layerspace_old.py` & `custertracking.example.py`; runtime `assert` (`C7`) | Stale files removed; `assert old is not None` replaced with `RuntimeError` (`custertracker.py:160-161`). |
