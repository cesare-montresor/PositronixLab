# `positronixlab/modularui` — Solved Issues Archive

> **Archived:** 2026-11-27.
> **Active open issues:** See [open/modularui_analysis.md](file:///home/cesare/Projects/PositronixLab/reports/open/modularui_analysis.md).

---

## Resolved Bugs & Improvements

| ID | Summary | Resolution Evidence |
| :--- | :--- | :--- |
| **Exports** | Empty `modularui/__init__.py` | Populated with PEP 562 lazy exports (`modularui/__init__.py`). |
| **#1 (P1, 2026-11-27)** | Broken relative `themes/*.css` paths in `external_stylesheets` | Cleared `HomeLayout.css = []` (`layouts.py:42`) so Dash serves `assets/` cleanly without 404s; xfail removed and passing. |
| **#2 (P1, 2026-11-27)** | `LayerBoardDisplay._filtered_layer_ids([])` returned all layers instead of `[]` | Changed `if not visible_layers:` to `if visible_layers is None:` (`plots.py:246`); xfail removed and passing. |
| **#11 (P1, 2026-11-27)** | `_position_offset_locked` assumed contiguous `+1` indices, misaligning cluster lookup on gaps | Tracked per-step absolute indices in `_position_indices` deque and used exact lookup in `get_cluster_for_index` (`modularui.py`); xfail removed and passing. |
| **#13 (P1, 2026-11-27)** | `HomeLayout._format_epoch()` crashed on scalar `int` and treated `0` as missing | Handled `None`, 2-tuples, and scalar values cleanly (`layouts.py:106-113`); xfail removed and passing. |
| **#14 (P1, 2026-11-27)** | `ModularUI.add_point()` raised `TypeError` on `None` / non-numeric scalar | Guarded `value is None` and caught `(TypeError, ValueError)` on float conversion (`modularui.py:264-275`); xfail removed and passing. |
| **#16 (P1, 2026-11-27)** | `VarsDisplay.render()` crashed on nested dicts with mixed key types | Sorted with `key=lambda kv: str(kv[0])` (`components.py:67-72`); xfail removed and passing. |
| **Defensive copy (P1, 2026-11-27)** | `ModularUI.get_model_graph()` returned internal mutable dict reference | Returned `copy.deepcopy(self.model_graph)` (`modularui.py:239`); xfail removed and passing. |
| **#19 (P1, 2026-11-27)** | `DashboardLayout.render()` silently dropped `top_panels[3:]` | Dynamically expanded grid columns/areas to render all `top_panels` (`docks.py:113-143`); xfail removed and passing. |
