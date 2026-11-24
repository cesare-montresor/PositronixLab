# `positronixlab.modularui` — Code Review & Analysis

## 1. Overview & Refactor Status

`positronixlab/modularui` is the thread-safe Dash / Plotly / Cytoscape view layer for live experiment monitoring. It sits downstream of `StatsPlugin` (which aggregates, projects via PCA/UMAP, and syncs clusters) and maintains bounded in-memory display windows (`deque(maxlen=...)`) served to Dash polling callbacks.

### Component Inventory & Status

| File | Responsibility | Status |
|---|---|---|
| `__init__.py` | Package public exports | **Incomplete** — completely empty (0 bytes); callers must import from internal submodules. |
| `modularui.py` | `UILocks` (fine-grained lock dataclass) & `ModularUI` (thread-safe view store + Werkzeug/Dash server lifecycle) | **Refactored to unified `self.series` store**, but retains dead locks (`UILocks.spaceloss`), unused flags (`reloader`), subtle locking/snapshot bugs, and broken `external_stylesheets` wiring. |
| `layouts.py` | `HomeLayout` (5-panel dashboard page layout + 15 Dash callbacks via delegation) and `_BoundWidget` helper | **Partially refactored** — migrated from old flexbox layout to `DashboardLayout`, but panel collapse/maximize rebuilds the entire `grid-root` DOM and mutates `default_state`. |
| `docks.py` | `Panel` (collapsible/maximizable card) and `DashboardLayout` (static 3-column CSS Grid) | **Mostly clean**, minor stale docstring (`assets/dashboard.css`), unused `all_panels`, silent truncation if `len(top_panels) > 3`. |
| `components.py` | `Widget` ABC, `Container`, `Title`, `ExpandableSection`, `VarsDisplay`, `ConfigOverlay`, `ModelDAGDisplay` (Cytoscape) | **Refactored to Cytoscape DAG**, but carries multi-client instance state (`_last_graph_version`), ignored `checked` param, and leftover `width: 260px` flexbox styling. |
| `plots.py` | `_line_figure`, `MetricRow`, `SpaceLossRow`, `GroupedChartsDisplay`, `LayerBoardDisplay` (2D positions + centroids + time slider + space loss) | **Active / WIP** — uses Dash `Patch()` for incremental updates, but has a trace-index race condition, stale centroid/color persistence when scrubbing backward, broken empty-filter logic, and non-atomic position/cluster reads. |
| `settings.py` | Theme (`cyborg`), Plotly template, CDN helper CSS, metric family constants | **Working**, contains deprecated unused constants (`PLOTS_SPACELOSS_KEYS`, `PLOTS_ROW_LABEL_STYLE`). |
| `assets/css/` | `main.css` (grid reflow, card/graph rules) & `toggle-switches.css` | **Partially mismatched** with `HomeLayout.css` paths and `plots.py` chart heights. |
| `examples/example_ui.py` | Replay script using `examples/dump/data.pickle` | **Broken** — still calls 4 removed `ModularUI` methods from before the `self.series` / `add_position_step` refactor. |
| `docs/` & `tests/` | Documentation & unit tests | **Previously empty** (now populated as part of this review). |

---

## 2. Bugs & Defects

### Critical / High Impact

1. **Broken CSS asset paths in `external_stylesheets` (404 on every page load)**
   - **Location**: `positronixlab/modularui/layouts.py:34`, `positronixlab/modularui/modularui.py:121-126`
   - **Root cause**: `HomeLayout.__init__` sets `self.css = [f"themes/{theme}.css", "themes/toggle-switches.css"]`, which `ModularUI.__init__` appends to `Dash(..., external_stylesheets=...)`. Relative URLs in `external_stylesheets` resolve to `/themes/default.css` and `/themes/toggle-switches.css`, neither of which exists (Dash serves `positronixlab/modularui/assets/css/main.css` and `assets/css/toggle-switches.css` automatically under `/assets/`).
   - **Impact**: Every browser load emits two 404 requests; `themes/default.css` (which only exists in `examples/assets/themes/default.css`) is never loaded.

2. **Deselecting all nodes in the Model DAG shows *all* layers instead of *none***
   - **Location**: `positronixlab/modularui/plots.py:237-242`
   - **Root cause**: `LayerBoardDisplay._filtered_layer_ids(visible_layers)` checks `if not visible_layers: return all_ids`. Both `None` (unfiltered) and `[]` (user toggled every node off in `ModelDAGDisplay`) are falsy.
   - **Impact**: Clicking the last active node in the Cytoscape DAG unexpectedly causes every layer card to reappear.

3. **`_update_panel_state` mutates `self.default_state` in place**
   - **Location**: `positronixlab/modularui/layouts.py:117-130`
   - **Root cause**: When `current_state` is falsy, `current_state = self.default_state` binds directly to the shared dictionary without deep-copying, then mutates `current_state[pid]["collapsed"]` / `["maximized"]`.
   - **Impact**: Permanently corrupts `HomeLayout.default_state` for subsequent renders/clients.

4. **Lock released before reading/iterating `self.observed_layers`**
   - **Location**: `positronixlab/modularui/modularui.py:413-428`
   - **Root cause**: `get_observed_layer_ids()` and `get_observed_layer_info()` grab `data = self.observed_layers` inside `with self.locks.observed_layers:`, exit the lock, and then iterate `list(data.keys())` or call `data.get(layer_id, {})`.
   - **Impact**: Concurrent mutation of `observed_layers` by a producer thread can raise `RuntimeError: dictionary changed size during iteration` or return inconsistent state.

5. **`GroupedChartsDisplay._patch` vs `_refresh` trace-index race condition**
   - **Location**: `positronixlab/modularui/plots.py:176-201`
   - **Root cause**: Both `_patch` and `_refresh` subscribe to `Input(POLL_ID, "n_intervals")`. When a new trace (e.g. `val` alongside `train`, or a new layer in `spaceloss`) appears before `_refresh` rebuilds the graph, `_patch` iterates `enumerate(self._trace_names(traces))` and writes `patched_fig["data"][i]["y"]` for `i >= len(existing_traces)` (or overwrites the wrong trace if the new name sorts earlier).
   - **Impact**: Client-side Plotly/Dash `Patch` error or temporarily mislabeled series whenever a new trace arrives dynamically.

6. **Stale cluster centroids & marker colors persist when scrubbing backward**
   - **Location**: `positronixlab/modularui/plots.py:400-422`
   - **Root cause**: `_patch_layer_graph` only updates `patched_fig["data"][1]` (centroids) and `patched_fig["data"][0]["marker"]["color"]` inside `if cluster_snapshot:`. There is no `else` branch to clear them when `get_cluster_for_index(...)` returns `None`.
   - **Impact**: Scrubbing the slider from a clustered step back to an earlier unclustered step leaves the old centroids and cluster colors on screen.

7. **Panel collapse/expand/maximize destroys and recreates the entire dashboard DOM**
   - **Location**: `positronixlab/modularui/layouts.py:132-154`
   - **Root cause**: `_render_layout` returns `[self.dashboard_layout.render(state)]` into `grid-root.children` whenever `panel-state` changes.
   - **Impact**: Toggling any single panel resets all Plotly zoom/pan state (and blanks charts until the next poll), resets `ModelDAGDisplay`'s `dcc.Store` selection to `[]`, resets all `LayerBoardDisplay` sliders to `0`, and collapses all open `<details>` in `VarsDisplay`.

8. **`examples/example_ui.py` is broken against the current `ModularUI` API**
   - **Location**: `positronixlab/modularui/examples/example_ui.py:32, 47, 56, 58`
   - **Root cause**: Calls four removed methods: `ui.update_vars()`, `ui.add_positions()`, `ui.add_metric_spaceloss()`, `ui.add_metric_training()`.
   - **Impact**: Running `example_ui.py` immediately raises `AttributeError`.

### Medium Impact

9. **`_render_vars` never clears stale variables after `ui.reset()`**
   - **Location**: `positronixlab/modularui/layouts.py:161-166`
   - **Root cause**: `if not snapshot: return dash.no_update` skips rendering when `dynamic_vars` becomes empty.
   - **Impact**: After `ui.reset()`, old variables remain visible in the Variables panel until a new variable is set.

10. **Non-atomic reads across 4 lock acquisitions in `LayerBoardDisplay._patch_layer_graph`**
    - **Location**: `positronixlab/modularui/plots.py:355, 379, 400, 423`
    - **Root cause**: Calls `get_position_steps`, `get_position_at`, `get_cluster_for_index`, and `get_position_offset` in four separate `with self.locks.positions:` blocks.
    - **Impact**: When `positions_plot` is rolling at `positions_maxlen`, concurrent appends can shift the window between calls, pairing `points` from step `k` with `cluster_snapshot` or `offset` from step `k+1`.

11. **`_position_offset_locked` assumes strictly contiguous +1 `index` increments**
    - **Location**: `positronixlab/modularui/modularui.py:296-297, 441-445`
    - **Root cause**: Computes `offset = last_index - len(window) + 1`. If `add_position_step` is called with non-contiguous indices (e.g., `index=0`, then `index=5`), `offset` becomes `4`, mapping relative index `0` to absolute index `4` instead of `0`.
    - **Impact**: `get_cluster_for_index` and the step title display misalign whenever indices have gaps.

12. **`LayerBoardDisplay._patch_layer_graph` crashes on `None` slider value or non-2D `stats`/`points`**
    - **Location**: `positronixlab/modularui/plots.py:368-369, 386-398, 414-415`
    - **Root cause**: When `is_slider` is True and `current_slider_val` is `None`, `get_position_at(layer_id, None)` raises `TypeError` (`None < 0`). Similarly, `arr[:, 0]` / `arr[:, 1]` on `stats` assumes a 2D array with `>= 2` columns without validation, and `points[:, 0]` assumes 2D `(n, >=2)`.

13. **`HomeLayout._format_epoch` crashes on non-2-tuple values and hides epoch `0`**
    - **Location**: `positronixlab/modularui/layouts.py:98-104`
    - **Root cause**: `if not epoch_value:` treats scalar `0` as missing, and `epoch, epochs = epoch_value` raises `TypeError`/`ValueError` for scalar ints, strings, or tuples of length != 2.

14. **`add_point` crashes on `None` / non-numeric scalar; `add_spaceloss_point` fails on CUDA tensor dicts**
    - **Location**: `positronixlab/modularui/modularui.py:257-266, 278-282`
    - **Root cause**: `_to_numpy(None)` produces a 0-d object array with `size == 1`, so `float(arr.reshape(-1)[0])` raises `TypeError`. In `add_spaceloss_point`, converting a `dict` of CUDA `torch.Tensor`s via `list(values.values())` passes a Python list of CUDA tensors to `np.asarray`, which raises `TypeError` because `_to_numpy` only unwraps a top-level `torch.Tensor`.

15. **`ModelDAGDisplay._last_graph_version` is instance-level mutable state**
    - **Location**: `positronixlab/modularui/components.py:161, 273-277`
    - **Root cause**: Stored on `self` rather than a client-side `dcc.Store`, so multiple browser tabs or a `reset()` (which sets `model_graph_version = 0` at `modularui.py:179`) can desync graph updates. Also `_sync` mutates `current_data` in place (`components.py:259-262`).

16. **`VarsDisplay.render` crashes on nested dicts with mixed key types**
    - **Location**: `positronixlab/modularui/components.py:70`
    - **Root cause**: `sorted(value.items())` raises `TypeError` if a dict has both `int` and `str` keys.

### Low Impact / Cleanup

17. **`ModularUI.stop()` does not close the server socket or reset `server`/`server_thread`** (`modularui.py:155-160`).
18. **`_update_panel_state` does not check `n_clicks` truthiness on `ALL` pattern-matching triggers** (`layouts.py:115-130`) and allows `collapsed=True` + `maximized=True` simultaneously.
19. **`DashboardLayout` silently drops `top_panels[3:]`** via `zip(self.top_panels, self.TOP_AREAS)` (`docks.py:95, 113`).

---

## 3. Inconsistencies & Design Issues

- **Thread safety & defensive copying**:
  - `get_config()` and `get_all_vars_snapshot()` return shallow copies (`dict(...)`), whereas `get_model_graph()`, `get_layer_stats()`, `get_observed_layer_info()`, `get_position_at()`, and `get_cluster_for_index()` return direct references to internal mutable objects.
  - `UILocks.spaceloss` (`modularui.py:39`) is declared and documented but never acquired anywhere (`self.series` is guarded exclusively by `locks.metrics`).
- **Dash callback & widget architecture**:
  - `ConfigOverlay` does not inherit from `Widget` (unlike all other renderers) and `_refresh_picker` unconditionally re-emits `(options, value)` every 5s even when `config` has not changed (`components.py:137-146`).
  - `ModelDAGDisplay.render(self, checked=None)` accepts an unused `checked` argument, annotates its return type as `html.Div` while returning `dbc.Card`, and sets inline `width: 260px` (`components.py:238`) which clashes with `DashboardLayout`'s `1fr` sidebar column.
  - `_BoundWidget` in `layouts.py:18-28` is instantiated with zero args for `training_metrics` and `overall_spaceloss` (`layouts.py:59-60`), adding an unnecessary wrapper layer.
  - `ModularUI.get_pca_progress()` / `set_position_progress()` (`modularui.py:248, 466`) are completely unused by `LayerBoardDisplay`.
- **CSS & Plotly sizing conflicts**:
  - `assets/css/main.css:24-30` enforces `.card-body .dash-graph { height: 100% !important; min-height: 260px; }`, overriding `GroupedChartsDisplay`'s `height=110` / `height=120` and conflicting with `LayerBoardDisplay`'s `height=450` (figure) vs `480px` (`dcc.Graph`).
  - `docks.py:93` references `assets/dashboard.css` (actual file: `assets/css/main.css`).
  - `settings.py:15` loads `DBC_HELPER_CSS` from `cdn.jsdelivr.net`, which fails in offline environments.
- **Stale imports, dead code & missing exports**:
  - `positronixlab/modularui/__init__.py` is empty (no public API exports).
  - Unused imports: `dash` in `components.py:5`, `ALL` in `plots.py:7`, `GLOBAL_TRACE_NAME` in `plots.py:11`, `sys` in `examples/example_ui.py:2`.
  - Unused loggers: `components.py:13`, `layouts.py:15`.
  - Unused attributes/constants: `ModularUI.reloader` (`modularui.py:70`), `DashboardLayout.all_panels` (`docks.py:101`), `PLOTS_SPACELOSS_KEYS` & `PLOTS_ROW_LABEL_STYLE` (`settings.py:34-35`).
  - `get_training_groups()` (`modularui.py:375-380`) silently falls back to `metrics_history` if `len(history) > metrics_maxlen`, and skips any family that mixes `"global"` and per-layer keys.

---

## 4. Opportunities & Recommendations

1. **Fix CSS & layout state preservation (highest UX impact)**:
   - Remove the broken `self.css = ["themes/..."]` from `HomeLayout` / `external_stylesheets` and let Dash auto-serve `positronixlab/modularui/assets/css/`.
   - Toggle panel collapse/maximize via CSS classes or per-panel `style` callbacks (or clientside callbacks) instead of rebuilding `grid-root.children`, preserving Plotly zoom, DAG selections, and slider positions.
   - Deep-copy `self.default_state` in `_update_panel_state` and guard `ALL` triggers against `None`/0 `n_clicks`.
2. **Harden thread safety & atomic snapshots**:
   - Perform dict copies/lookups *inside* `with self.locks.observed_layers:` in `get_observed_layer_ids` / `get_observed_layer_info`.
   - Add an atomic `get_position_frame(layer_id, target_index)` helper on `ModularUI` that returns `(points, offset, cluster_snapshot, n_steps)` under a single `locks.positions` acquisition, and store per-step `(abs_index, points)` tuples so non-contiguous indices work accurately.
   - Remove dead `UILocks.spaceloss`.
3. **Fix Plotly/Cytoscape edge cases**:
   - In `LayerBoardDisplay._filtered_layer_ids`, check `if visible_layers is None:` instead of `if not visible_layers:`.
   - Clear `patched_fig["data"][1]` (centroids) and reset marker color when `cluster_snapshot is None`.
   - Store `graph_version` in a client-side `dcc.Store` in `ModelDAGDisplay`, and guard `GroupedChartsDisplay._patch` against trace-count mismatches (e.g., pass `State(f"{self.css_id}-keys", "data")` or `State(..., "figure")` trace count).
   - Surface `ui.get_pca_progress(layer_id)` in `LayerBoardDisplay` when `n_steps == 0`.
4. **Expose clean public API & fix `example_ui.py`**:
   - Re-export `ModularUI`, `UILocks`, `HomeLayout`, `DashboardLayout`, `Panel`, `Widget`, etc. in `positronixlab/modularui/__init__.py`.
   - Update `examples/example_ui.py` to use `set_var`, `add_position_step`, `add_spaceloss_point`, and `add_training_point`.

---

## 5. Docs / Tests / Examples Summary

- **Documentation created**:
  - `positronixlab/modularui/docs/MODULARUI_SYSTEM.md` — architecture, data/locking model, widget hierarchy, Dash callback map, and CSS/theming notes.
  - `positronixlab/modularui/docs/MODULARUI_USAGE.md` — quick start, data ingestion API, server lifecycle, layout customization, and known caveats.
- **Unit tests created**:
  - `positronixlab/modularui/tests/test_modularui.py` — headless `pytest` suite covering `UILocks`, `ModularUI` data ingestion/accessors/rolling windows/cluster binary search/thread safety, `Widget`/`Panel`/`DashboardLayout`/`HomeLayout`/`ModelDAGDisplay`/`GroupedChartsDisplay`/`LayerBoardDisplay` rendering, Dash callback registration, and `@pytest.mark.xfail(strict=True)` regression tests documenting confirmed defects.
- **Headless example created**:
  - `positronixlab/modularui/examples/example_modularui_headless.py` — populates a `ModularUI` instance with synthetic training metrics, per-layer space loss, 2D PCA positions, clusters, model DAG, config, and dynamic vars, renders all layout/widget trees and Plotly figures headlessly, and prints a structured summary without starting a live server.
