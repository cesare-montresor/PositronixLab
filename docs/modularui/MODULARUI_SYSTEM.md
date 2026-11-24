# PositronixLab ModularUI System

Thread-safe Dash / Plotly / Cytoscape view layer for live monitoring of PositronixLab experiments. `ModularUI` acts purely as a bounded in-memory display store and web server: all heavy aggregation (EMA/mean reduction, PCA/UMAP 2D projection, cluster centroid projection, and persistence) is performed upstream by `StatsPlugin`.

---

## Architecture

```
  StatsPlugin / Producers (background threads)        Dash Web Server (Werkzeug threaded)
 ┌───────────────────────────────────────────┐       ┌──────────────────────────────────────┐
 │ set_var / set_config / set_model_graph    │       │  HomeLayout (poll_interval_ms=3000)  │
 │ set_observed_layers / set_space_stats     │       │   ├─ Top bar: Title, Epoch, Config   │
 │ ingest_stat / add_point                   ├───┐   │   ├─ Top row (3 panels):             │
 │ add_position_step / add_cluster           │   │   │   │   • GroupedChartsDisplay (train) │
 └───────────────────────────────────────────┘   │   │   │   • GroupedChartsDisplay (loss)  │
                                                 ▼   │   │   • VarsDisplay                  │
                                        ┌────────────────┐   ├─ Sidebar: ModelDAGDisplay    │
                                        │   ModularUI    │◄──┤   (Cytoscape -> dcc.Store)   │
                                        │   + UILocks    │   └─ Main: LayerBoardDisplay     │
                                        │ (bounded deques│       (per-layer 2D scatter +    │
                                        │  & snapshots)  │        centroids + slider +      │
                                        └────────────────┘        SpaceLossRow)             │
                                                     └──────────────────────────────────────┘
```

### Module Layout

| File | Contents |
|---|---|
| `positronixlab/modularui/modularui.py` | `UILocks` (frozen dataclass of `threading.Lock`s), `_to_numpy`, and `ModularUI` (store + Werkzeug/Dash server). |
| `positronixlab/modularui/layouts.py` | `HomeLayout` (5-panel dashboard page layout & callback wiring) and `_BoundWidget`. |
| `positronixlab/modularui/docks.py` | `Panel` (card with collapse/maximize header buttons) and `DashboardLayout` (static 3-column CSS Grid). |
| `positronixlab/modularui/components.py` | `Widget` ABC, `Container`, `Title`, `ExpandableSection`, `VarsDisplay`, `ConfigOverlay`, `ModelDAGDisplay`. |
| `positronixlab/modularui/plots.py` | `_line_figure`, `MetricRow`, `SpaceLossRow`, `loss_then_accuracy_first`, `GroupedChartsDisplay`, `LayerBoardDisplay`. |
| `positronixlab/modularui/settings.py` | `THEME_NAME`, `BOOTSTRAP_THEME`, `DBC_HELPER_CSS`, Plotly template constants, `SPACELOSS_FAMILY`, `GLOBAL_TRACE_NAME`. |
| `positronixlab/modularui/assets/css/` | `main.css` (grid responsive reflow, card/graph styles) and `toggle-switches.css` (CSS-only toggle checkboxes). |

---

## Concurrency & Locking Model (`UILocks`)

`UILocks` is a `@dataclass(frozen=True)` providing fine-grained `threading.Lock` instances so producer threads and Dash HTTP worker threads do not contend on unrelated data:

| Lock | Protected Fields on `ModularUI` |
|---|---|
| `locks.vars` | `dynamic_vars`, `dynamic_vars_version` |
| `locks.config` | `trial_config` |
| `locks.model_graph` | `model_graph`, `model_graph_version` |
| `locks.stats` | `stats` (per-layer activation counts `[c_on, c_off]`) |
| `locks.observed_layers` | `observed_layers` |
| `locks.metrics` | `series` (`family -> layer_id -> label -> deque[float]`) |
| `locks.positions` | `positions_plot`, `_position_last_index`, `position_progress`, `cluster_history` |
| `locks.spaceloss` | *(Unused legacy field; spaceloss is stored in `series` under `locks.metrics`.)* |

---

## Data Model

### 1. Unified Scalar Series (`self.series`)
All scalar metrics share a three-level mapping:
```
series[family][layer_id][label] -> deque(maxlen=metrics_maxlen)
```
- Global (non-per-layer) metrics use `layer_id = "global"`.
- Per-layer space losses use `family = SPACELOSS_FAMILY` (`"spaceloss"`), with `"global"` representing the aggregate mean across layers (mapped to `GLOBAL_TRACE_NAME = "mean (all layers)"` by `get_family_groups()`).
- `get_training_groups()` merges `self.series` (for global-only non-spaceloss families) with the `metrics_history` dynamic variable fallback (`{family: {label: [[t, value], ...]}}`), keeping whichever has more points.

### 2. Position Timeline & Cluster Synchronization
- `positions_plot[layer_id]` holds a rolling `deque(maxlen=positions_maxlen)` of `(N, 2)` NumPy arrays.
- `_position_last_index[layer_id]` tracks the absolute timeline index of the most recent step. The first retained step's absolute index is computed as:
  `offset = last_index - len(window) + 1`.
- `cluster_history[layer_id]` holds a rolling `deque(maxlen=cluster_maxlen)` of snapshots:
  `{"position_index", "labels", "best_k", "best_score", "centroids"}`.
- `get_cluster_for_index(layer_id, relative_index)` converts `relative_index` to `absolute = relative_index + offset` and uses `bisect_right` to return the most recent snapshot at or before `absolute`.

### 3. Model DAG & Dynamic Vars
- `model_graph`: `{"nodes": {node_id: {"class_name": ...}}, "edges": [{"source": ..., "target": ...}]}` with a version counter `model_graph_version`.
- `dynamic_vars`: arbitrary key/value metadata (e.g. `"epoch": (current, total)`, `"metrics_history"`), each with a per-key monotonic version in `dynamic_vars_version`.
- `trial_config`: flat dict (or object with `.to_dict()`) rendered in `ConfigOverlay`.

---

## Widget & Callback Architecture

All widgets follow the `Widget` ABC (`render(...)` + `__call__`) except `ConfigOverlay`, which exposes `render_button()`, `render_overlay()`, and `render_selected()`.

`HomeLayout` registers **15 Dash callbacks** across 5 dashboard panels (`metrics-panel`, `spaceloss-panel`, `vars-panel`, `dag-panel`, `layer-board-panel`):

| Callback Owner | Trigger(s) | Output(s) | Mechanism |
|---|---|---|---|
| `HomeLayout` | `btn-collapse`, `btn-expand`, `btn-maximize` (`ALL`) | `panel-state.data` | Updates panel visibility/maximize flags |
| `HomeLayout` | `panel-state.data` | `grid-root.children`, `collapsed-tabs-bar.children` | Re-renders `DashboardLayout` and collapsed tabs |
| `HomeLayout` | `positronixlab-poll` | `epoch-container.children` | Formats `ui.get_var("epoch")` |
| `HomeLayout` | `positronixlab-poll` | `vars-container.children` | Re-renders `VarsDisplay` |
| `HomeLayout` | `positronixlab-poll`, `config-overlay-picker.value` | `config-selected-container.children` | Renders pinned config keys |
| `ConfigOverlay` | `config-overlay-button.n_clicks` | `config-overlay.is_open` | Toggles modal |
| `ConfigOverlay` | `config-overlay-picker-poll` (5s) | `config-overlay-picker.{options,value}` | Refreshes config checklist |
| `ModelDAGDisplay` | `positronixlab-dag-poll` (5s), `tapNodeData` | `cyto.elements`, `toggle_id.data` | Syncs DAG nodes & independent node toggle |
| `GroupedChartsDisplay` (×2) | `positronixlab-poll` | `container.children`, `keys.data` | Rebuilds charts when label/trace signature changes |
| `GroupedChartsDisplay` (×2) | `positronixlab-poll` | `graph[MATCH].figure` | Incremental `Patch()` of trace `y` values |
| `LayerBoardDisplay` | `dag_toggle_id.data`, `positronixlab-poll` | `layerboard.children`, `ids.data` | Rebuilds layer cards when visible layers/labels change |
| `LayerBoardDisplay` | `positronixlab-poll`, `live-btn[MATCH]`, `slider[MATCH]` | `graph[MATCH].figure`, `slider[MATCH].{max,value}` | Incremental `Patch()` of 2D points, marker sizes, clusters, centroids |
| `SpaceLossRow` (`MetricRow`) | `positronixlab-poll` | `spaceloss-graph[MATCH].figure` | Incremental `Patch()` per `layer_id\|label` |

---

## Styling & Theming

- Controlled centrally in `settings.py` via `THEME_NAME` (default `"cyborg"`), resolving `dbc.themes.CYBORG` and calling `dash_bootstrap_templates.load_figure_template(THEME_NAME)` on `plots.py` import.
- Figures use transparent backgrounds (`PLOTLY_PAPER_BGCOLOR = PLOTLY_PLOT_BGCOLOR = "rgba(0,0,0,0)"`) so cards blend with the active dark theme.
- Responsive single-column reflow (`<= 900px`) and card/toggle styling live in `assets/css/main.css` and `assets/css/toggle-switches.css`.
