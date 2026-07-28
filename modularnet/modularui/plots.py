from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
from dash import ALL, MATCH, Dash, Input, Output, State, dcc, html

from modularnet.modularui.components import Widget

if TYPE_CHECKING:
    from modularnet.modularui.modularui import ModularUI

# themes some day ?
PLOTLY_TEMPLATE = "plotly_dark"
PLOTLY_PAPER_BGCOLOR = "rgba(0,0,0,0)"
PLOTLY_PLOT_BGCOLOR = "rgba(0,0,0,0)"

# -------------------------------------------------------------------------
# Reusable Plot Abstractions
# -------------------------------------------------------------------------

class LineChart(Widget):
    """Reusable widget for a single chart with multiple lines (Plots 1 & 4)."""
    def __init__(self, title: str, css_id: str):
        self.title = title
        self.css_id = css_id

    def render(self, metrics: Optional[Dict[str, List[float]]] = None):
        metrics = metrics or {}
        fig = go.Figure()
        for metric_name, values in metrics.items():
            fig.add_trace(go.Scatter(y=values, mode="lines", name=metric_name))

        fig.update_layout(
            title=self.title,
            margin=dict(l=20, r=20, t=40, b=20),
            height=300,

            template=PLOTLY_TEMPLATE,
            paper_bgcolor=PLOTLY_PAPER_BGCOLOR,
            plot_bgcolor=PLOTLY_PLOT_BGCOLOR
        )
        return dcc.Graph(id=self.css_id, figure=fig)


class PerLayerLineChart(Widget):
    """Reusable widget for generating one line chart per layer (Plot 3)."""
    def __init__(self, title_prefix: str, css_id: str):
        self.title_prefix = title_prefix
        self.css_id = css_id

    def render(self, layer_metrics: Optional[Dict[str, Dict[str, List[float]]]] = None):
        layer_metrics = layer_metrics or {}
        graphs = []
        for layer_id, metrics in sorted(layer_metrics.items()):
            fig = go.Figure()
            for metric_name, values in metrics.items():
                fig.add_trace(go.Scatter(y=values, mode="lines", name=metric_name))

            fig.update_layout(
                title=f"{self.title_prefix}: {layer_id}",
                margin=dict(l=20, r=20, t=40, b=20),
                height=250,

                template=PLOTLY_TEMPLATE,
                paper_bgcolor=PLOTLY_PAPER_BGCOLOR,
                plot_bgcolor=PLOTLY_PLOT_BGCOLOR
            )
            graphs.append(dcc.Graph(figure=fig))

        return html.Div(graphs, id=self.css_id)


# -------------------------------------------------------------------------
# Specific Plot Implementations
# -------------------------------------------------------------------------

# Plot 1: Overall training metrics
class TrainingMetricsDisplay(LineChart):
    def __init__(self, css_id: str = "modularnet-training-metrics"):
        super().__init__(title="Overall Training Metrics", css_id=css_id)


# Plot 4: Overall space loss
class OverallSpaceLossDisplay(LineChart):
    def __init__(self, css_id: str = "modularnet-overall-spaceloss"):
        super().__init__(title="Overall Space Loss", css_id=css_id)


# Plot 3: Space loss per layer
class LayerSpaceLossDisplay(PerLayerLineChart):
    def __init__(self, css_id: str = "modularnet-layer-spaceloss"):
        super().__init__(title_prefix="Space Loss", css_id=css_id)


# Plot 2: Positions per layer, evolving in time
class PositionsDisplay(Widget):
    """One box per observed layer.

    Each box shows:
      - a header built from modularui's observed_layers metadata for that layer
      - while PCA hasn't been fit yet: a "collecting samples" progress readout
        (current samples vs. required samples)
      - once PCA is ready: a scatter plot of that layer's 2D embedding
        positions at a chosen time step, with a slider underneath to scrub
        back and forth through history

    Dot sizing/labels/color:
      - each point is labeled at its center with its embedding index
      - point size comes from that layer's space_stats on/off ratio:
            ratio = count_on / (count_on + count_off)
            scale = 1.0 at ratio 0.0, 2.0 at ratio 0.5, 3.0 at ratio 1.0
      - points are colored by cluster assignment (ClusterPlugin's "clusters"
        topic, resolved via ModularUI.get_cluster_for_index so scrubbing the
        slider shows the clustering that was valid at that position step,
        since clustering updates at a lower rate than positions)
      - cluster centroids (PCA-transformed, same sync) are overlaid as a
        second trace of "Cn" markers
    """

    SIZE_MIN_SCALE = 1.0
    SIZE_MAX_SCALE = 3.0
    BASE_MARKER_SIZE = 14
    POLL_INTERVAL_MS = 1000

    def __init__(self, ui: "ModularUI", css_id: str = "modularnet-positions"):
        self.ui = ui
        self.css_id = css_id

    # ---------------------------------------------------------------
    # Layout
    # ---------------------------------------------------------------
    def render(self) -> html.Div:
        layer_ids = self._known_layer_ids()
        boxes = [self._render_layer_box(layer_id, forced_index=None) for layer_id in layer_ids]
        return html.Div(
            boxes + [dcc.Interval(id=f"{self.css_id}-poll", interval=self.POLL_INTERVAL_MS)],
            id=self.css_id,
            style={"display": "flex", "flexWrap": "wrap", "gap": "16px"},
        )

    def _known_layer_ids(self) -> List[str]:
        observed = set(self.ui.get_observed_layer_ids())
        with self.ui.locks.positions:
            seen = set(self.ui.positions_plot.keys()) | set(self.ui.pca_history.keys())
        return sorted(observed | seen)

    def _render_layer_box(self, layer_id: str, forced_index: Optional[int]) -> html.Div:
        info = self.ui.get_observed_layer_info(layer_id)
        header = self._render_header(layer_id, info)

        progress = self.ui.get_pca_progress(layer_id)
        if progress is not None:
            body = self._render_loading(progress)
        else:
            n_steps = self.ui.get_position_steps(layer_id)
            last_index = max(n_steps - 1, 0)
            # Follow the live edge unless the user has scrubbed backwards.
            if forced_index is None or forced_index >= last_index:
                index = last_index
            else:
                index = forced_index

            body = html.Div([
                dcc.Graph(
                    id={"type": f"{self.css_id}-graph", "index": layer_id},
                    figure=self._build_figure(layer_id, index),
                ),
                dcc.Slider(
                    id={"type": f"{self.css_id}-slider", "index": layer_id},
                    min=0, max=last_index, step=1, value=index,
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ])

        return html.Div(
            [header, body],
            id=f"{self.css_id}-box-{layer_id}",
            className="position-box",
            style={
                "border": "1px solid rgba(255,255,255,0.15)",
                "borderRadius": "8px",
                "padding": "12px",
                "minWidth": "360px",
                "flex": "1 1 360px",
            },
        )

    @staticmethod
    def _render_header(layer_id: str, info: Dict[str, Any]) -> html.Div:
        details = ", ".join(f"{k}: {v}" for k, v in info.items()) if info else ""
        return html.Div([
            html.Strong(layer_id),
            html.Span(f" — {details}" if details else "", style={"opacity": 0.7, "fontSize": "0.85em"}),
        ], style={"marginBottom": "8px"})

    @staticmethod
    def _render_loading(progress: Dict[str, int]) -> html.Div:
        collected, required = progress["collected"], progress["required"]
        pct = 0 if required <= 0 else min(100, int(100 * collected / required))
        return html.Div([
            html.Div("Collecting samples for PCA…", style={"marginBottom": "4px"}),
            html.Div(f"{collected} / {required}", style={"fontSize": "0.85em", "opacity": 0.8}),
            html.Div(
                html.Div(style={
                    "width": f"{pct}%", "height": "6px",
                    "backgroundColor": "#4C78A8", "borderRadius": "3px",
                    "transition": "width 0.3s ease",
                }),
                style={
                    "width": "100%", "height": "6px",
                    "backgroundColor": "rgba(255,255,255,0.1)",
                    "borderRadius": "3px", "marginTop": "4px",
                },
            ),
        ], style={"height": "300px", "display": "flex", "flexDirection": "column", "justifyContent": "center"})

    # ---------------------------------------------------------------
    # Figure construction
    # ---------------------------------------------------------------
    def _build_figure(self, layer_id: str, position_index: int) -> go.Figure:
        points = self.ui.get_position_at(layer_id, position_index)
        fig = go.Figure()

        if points is None or len(points) == 0:
            fig.update_layout(
                title=f"Positions: {layer_id}", template=PLOTLY_TEMPLATE,
                paper_bgcolor=PLOTLY_PAPER_BGCOLOR, plot_bgcolor=PLOTLY_PLOT_BGCOLOR, height=300,
            )
            return fig

        n = len(points)
        count_on, count_off = self._extract_on_off_counts(layer_id, n)
        total = count_on + count_off
        ratio = np.divide(count_on, total, out=np.full(n, 0.5), where=total > 0)
        scale = self.SIZE_MIN_SCALE + (self.SIZE_MAX_SCALE - self.SIZE_MIN_SCALE) * ratio
        sizes = self.BASE_MARKER_SIZE * scale

        cluster_snapshot = self.ui.get_cluster_for_index(layer_id, position_index)
        labels = cluster_snapshot.get("labels") if cluster_snapshot else None
        if labels is not None and len(labels) == n:
            colors, colorscale = labels, "Turbo"
        else:
            colors, colorscale = "#4C78A8", None

        fig.add_trace(go.Scatter(
            x=points[:, 0], y=points[:, 1],
            mode="markers+text",
            text=[str(i) for i in range(n)],
            textposition="middle center",
            textfont=dict(size=9, color="white"),
            marker=dict(
                size=sizes,
                color=colors,
                colorscale=colorscale,
                showscale=False,
                line=dict(width=1, color="rgba(255,255,255,0.4)"),
            ),
            name=layer_id,
        ))

        if cluster_snapshot is not None:
            centroids = cluster_snapshot.get("centroids")
            if centroids is not None and len(centroids) > 0:
                fig.add_trace(go.Scatter(
                    x=centroids[:, 0], y=centroids[:, 1],
                    mode="markers+text",
                    text=[f"C{i}" for i in range(len(centroids))],
                    textposition="top center",
                    marker=dict(size=16, symbol="x", color="white", line=dict(width=2, color="black")),
                    name=f"{layer_id} centroids",
                ))

        fig.update_layout(
            title=f"Positions: {layer_id} (step {position_index})",
            margin=dict(l=20, r=20, t=40, b=20),
            height=300,
            template=PLOTLY_TEMPLATE,
            paper_bgcolor=PLOTLY_PAPER_BGCOLOR,
            plot_bgcolor=PLOTLY_PLOT_BGCOLOR,
            showlegend=False,
        )
        return fig

    def _extract_on_off_counts(self, layer_id: str, n: int) -> "tuple[np.ndarray, np.ndarray]":
        """Returns (count_on, count_off) arrays of length n for this layer.

        ModularUI.stats[layer_id] is an ndarray of shape (N, 4), columns
        [count_on, count_off, score_on, score_off] — one row per embedding
        index, matching the layer's position ordering. N can momentarily
        differ from the current position count n (stats/positions can update
        a beat apart), so we align to n defensively.

        Never truthy-checks the stats value directly (`x or {}` raises
        ValueError on a multi-element ndarray) — always an explicit `is None`.
        """
        stats = self.ui.get_layer_stats(layer_id)

        if stats is None:
            return np.zeros(n), np.zeros(n)

        arr = np.asarray(stats, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2:
            return np.zeros(n), np.zeros(n)

        return self._align(arr[:, 0], n), self._align(arr[:, 1], n)

    @staticmethod
    def _align(arr: np.ndarray, n: int) -> np.ndarray:
        """Pads/truncates a stats array to match the current point count, so a
        transient mismatch between space_stats and positions doesn't crash
        rendering (e.g. stats arriving a beat before/after positions)."""
        if len(arr) == n:
            return arr
        if len(arr) > n:
            return arr[:n]
        out = np.zeros(n)
        out[: len(arr)] = arr
        return out

    # ---------------------------------------------------------------
    # Callbacks
    #
    # NOTE: call `positions_display.register_callbacks(app)` once during
    # setup (e.g. alongside `home_layout.register_callbacks(app, ui)` in
    # ModularUI.__init__). This widget owns its own polling Interval and
    # pattern-matching slider callbacks so it doesn't need any changes to
    # HomeLayout's existing per-poll rendering.
    # ---------------------------------------------------------------
    def register_callbacks(self, app: Dash) -> None:

        @app.callback(
            Output(self.css_id, "children"),
            Input(f"{self.css_id}-poll", "n_intervals"),
            State({"type": f"{self.css_id}-slider", "index": ALL}, "value"),
            State({"type": f"{self.css_id}-slider", "index": ALL}, "id"),
        )
        def _poll(_n_intervals, slider_values, slider_ids):
            # Preserve each layer's current scrub position across the periodic
            # full re-render, instead of always snapping back to the latest step.
            overrides = {sid["index"]: val for sid, val in zip(slider_ids, slider_values)}
            return [
                self._render_layer_box(layer_id, overrides.get(layer_id))
                for layer_id in self._known_layer_ids()
            ]

        @app.callback(
            Output({"type": f"{self.css_id}-graph", "index": MATCH}, "figure"),
            Input({"type": f"{self.css_id}-slider", "index": MATCH}, "value"),
            State({"type": f"{self.css_id}-slider", "index": MATCH}, "id"),
        )
        def _on_slider(value, slider_id):
            return self._build_figure(slider_id["index"], value)