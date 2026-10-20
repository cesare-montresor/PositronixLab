from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import ALL, MATCH, Dash, Input, Output, State, dcc, html
from dash_bootstrap_templates import load_figure_template

from modularnet.modularui.components import Widget
from modularnet.modularui.settings import PLOTLY_PAPER_BGCOLOR, PLOTLY_PLOT_BGCOLOR, PLOTLY_TEMPLATE, PLOTS_ROW_LABEL_STYLE, PLOTS_SPACELOSS_KEYS, THEME_NAME

if TYPE_CHECKING:
    from modularnet.modularui.modularui import ModularUI

load_figure_template(   THEME_NAME)

class MetricRow(Widget):
    """One small chart per metric key. Uses flexWrap for grid modes."""

    def __init__(self, css_id: str, height: int = 90, wrap: bool = False):
        self.css_id = css_id
        self.height = height
        self.wrap = wrap

    def render(self, metrics: Optional[Dict[str, List[float]]] = None, keys: Optional[List[str]] = None):
        metrics = metrics or {}
        active_keys = keys if keys is not None else sorted(metrics.keys())

        charts = []
        for key in active_keys:
            values = metrics.get(key, [])
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=values, mode="lines", name=key))
            fig.update_layout(
                title=dict(text=key, font=dict(size=11)),
                margin=dict(l=10, r=10, t=24, b=10),
                height=self.height,
                template=PLOTLY_TEMPLATE,
                paper_bgcolor=PLOTLY_PAPER_BGCOLOR,
                plot_bgcolor=PLOTLY_PLOT_BGCOLOR,
                showlegend=False,
            )
            
            # If wrapped is True, elements scale evenly at 50% max to form a 2x2 grid when 4 keys exist.
            flex_style = "0 0 calc(50% - 5px)" if self.wrap else "1 1 0"
            charts.append(html.Div(
                dcc.Graph(figure=fig, config={"displayModeBar": False}),
                style={"flex": flex_style, "minWidth": "0"},
            ))

        if not charts:
            return html.Div("no data yet", className=self.css_id, style={"opacity": 0.6, "fontSize": "0.85em"})

        wrap_style = "wrap" if self.wrap else "nowrap"
        return html.Div(charts, className=self.css_id, style={"display": "flex", "flexWrap": wrap_style, "gap": "10px"})


class SpaceLossRow(Widget):
    def __init__(self, css_id: str, height: int = 90, wrap: bool = False):
        self.css_id = css_id
        self._row = MetricRow(css_id=css_id, height=height, wrap=wrap)

    def render(self, metrics: Optional[Dict[str, List[float]]] = None):
        return self._row.render(metrics, PLOTS_SPACELOSS_KEYS)


class TrainingMetricsDisplay(Widget):
    def __init__(self, css_id: str = "modularnet-training-metrics"):
        self.css_id = css_id
        self._loss_row = MetricRow(css_id=f"{css_id}-loss")
        self._acc_row = MetricRow(css_id=f"{css_id}-acc")
        self._other_row = MetricRow(css_id=f"{css_id}-other")

    @staticmethod
    def _split(metrics: Dict[str, List[float]]):
        loss_keys = sorted(k for k in metrics if "loss" in k.lower())
        acc_keys = sorted(k for k in metrics if "acc" in k.lower())
        other_keys = sorted(k for k in metrics if k not in loss_keys and k not in acc_keys)
        return loss_keys, acc_keys, other_keys

    def render(self, metrics: Optional[Dict[str, List[float]]] = None):
        metrics = metrics or {}
        loss_keys, acc_keys, other_keys = self._split(metrics)

        sections = []
        if loss_keys:
            sections.append(html.Div([html.Div("loss", style=PLOTS_ROW_LABEL_STYLE), self._loss_row.render(metrics, loss_keys)]))
        if acc_keys:
            sections.append(html.Div([html.Div("acc", style=PLOTS_ROW_LABEL_STYLE), self._acc_row.render(metrics, acc_keys)]))
        if other_keys:
            sections.append(html.Div([html.Div("other", style=PLOTS_ROW_LABEL_STYLE), self._other_row.render(metrics, other_keys)]))

        if not sections:
            return html.Div("no training metrics yet", className=self.css_id, style={"opacity": 0.6, "fontSize": "0.85em"})

        return html.Div(sections, className=self.css_id, style={"display": "flex", "flexDirection": "column", "gap": "8px"})


class LayerBoardDisplay(Widget):
    SIZE_MIN_SCALE = 1.0
    SIZE_MAX_SCALE = 3.0
    BASE_MARKER_SIZE = 14
    POLL_INTERVAL_MS = 1000

    _HEADER_INFO_KEYS = ("class_name", "unit_type", "in_units", "out_units", "activation_fn")

    def __init__(self, ui: "ModularUI", css_id: str = "modularnet-layerboard"):
        self.ui = ui
        self.css_id = css_id
        # Enable wrapping for the local metric chart (2x2 grid view)
        self._spaceloss_row = SpaceLossRow(css_id=f"{css_id}-spaceloss", wrap=True)

    def render(self, visible_layers: Optional[List[str]] = None) -> html.Div:
        return html.Div([
            html.Div(
                self._render_all_cards(visible_layers),
                id=self.css_id,
                style={"display": "flex", "flexDirection": "column", "gap": "16px"},
            ),
            dcc.Interval(id=f"{self.css_id}-poll", interval=self.POLL_INTERVAL_MS),
        ])

    def _known_layer_ids(self) -> List[str]:
        return self.ui.get_known_layer_ids()

    def _filtered_layer_ids(self, visible_layers: Optional[List[str]]) -> List[str]:
        all_ids = self._known_layer_ids()
        if not visible_layers:
            return all_ids
        visible = set(visible_layers)
        return [lid for lid in all_ids if lid in visible]

    def _render_all_cards(self, visible_layers: Optional[List[str]]) -> List[html.Div]:
        return [self._render_layer_card(layer_id, forced_index=None) for layer_id in self._filtered_layer_ids(visible_layers)]

    def _render_layer_card(self, layer_id: str, forced_index: Optional[int]) -> html.Div:
        info = self.ui.get_observed_layer_info(layer_id)
        header = self._render_header(layer_id, info)

        spaceloss_section = self._spaceloss_row.render(self.ui.get_layer_spaceloss(layer_id))

        progress = self.ui.get_pca_progress(layer_id)
        if progress is not None:
            position_section = self._render_loading(progress)
        else:
            n_steps = self.ui.get_position_steps(layer_id)
            last_index = max(n_steps - 1, 0)
            if forced_index is None or forced_index >= last_index:
                index = last_index
            else:
                index = forced_index

            position_section = html.Div([
                dcc.Graph(
                    id={"type": f"{self.css_id}-graph", "index": layer_id},
                    figure=self._build_figure(layer_id, index),
                    style={"height": "480px"},
                ),
                dcc.Slider(
                    id={"type": f"{self.css_id}-slider", "index": layer_id},
                    min=0, max=last_index, step=1, value=index,
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ])

        # Left: Position section, Right: 2x2 metric chart matrix
        content = html.Div([
            html.Div(position_section, style={"flex": "1 1 60%", "minWidth": "0"}),
            html.Div([
                html.Div("Space Loss", style={"fontSize": "0.85em", "opacity": 0.7, "marginBottom": "8px"}),
                spaceloss_section
            ], style={"flex": "1 1 40%", "minWidth": "0", "display": "flex", "flexDirection": "column", "justifyContent": "center"})
        ], style={"display": "flex", "gap": "20px"})

        return dbc.Card(
            dbc.CardBody([header, html.Hr(className="my-2"), content]),
            id=f"{self.css_id}-box-{layer_id}",
            className="layer-board-card mb-0",
            style={"width": "100%"},
        )

    @classmethod
    def _render_header(cls, layer_id: str, info: Dict[str, Any]) -> html.Div:
        parts = [f"{k}: {info[k]}" for k in cls._HEADER_INFO_KEYS if info.get(k) not in (None, "")]
        details = ", ".join(parts)
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

    def _build_figure(self, layer_id: str, position_index: int) -> go.Figure:
        points = self.ui.get_position_at(layer_id, position_index)
        fig = go.Figure()

        if points is None or len(points) == 0:
            fig.update_layout(
                title=f"Positions: {layer_id}", template=PLOTLY_TEMPLATE,
                paper_bgcolor=PLOTLY_PAPER_BGCOLOR, plot_bgcolor=PLOTLY_PLOT_BGCOLOR, height=450,
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
            title=f"Positions: {layer_id} (step {position_index + self.ui.get_position_offset(layer_id)})",
            margin=dict(l=20, r=20, t=40, b=20),
            height=450,
            template=PLOTLY_TEMPLATE,
            paper_bgcolor=PLOTLY_PAPER_BGCOLOR,
            plot_bgcolor=PLOTLY_PLOT_BGCOLOR,
            showlegend=False,
        )
        return fig

    def _extract_on_off_counts(self, layer_id: str, n: int) -> "tuple[np.ndarray, np.ndarray]":
        stats = self.ui.get_layer_stats(layer_id)

        if stats is None:
            return np.zeros(n), np.zeros(n)

        arr = np.asarray(stats, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2:
            return np.zeros(n), np.zeros(n)

        return self._align(arr[:, 0], n), self._align(arr[:, 1], n)

    @staticmethod
    def _align(arr: np.ndarray, n: int) -> np.ndarray:
        if len(arr) == n:
            return arr
        if len(arr) > n:
            return arr[:n]
        out = np.zeros(n)
        out[: len(arr)] = arr
        return out

    def register_callbacks(self, app: Dash, dag_toggle_id: str) -> None:

        @app.callback(
            Output(self.css_id, "children"),
            Input(f"{self.css_id}-poll", "n_intervals"),
            Input(dag_toggle_id, "data"), # Store outputs its values via "data", not "value"
            State({"type": f"{self.css_id}-slider", "index": ALL}, "value"),
            State({"type": f"{self.css_id}-slider", "index": ALL}, "id"),
        )
        def _poll(_n_intervals, visible_layers, slider_values, slider_ids):
            overrides = {sid["index"]: val for sid, val in zip(slider_ids, slider_values)}
            return [
                self._render_layer_card(layer_id, overrides.get(layer_id))
                for layer_id in self._filtered_layer_ids(visible_layers)
            ]

        @app.callback(
            Output({"type": f"{self.css_id}-graph", "index": MATCH}, "figure"),
            Input({"type": f"{self.css_id}-slider", "index": MATCH}, "value"),
            State({"type": f"{self.css_id}-slider", "index": MATCH}, "id"),
        )
        def _on_slider(value, slider_id):
            return self._build_figure(slider_id["index"], value)