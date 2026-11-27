from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import MATCH, Dash, Input, Output, Patch, State, ctx, dcc, html
from dash_bootstrap_templates import load_figure_template

from positronixlab.modularui.components import Widget
from positronixlab.modularui.settings import (
    PLOTLY_PAPER_BGCOLOR,
    PLOTLY_PLOT_BGCOLOR,
    PLOTLY_TEMPLATE,
    SPACELOSS_FAMILY,
    THEME_NAME,
)

if TYPE_CHECKING:
    from positronixlab.modularui.modularui import ModularUI

load_figure_template(THEME_NAME)

POLL_ID = "positronixlab-poll"
_NO_DATA_STYLE = {"opacity": 0.6, "fontSize": "0.85em"}


def _line_figure(title: str, height: int, trace_names: list[str], legend: bool = False,
                 emphasize: str | None = None) -> go.Figure:
    fig = go.Figure()
    for name in trace_names:
        if emphasize is not None:
            # overview charts: thin line per layer, bold line for the aggregate
            line = dict(width=3) if name == emphasize else dict(width=1)
            fig.add_trace(go.Scatter(y=[], mode="lines", name=name, line=line,
                                     opacity=1.0 if name == emphasize else 0.55))
        else:
            fig.add_trace(go.Scatter(y=[], mode="lines+markers", name=name, marker=dict(size=4)))
    fig.update_layout(
        title=dict(text=title, font=dict(size=11)),
        margin=dict(l=10, r=10, t=24, b=10),
        height=height,
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=PLOTLY_PAPER_BGCOLOR,
        plot_bgcolor=PLOTLY_PLOT_BGCOLOR,
        showlegend=legend,
        legend=dict(orientation="h", x=1, xanchor="right", y=1.18, font=dict(size=9)),
    )
    return fig


class MetricRow(Widget):
    """Wrapping row of small line charts, one per key (any number of keys).

    per_layer=True: the same row is rendered once per layer card, so the pattern
    index is "<layer_id>|<key>" (unique across cards) and the data source is
    called as data_source_fn(layer_id).
    """
    def __init__(self, css_id: str, height: int = 90, wrap: bool = False, per_layer: bool = False):
        self.css_id = css_id
        self.height = height
        self.wrap = wrap
        self.per_layer = per_layer

    def render(self, keys: list[str], layer_id: str | None = None):
        if not keys:
            return html.Div("no data yet", className=self.css_id, style=_NO_DATA_STYLE)

        flex = "0 0 calc(50% - 5px)" if self.wrap else "1 1 150px"
        charts = [
            html.Div(
                dcc.Graph(
                    id={"type": f"{self.css_id}-graph", "index": f"{layer_id}|{key}" if self.per_layer else key},
                    figure=_line_figure(key, self.height, [key]),
                    config={"displayModeBar": False},
                ),
                style={"flex": flex, "minWidth": "0"},
            )
            for key in keys
        ]
        return html.Div(charts, className=self.css_id, style={"display": "flex", "flexWrap": "wrap", "gap": "10px"})

    def register_callbacks(self, app: Dash, data_source_fn) -> None:
        @app.callback(
            Output({"type": f"{self.css_id}-graph", "index": MATCH}, "figure"),
            Input(POLL_ID, "n_intervals"),
            State({"type": f"{self.css_id}-graph", "index": MATCH}, "id"),
        )
        def _patch_metric(_n, graph_id):
            index = graph_id["index"]
            if self.per_layer:
                layer_id, key = index.rsplit("|", 1)
                metrics = data_source_fn(layer_id)
            else:
                key = index
                metrics = data_source_fn()
            values = metrics.get(key, [])
            if not values:
                return dash.no_update

            patched_fig = Patch()
            patched_fig["data"][0]["y"] = values
            return patched_fig


class SpaceLossRow(Widget):
    """One chart per loss label for ONE layer card. Labels (so the number of charts) come
    from labels_fn(), i.e. from whatever dict the loss function returned."""
    def __init__(self, css_id: str, labels_fn: Callable[[], list[str]], height: int = 90, wrap: bool = False):
        self.css_id = css_id
        self.labels_fn = labels_fn
        self._row = MetricRow(css_id=css_id, height=height, wrap=wrap, per_layer=True)

    def render(self, layer_id: str):
        return self._row.render(list(self.labels_fn()), layer_id)

    def register_callbacks(self, app: Dash, ui: "ModularUI"):
        # ONE callback for every layer card (the layer is encoded in the graph id).
        self._row.register_callbacks(app, ui.get_layer_spaceloss)


def loss_then_accuracy_first(labels: list[str]) -> list[str]:
    """Order for the training panel: loss charts, then accuracy charts, then the rest."""
    def rank(label: str):
        low = label.lower()
        return (0 if "loss" in low else 1 if "acc" in low else 2, label)
    return sorted(labels, key=rank)


class GroupedChartsDisplay(Widget):
    """One chart per label; every series of that label (train/val, or one per layer...) is a
    trace on it. groups_fn() -> {label: {trace_name: values}}. Charts, traces and labels are
    all discovered from the data, and the panel rebuilds itself when that set changes."""
    def __init__(self, css_id: str, groups_fn: Callable[[], dict[str, dict[str, list[float]]]],
                 label_order: Callable[[list[str]], list[str]] = loss_then_accuracy_first,
                 emphasize: str | None = None, height: int = 120, flex_basis: int = 240,
                 empty_text: str = "no data yet"):
        self.css_id = css_id
        self.groups_fn = groups_fn
        self.label_order = label_order
        self.emphasize = emphasize
        self.height = height
        self.flex_basis = flex_basis
        self.empty_text = empty_text

    def _trace_names(self, traces) -> list[str]:
        """Trace order, identical at render and patch time. The emphasized one goes last (on top)."""
        names = sorted(t for t in traces if t != self.emphasize)
        if self.emphasize in traces:
            names.append(self.emphasize)
        return names

    def _signature(self, groups) -> list[str]:
        return [f"{label}|{t}" for label in self.label_order(list(groups)) for t in self._trace_names(groups[label])]

    def _render_charts(self, groups):
        if not groups:
            return html.Div(self.empty_text, className=self.css_id, style=_NO_DATA_STYLE)
        charts = []
        for label in self.label_order(list(groups)):
            traces = self._trace_names(groups[label])
            charts.append(html.Div(
                dcc.Graph(
                    id={"type": f"{self.css_id}-graph", "index": label},
                    figure=_line_figure(label, self.height, traces,
                                        legend=self.emphasize is None and len(traces) > 1,
                                        emphasize=self.emphasize),
                    config={"displayModeBar": False},
                ),
                style={"flex": f"1 1 {self.flex_basis}px", "minWidth": "0"},
            ))
        return html.Div(charts, className=self.css_id, style={"display": "flex", "flexWrap": "wrap", "gap": "10px"})

    def render(self):
        groups = self.groups_fn()
        return html.Div([
            dcc.Store(id=f"{self.css_id}-keys", data=self._signature(groups)),
            html.Div(self._render_charts(groups), id=f"{self.css_id}-container"),
        ])

    def register_callbacks(self, app: Dash):
        @app.callback(
            Output({"type": f"{self.css_id}-graph", "index": MATCH}, "figure"),
            Input(POLL_ID, "n_intervals"),
            State({"type": f"{self.css_id}-graph", "index": MATCH}, "id"),
        )
        def _patch(_n, graph_id):
            traces = self.groups_fn().get(graph_id["index"])
            if not traces:
                return dash.no_update
            patched_fig = Patch()
            for i, name in enumerate(self._trace_names(traces)):
                patched_fig["data"][i]["y"] = traces[name]
            return patched_fig

        @app.callback(
            Output(f"{self.css_id}-container", "children"),
            Output(f"{self.css_id}-keys", "data"),
            Input(POLL_ID, "n_intervals"),
            State(f"{self.css_id}-keys", "data"),
        )
        def _refresh(_n, rendered):
            groups = self.groups_fn()
            sig = self._signature(groups)
            if sig == (rendered or []):
                return dash.no_update, dash.no_update
            return self._render_charts(groups), sig


class LayerBoardDisplay(Widget):
    SIZE_MIN_SCALE = 1.0
    SIZE_MAX_SCALE = 3.0
    BASE_MARKER_SIZE = 14

    _HEADER_INFO_KEYS = ("class_name", "unit_type", "in_units", "out_units", "activation_fn")

    def __init__(self, ui: "ModularUI", css_id: str = "positronixlab-layerboard"):
        self.ui = ui
        self.css_id = css_id
        self._spaceloss_row = SpaceLossRow(
            css_id=f"{css_id}-spaceloss",
            labels_fn=lambda: ui.get_labels(SPACELOSS_FAMILY),
            wrap=True,
            height=215,
        )

    def render(self, visible_layers: list[str] | None = None) -> html.Div:
        return html.Div([
            dcc.Store(id=f"{self.css_id}-ids", data=self._signature(visible_layers)),
            html.Div(
                self._render_all_cards(visible_layers),
                id=self.css_id,
                style={"display": "flex", "flexDirection": "column", "gap": "16px"},
            ),
        ])

    def _signature(self, visible_layers: list[str] | None) -> dict[str, list[str]]:
        return {"layers": self._filtered_layer_ids(visible_layers), "labels": list(self._spaceloss_row.labels_fn())}

    def _known_layer_ids(self) -> list[str]:
        return self.ui.get_known_layer_ids()

    def _filtered_layer_ids(self, visible_layers: list[str] | None) -> list[str]:
        all_ids = self._known_layer_ids()
        if visible_layers is None:
            return all_ids
        visible = set(visible_layers)
        return [lid for lid in all_ids if lid in visible]

    def _render_all_cards(self, visible_layers: list[str] | None) -> list[dbc.Card]:
        return [self._render_layer_card(layer_id) for layer_id in self._filtered_layer_ids(visible_layers)]

    def _render_layer_card(self, layer_id: str) -> dbc.Card:
        info = self.ui.get_observed_layer_info(layer_id)
        header = self._render_header(layer_id, info)
        spaceloss_section = self._spaceloss_row.render(layer_id)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[], y=[], mode="markers+text", name=layer_id))
        fig.add_trace(go.Scatter(
            x=[], y=[], mode="markers+text", name=f"{layer_id} centroids",
            marker=dict(symbol="star", size=18, line=dict(width=1, color="white")),
            textposition="top center",
            textfont=dict(color="white")
        ))
        fig.update_layout(
            title=f"Positions: {layer_id}",
            margin=dict(l=20, r=20, t=40, b=20),
            height=450,
            template=PLOTLY_TEMPLATE,
            paper_bgcolor=PLOTLY_PAPER_BGCOLOR,
            plot_bgcolor=PLOTLY_PLOT_BGCOLOR,
            showlegend=False,
        )

        position_section = html.Div([
            dcc.Graph(
                id={"type": f"{self.css_id}-graph", "index": layer_id},
                figure=fig,
                style={"height": "480px"},
            ),
            html.Div([
                html.Div(
                    dcc.Slider(
                        id={"type": f"{self.css_id}-slider", "index": layer_id},
                        min=0, max=0, step=1, value=0,  # Initialize max=0 so first tick tracks live
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                    style={"flex": "1"}
                ),
                dbc.Button(
                    "Live",
                    id={"type": f"{self.css_id}-live-btn", "index": layer_id},
                    size="sm",
                    color="primary",
                    outline=True,
                    className="ms-3 mb-3" # Give slight margin to align with slider rail
                )
            ], style={"display": "flex", "alignItems": "center", "padding": "0 10px"})
        ])

        content = html.Div([
            html.Div(position_section, style={"flex": "1 1 50%", "minWidth": "0"}),
            html.Div([
                html.Div("Space Loss", style={"fontSize": "0.85em", "opacity": 0.7, "marginBottom": "8px"}),
                spaceloss_section
            ], style={
                "flex": "1 1 50%", 
                "minWidth": "0", 
                "display": "flex", 
                "flexDirection": "column", 
                "justifyContent": "flex-start" 
            })
        ], style={"display": "flex", "gap": "20px"})

        return dbc.Card(
            dbc.CardBody([header, html.Hr(className="my-2"), content]),
            id=f"{self.css_id}-box-{layer_id}",
            className="layer-board-card mb-0",
            style={"width": "100%"},
        )

    @classmethod
    def _render_header(cls, layer_id: str, info: dict[str, Any]) -> html.Div:
        parts = [f"{k}: {info[k]}" for k in cls._HEADER_INFO_KEYS if info.get(k) not in (None, "")]
        details = ", ".join(parts)
        return html.Div([
            html.Strong(layer_id),
            html.Span(f" — {details}" if details else "", style={"opacity": 0.7, "fontSize": "0.85em"}),
        ], style={"marginBottom": "8px"})

    def register_callbacks(self, app: Dash, dag_toggle_id: str) -> None:

        @app.callback(
            Output(self.css_id, "children"),
            Output(f"{self.css_id}-ids", "data"),
            Input(dag_toggle_id, "data"),
            Input(POLL_ID, "n_intervals"),
            State(f"{self.css_id}-ids", "data"),
        )
        def _update_structure(visible_layers, _n, rendered_sig):
            sig = self._signature(visible_layers)
            if ctx.triggered_id == POLL_ID and sig == (rendered_sig or {}):
                return dash.no_update, dash.no_update
            return self._render_all_cards(visible_layers), sig

        @app.callback(
            Output({"type": f"{self.css_id}-graph", "index": MATCH}, "figure"),
            Output({"type": f"{self.css_id}-slider", "index": MATCH}, "max"),
            Output({"type": f"{self.css_id}-slider", "index": MATCH}, "value"),
            Input(POLL_ID, "n_intervals"),
            Input({"type": f"{self.css_id}-live-btn", "index": MATCH}, "n_clicks"),
            Input({"type": f"{self.css_id}-slider", "index": MATCH}, "value"),
            State({"type": f"{self.css_id}-slider", "index": MATCH}, "max"),
            State({"type": f"{self.css_id}-graph", "index": MATCH}, "id"),
        )
        def _patch_layer_graph(_n, _live_clicks, current_slider_val, current_max, graph_id):
            layer_id = graph_id["index"]

            n_steps = self.ui.get_position_steps(layer_id)
            if n_steps == 0:
                return dash.no_update, dash.no_update, dash.no_update

            last_index = max(n_steps - 1, 0)
            
            # Determine logic flow based on what triggered the callback
            triggered = ctx.triggered_id
            is_live_btn = isinstance(triggered, dict) and triggered.get("type") == f"{self.css_id}-live-btn"
            is_slider = isinstance(triggered, dict) and triggered.get("type") == f"{self.css_id}-slider"

            if is_live_btn:
                target_index = last_index
            elif is_slider:
                target_index = current_slider_val
            else:
                # Interval trigger: pin to latest if it was previously at max, otherwise stay at manual offset
                c_val = current_slider_val or 0
                c_max = current_max or 0
                if c_val >= c_max:
                    target_index = last_index
                else:
                    target_index = c_val

            points = self.ui.get_position_at(layer_id, target_index)
            if points is None or len(points) == 0:
                return dash.no_update, dash.no_update, dash.no_update

            patched_fig = Patch()
            n = len(points)

            patched_fig["data"][0]["x"] = points[:, 0]
            patched_fig["data"][0]["y"] = points[:, 1]
            patched_fig["data"][0]["text"] = [str(i) for i in range(n)]

            stats = self.ui.get_layer_stats(layer_id)
            if stats is not None:
                arr = np.asarray(stats, dtype=float)
                c_on = arr[:n, 0] if len(arr) >= n else np.pad(arr[:, 0], (0, max(0, n - len(arr))))
                c_off = arr[:n, 1] if len(arr) >= n else np.pad(arr[:, 1], (0, max(0, n - len(arr))))
                total = c_on + c_off
                ratio = np.divide(c_on, total, out=np.full(n, 0.5), where=total > 0)
                sizes = self.BASE_MARKER_SIZE * (self.SIZE_MIN_SCALE + (self.SIZE_MAX_SCALE - self.SIZE_MIN_SCALE) * ratio)
                patched_fig["data"][0]["marker"]["size"] = sizes

            cluster_snapshot = self.ui.get_cluster_for_index(layer_id, target_index)
            if cluster_snapshot:
                centroids = cluster_snapshot.get("centroids")
                k_clusters = len(centroids) if centroids is not None else 0
                cmax_val = max(1, k_clusters - 1)

                labels = cluster_snapshot.get("labels")
                if labels is not None and len(labels) == n:
                    patched_fig["data"][0]["marker"]["color"] = labels
                    patched_fig["data"][0]["marker"]["colorscale"] = "Turbo"
                    patched_fig["data"][0]["marker"]["cmin"] = 0
                    patched_fig["data"][0]["marker"]["cmax"] = cmax_val

                if centroids is not None and k_clusters > 0:
                    patched_fig["data"][1]["x"] = centroids[:, 0]
                    patched_fig["data"][1]["y"] = centroids[:, 1]
                    patched_fig["data"][1]["text"] = [f"C{i}" for i in range(k_clusters)]

                    patched_fig["data"][1]["marker"]["color"] = list(range(k_clusters))
                    patched_fig["data"][1]["marker"]["colorscale"] = "Turbo"
                    patched_fig["data"][1]["marker"]["cmin"] = 0
                    patched_fig["data"][1]["marker"]["cmax"] = cmax_val

            offset = self.ui.get_position_offset(layer_id)
            patched_fig["layout"]["title"]["text"] = f"Positions: {layer_id} (step {target_index + offset})"

            return patched_fig, last_index, target_index

        self._spaceloss_row.register_callbacks(app, self.ui)
