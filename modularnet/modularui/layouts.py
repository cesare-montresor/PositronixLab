import logging
from typing import TYPE_CHECKING

from dash import Dash, Input, Output, dcc, html

from modularnet.modularui.components import ConfigDisplay, ModelDisplay, Title, VarsDisplay
from modularnet.modularui.plots import (
    LayerSpaceLossDisplay,
    OverallSpaceLossDisplay,
    PositionsDisplay,
    TrainingMetricsDisplay,
)

if TYPE_CHECKING:
    from modularnet.modularui.modularui import ModularUI

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Layout and Callbacks
# ---------------------------------------------------------
class HomeLayout:

    def __init__(self, ui: "ModularUI", theme: str = 'default', poll_interval_ms: int = 3000):
        self.css = [f"themes/{theme}.css"]
        self.poll_interval_ms = poll_interval_ms

        self.title = Title("ModularNet UI")
        self.vars = VarsDisplay(css_id="modularnet-vars")
        self.model = ModelDisplay(css_id="modularnet-model")
        self.config = ConfigDisplay(css_id="modularnet-config")

        self.training_metrics = TrainingMetricsDisplay(css_id="modularnet-training-metrics")
        self.overall_spaceloss = OverallSpaceLossDisplay(css_id="modularnet-overall-spaceloss")
        self.layer_spaceloss = LayerSpaceLossDisplay(css_id="modularnet-layer-spaceloss")
        # PositionsDisplay needs a live `ui` handle: it pulls PCA-collection
        # progress, layer stats, and cluster-sync data on demand (from its own
        # Interval + slider callbacks), rather than being fed a data blob by
        # HomeLayout's generic poll callback like the other widgets.
        self.positions = PositionsDisplay(ui=ui, css_id="modularnet-positions")

    def page_layout(self, ):
        return self._layout_dashboard()

    def _layout_dashboard(self):
        return html.Div([
            self.title.render(),
            dcc.Interval(id="modularnet-poll", interval=self.poll_interval_ms, n_intervals=0),

            html.Div(className="dashboard-layout", children=[
                html.Div(className="sidebar", children=[
                    html.Div(id="vars-container", className="card", children=self.vars.render()),
                    html.Div(id="model-container", className="card", children=self.model.render()),
                ]),

                html.Div(className="main-content", children=[
                    html.Div(className="chart-grid", children=[
                        html.Div(id="config-container", className="card", children=self.config.render()),
                        html.Div(id="training-metrics-container", className="card", children=self.training_metrics.render()),
                        html.Div(id="overall-spaceloss-container", className="card", children=self.overall_spaceloss.render()),
                    ]),

                    html.Div(id="layer-spaceloss-container", className="card", children=self.layer_spaceloss.render()),
                    html.Div(id="positions-container", className="card", children=self.positions.render()),
                ])
            ])
        ])

    def _layout_simple(self):
            return html.Div([
                self.title.render(),
                dcc.Interval(id="modularnet-poll", interval=self.poll_interval_ms, n_intervals=0),

                html.Div(id="vars-container", children=self.vars.render()),
                html.Div(id="model-container", children=self.model.render()),
                html.Div(id="config-container", children=self.config.render()),

                html.Hr(),

                html.Div(id="training-metrics-container", children=self.training_metrics.render()),
                html.Div(id="overall-spaceloss-container", children=self.overall_spaceloss.render()),
                html.Div(id="layer-spaceloss-container", children=self.layer_spaceloss.render()),
                html.Div(id="positions-container", children=self.positions.render()),
            ])

    def register_callbacks(self, app: Dash, ui: "ModularUI") -> None:

        @app.callback(Output("vars-container", "children"), Input("modularnet-poll", "n_intervals"))
        def _render_vars(_n):
            with ui.locks.vars:
                # `trial_config`, etc. are already removed in `update_vars`
                all_vars = dict(ui.all_vars)
            return self.vars.render(all_vars)

        @app.callback(Output("model-container", "children"), Input("modularnet-poll", "n_intervals"))
        def _render_model(_n):
            with ui.locks.model:
                model_summary = ui.model_summary
            return self.model.render(model_summary)

        @app.callback(Output("config-container", "children"), Input("modularnet-poll", "n_intervals"))
        def _render_config(_n):
            with ui.locks.config:
                config = dict(ui.trial_config) if ui.trial_config else {}
            return self.config.render(config)

        # --- Chart Callbacks ---

        @app.callback(Output("training-metrics-container", "children"), Input("modularnet-poll", "n_intervals"))
        def _render_training_metrics(_n):
            with ui.locks.metrics:
                metrics = {k: list(v) for k, v in ui.training_metrics.items()}
            return self.training_metrics.render(metrics)

        @app.callback(Output("overall-spaceloss-container", "children"), Input("modularnet-poll", "n_intervals"))
        def _render_overall_spaceloss(_n):
            # No manual lock needed here! The property manages `locks.spaceloss` internally.
            metrics = {k: list(v) for k, v in ui.overall_spaceloss.items()}
            return self.overall_spaceloss.render(metrics)

        @app.callback(Output("layer-spaceloss-container", "children"), Input("modularnet-poll", "n_intervals"))
        def _render_layer_spaceloss(_n):
            with ui.locks.spaceloss:
                metrics = {
                    layer_id: {k: list(v) for k, v in layer_metrics.items()}
                    for layer_id, layer_metrics in ui.layer_spaceloss.items()
                }
            return self.layer_spaceloss.render(metrics)

        # NOTE: positions no longer has a generic "poll -> push a data blob"
        # callback here. PositionsDisplay owns its own dcc.Interval and its
        # own pattern-matching slider callbacks (per-layer PCA-collection
        # state, cluster-sync lookups, and scrub position all live behind
        # `ui`, which it already holds a reference to). We just need to
        # register them against the same `app`.
        self.positions.register_callbacks(app)
