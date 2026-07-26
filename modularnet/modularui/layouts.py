from typing import TYPE_CHECKING

from dash import Dash, Input, Output, dcc, html

from modularnet.modularui.components import ConfigDisplay, ModelDisplay,Title, VarsDisplay
from modularnet.modularui.plots import OverallSpaceLossDisplay,LayerSpaceLossDisplay,PositionsDisplay,TrainingMetricsDisplay

if TYPE_CHECKING:
    from modularnet.modularui.modularui import ModularUI

class HomeLayout:
    """Builds the static page skeleton once, and wires the callbacks that
    keep it live from then on.

    Holds no per-request data itself. Each widget instance here only holds
    static config (ids/styling); live data is read from the `ModularUI`
    instance passed into `register_callbacks`, fresh, on every poll tick.
    """

    def __init__(self, theme:str = 'default'):
        self.css = [f"themes/{theme}.css"]

        self.title = Title("ModularNet UI")
        self.vars = VarsDisplay(css_id="modularnet-vars")
        self.model = ModelDisplay(css_id="modularnet-model")
        self.config = ConfigDisplay(css_id="modularnet-config")
        
        # The 4 distinct plot widgets
        self.training_metrics = TrainingMetricsDisplay(css_id="modularnet-training-metrics")
        self.overall_spaceloss = OverallSpaceLossDisplay(css_id="modularnet-overall-spaceloss")
        self.layer_spaceloss = LayerSpaceLossDisplay(css_id="modularnet-layer-spaceloss")
        self.positions = PositionsDisplay(css_id="modularnet-positions")

    def page_layout(self, poll_interval_ms: int = 1000):
        return self._layout_dashboard(poll_interval_ms=poll_interval_ms)
    
   

    def _layout_dashboard(self, poll_interval_ms: int = 1000):
        return html.Div([
            self.title.render(),
            dcc.Interval(id="modularnet-poll", interval=poll_interval_ms, n_intervals=0),
            
            # Master Grid
            html.Div(className="dashboard-layout", children=[
                
                # Left Sidebar (Text, configs, model summaries)
                html.Div(className="sidebar", children=[
                    html.Div(id="vars-container", className="card", children=self.vars.render()),
                    html.Div(id="config-container", className="card", children=self.config.render()),
                    html.Div(id="model-container", className="card", children=self.model.render()),
                ]),
                
                # Right Main Area (Charts)
                html.Div(className="main-content", children=[
                    
                    # Top wide charts
                    html.Div(className="chart-grid", children=[
                        html.Div(id="training-metrics-container", className="card", children=self.training_metrics.render()),
                        html.Div(id="overall-spaceloss-container", className="card", children=self.overall_spaceloss.render()),
                    ]),
                    
                    # Per-layer charts wrapping grid
                    html.Div(id="layer-spaceloss-container", className="card", children=self.layer_spaceloss.render()),
                    html.Div(id="positions-container", className="card", children=self.positions.render()),
                ])
                
            ])
        ])

    def _layout_simple(self, poll_interval_ms: int = 1000):
            return html.Div([
                self.title.render(),
                dcc.Interval(id="modularnet-poll", interval=poll_interval_ms, n_intervals=0),
                
                # Text / Config panels
                html.Div(id="vars-container", children=self.vars.render()),
                html.Div(id="model-container", children=self.model.render()),
                html.Div(id="config-container", children=self.config.render()),
                
                html.Hr(),
                
                # The 4 distinct charting containers
                html.Div(id="training-metrics-container", children=self.training_metrics.render()),
                html.Div(id="overall-spaceloss-container", children=self.overall_spaceloss.render()),
                html.Div(id="layer-spaceloss-container", children=self.layer_spaceloss.render()),
                html.Div(id="positions-container", children=self.positions.render()),
            ])

    def register_callbacks(self, app: Dash, ui: "ModularUI") -> None:  # noqa: F821
        """`ui` is the live ModularUI instance. Each callback grabs a
        consistent snapshot of the relevant state under `ui._lock`, then
        hands it to the matching widget's `render(...)`.
        """

        @app.callback(Output("vars-container", "children"), Input("modularnet-poll", "n_intervals"))
        def _render_vars(_n):
            with ui._lock:
                all_vars = dict(ui.all_vars)
                all_vars.pop("trial_config", None)  # Don't render the full config in the vars panel
            return self.vars.render(all_vars)

        @app.callback(Output("model-container", "children"), Input("modularnet-poll", "n_intervals"))
        def _render_model(_n):
            with ui._lock:
                model_summary = ui.all_vars.get("model_summary")
            return self.model.render(model_summary)

        @app.callback(Output("config-container", "children"), Input("modularnet-poll", "n_intervals"))
        def _render_config(_n):
            with ui._lock:
                config = ui.all_vars.get("trial_config", {})
            return self.config.render(config)

        # --- Chart Callbacks ---

        @app.callback(Output("training-metrics-container", "children"), Input("modularnet-poll", "n_intervals"))
        def _render_training_metrics(_n):
            with ui._lock:
                # Assuming shape: {metric_name: [values]}
                metrics = {k: list(v) for k, v in ui.training_metrics.items()}
            return self.training_metrics.render(metrics)

        @app.callback(Output("overall-spaceloss-container", "children"), Input("modularnet-poll", "n_intervals"))
        def _render_overall_spaceloss(_n):
            with ui._lock:
                # Assuming shape: {metric_name: [values]}
                metrics = {k: list(v) for k, v in ui.overall_spaceloss.items()}
            return self.overall_spaceloss.render(metrics)

        @app.callback(Output("layer-spaceloss-container", "children"), Input("modularnet-poll", "n_intervals"))
        def _render_layer_spaceloss(_n):
            with ui._lock:
                # Assuming shape: {layer_id: {metric_name: [values]}}
                metrics = {
                    layer_id: {k: list(v) for k, v in layer_metrics.items()} 
                    for layer_id, layer_metrics in ui.layer_spaceloss.items()
                }
            return self.layer_spaceloss.render(metrics)

        @app.callback(Output("positions-container", "children"), Input("modularnet-poll", "n_intervals"))
        def _render_positions(_n):
            with ui._lock:
                # Assuming shape: {layer_id: [arrays]}
                positions = {k: list(v) for k, v in ui.positions_plot.items()}
            return self.positions.render(positions)