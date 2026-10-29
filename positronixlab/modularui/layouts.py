import logging
from typing import TYPE_CHECKING
import dash
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, dcc, html, ALL, ctx

from positronixlab.modularui.components import ConfigOverlay, Container, ModelDAGDisplay, Title, VarsDisplay, Widget
from positronixlab.modularui.docks import DashboardLayout, Panel
from positronixlab.modularui.plots import GroupedChartsDisplay, LayerBoardDisplay
from positronixlab.modularui.settings import GLOBAL_TRACE_NAME, SPACELOSS_FAMILY

if TYPE_CHECKING:
    from positronixlab.modularui.modularui import ModularUI

logger = logging.getLogger(__name__)


class _BoundWidget(Widget):
    """Helper to delay a widget's render call with specific arguments."""
    def __init__(self, widget: Widget, *args, **kwargs):
        self.widget = widget
        self.args = args
        self.kwargs = kwargs

    def render(self):
        # Callable args are evaluated at render time, so a re-render sees current data.
        args = [a() if callable(a) else a for a in self.args]
        return self.widget.render(*args, **self.kwargs)


class HomeLayout:

    def __init__(self, ui: "ModularUI", theme: str = 'default', poll_interval_ms: int = 3000):
        self.css = [f"themes/{theme}.css", "themes/toggle-switches.css"]
        self.poll_interval_ms = poll_interval_ms

        self.title = Title("PositronixLab UI")
        self.vars = VarsDisplay(css_id="positronixlab-vars")
        self.config_overlay = ConfigOverlay()

        # Initialize displays
        # train/val on the same chart per metric (loss first, then accuracy, then the rest)
        self.training_metrics = GroupedChartsDisplay(
            css_id="positronixlab-training-metrics", groups_fn=ui.get_training_groups,
            empty_text="no training metrics yet",
        )
        # one chart per space-loss metric: a thin line per layer + a bold mean over all layers
        self.overall_spaceloss = GroupedChartsDisplay(
            css_id="positronixlab-overall-spaceloss",
            groups_fn=lambda: ui.get_family_groups(SPACELOSS_FAMILY),
            label_order=lambda labels: list(labels), emphasize=GLOBAL_TRACE_NAME,
            height=110, flex_basis=200, empty_text="no space loss yet",
        )
        
        self.dag = ModelDAGDisplay(ui=ui, css_id="positronixlab-dag", toggle_id="positronixlab-dag-toggle")
        self.layer_board = LayerBoardDisplay(ui=ui, css_id="positronixlab-layerboard")

        panels = {
            "metrics-panel": Panel("metrics-panel", "Training Metrics", _BoundWidget(self.training_metrics)),
            "spaceloss-panel": Panel("spaceloss-panel", "Space Loss", _BoundWidget(self.overall_spaceloss)),
            "vars-panel": Panel("vars-panel", "Variables", Container("vars-container", self.vars)),
            "dag-panel": Panel("dag-panel", "Model DAG", self.dag),
            "layer-board-panel": Panel("layer-board-panel", "Layer Board", self.layer_board),
        }

        self.dashboard_layout = DashboardLayout(
            top_panels=[panels["metrics-panel"], panels["spaceloss-panel"], panels["vars-panel"]],
            left_panel=panels["dag-panel"],
            main_panel=panels["layer-board-panel"],
        )

        self.default_state = {pid: {"collapsed": False, "maximized": False, "title": p.title} for pid, p in panels.items()}

    def page_layout(self):
        return self._layout_dashboard()

    def _layout_dashboard(self):
        return dbc.Container(
            fluid=True, className="dbc",
            style={"minHeight": "100vh", "display": "flex", "flexDirection": "column", "padding": "0"},
            children=[
                html.Div([self.title.render(), html.Div(id="epoch-container"), html.Div(id="config-selected-container"), self.config_overlay.render_button()], style={"flex": "0 0 auto", "padding": "8px 16px", "display": "flex", "justifyContent": "space-between", "alignItems": "center"}),
                dcc.Interval(id="positronixlab-poll", interval=self.poll_interval_ms, n_intervals=0),

                html.Div(id="collapsed-tabs-bar", style={"display": "flex", "gap": "8px", "padding": "0 16px", "flexWrap": "wrap", "marginBottom": "4px", "flexShrink": 0}),
                dcc.Store(id="panel-state", data=self.default_state),

                html.Div(
                    id="grid-root",
                    style={"width": "100%", "padding": "0 16px 16px"},
                    children=[self.dashboard_layout.render(self.default_state)]
                ),
                self.config_overlay.render_overlay(),
            ]
        )

    @staticmethod
    def _format_epoch(epoch_value):
        if not epoch_value:
            text = "epoch: —"
        else:
            epoch, epochs = epoch_value
            text = f"epoch: {epoch} / {epochs}"
        return html.Div(text, style={"fontFamily": "var(--font-mono, monospace)", "fontSize": "0.9em"})

    def register_callbacks(self, app: Dash, ui: "ModularUI") -> None:
        @app.callback(
            Output("panel-state", "data"),
            Input({"type": "btn-collapse", "index": ALL}, "n_clicks"),
            Input({"type": "btn-expand", "index": ALL}, "n_clicks"),
            Input({"type": "btn-maximize", "index": ALL}, "n_clicks"),
            State("panel-state", "data"),
            prevent_initial_call=True
        )
        def _update_panel_state(col_clicks, exp_clicks, max_clicks, current_state):
            trigger = ctx.triggered_id
            if not current_state:
                current_state = self.default_state

            if isinstance(trigger, dict):
                pid = trigger.get("index")
                if pid in current_state:
                    action = trigger.get("type")
                    if action == "btn-collapse":
                        current_state[pid]["collapsed"] = True
                    elif action == "btn-expand":
                        current_state[pid]["collapsed"] = False
                    elif action == "btn-maximize":
                        current_state[pid]["maximized"] = not current_state[pid].get("maximized", False)
            return current_state

        @app.callback(
            Output("grid-root", "children"),
            Output("collapsed-tabs-bar", "children"),
            Input("panel-state", "data")
        )
        def _render_layout(state):
            if not state:
                state = self.default_state

            tabs = []
            for pid, p_state in state.items():
                if p_state.get("collapsed"):
                    tabs.append(
                        html.Div(
                            className="collapsed-tab",
                            style={"display": "flex", "alignItems": "center", "backgroundColor": "rgba(255,255,255,0.05)", "padding": "4px 8px", "borderRadius": "4px", "border": "1px solid rgba(255,255,255,0.1)", "cursor": "pointer", "fontSize": "0.75em"},
                            children=[
                                html.Span(p_state["title"], style={"marginRight": "8px"}),
                                html.Button("□", id={"type": "btn-expand", "index": pid}, style={"background": "none", "border": "none", "color": "inherit", "padding": "0", "cursor": "pointer", "opacity": 0.6})
                            ]
                        )
                    )
            return [self.dashboard_layout.render(state)], tabs

        @app.callback(Output("epoch-container", "children"), Input("positronixlab-poll", "n_intervals"))
        def _render_epoch(_n):
            return self._format_epoch(ui.get_var("epoch"))

        # VarsDisplay is updated directly via text replacements without triggering heavy React graph diffs.
        @app.callback(Output("vars-container", "children"), Input("positronixlab-poll", "n_intervals"))
        def _render_vars(_n):
            snapshot = ui.get_all_vars_snapshot()
            if not snapshot:
                return dash.no_update
            return self.vars.render(snapshot)

        @app.callback(
            Output("config-selected-container", "children"),
            Input("positronixlab-poll", "n_intervals"),
            Input(self.config_overlay.picker_id, "value"),
        )
        def _render_config_selected(_n, selected_keys):
            return self.config_overlay.render_selected(ui.get_config(), selected_keys)

        # Delegate patch lifecycle execution to the display widgets
        self.training_metrics.register_callbacks(app)
        self.overall_spaceloss.register_callbacks(app)
        
        self.dag.register_callbacks(app)
        self.layer_board.register_callbacks(app, dag_toggle_id=self.dag.toggle_id)
        self.config_overlay.register_callbacks(app, ui)