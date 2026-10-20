import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import dash
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
from dash import Dash, Input, Output, State, dcc, html, ctx, no_update

if TYPE_CHECKING:
    from modularnet.modularui.modularui import ModularUI

logger = logging.getLogger(__name__)


class Widget(ABC):
    """Base class for a reusable Dash component renderer."""
    @abstractmethod
    def render(self, *args, **kwargs):
        ...

    def __call__(self, *args, **kwargs):
        return self.render(*args, **kwargs)


class Container(Widget):
    """Wraps another widget's render() in a div with a stable id, so a poll
    callback can target it (the Output ids used in HomeLayout.register_callbacks)."""
    def __init__(self, container_id: str, inner: Widget):
        self.container_id = container_id
        self.inner = inner

    def render(self):
        return html.Div(self.inner.render(), id=self.container_id)


class Title(Widget):
    def __init__(self, text: str):
        self.text = text

    def render(self):
        return html.H1(self.text)


class ExpandableSection(Widget):
    def __init__(self, expanded: bool = False):
        self.expanded = expanded

    def render(self, title: str, content: str):
        return html.Details(
            [html.Summary(title), html.Pre(content)],
            open=self.expanded,
        )


class VarsDisplay(Widget):
    def __init__(self, css_id: str = "modularnet-vars", css_style: Optional[dict] = None):
        self.css_id = css_id
        self.css_style = css_style or {"whiteSpace": "pre-wrap", "wordBreak": "break-all", "fontSize": "0.85em"}
        self._section = ExpandableSection()

    def render(self, all_vars: Optional[Dict[str, Any]] = None):
        all_vars = all_vars or {}
        if not all_vars:
            return html.Div("no other vars", className=self.css_id, style=self.css_style)

        sections = []
        for name, value in sorted(all_vars.items()):
            if isinstance(value, dict):
                content = "\n".join(f"{k}: {v}" for k, v in sorted(value.items()))
            else:
                content = str(value)
            sections.append(self._section.render(title=name, content=content))

        return html.Div(sections, className=self.css_id, style=self.css_style)


def _format_config_value(value: Any, max_len: int = 60) -> str:
    text = str(value)
    if len(text) > max_len:
        text = text[: max_len - 1] + "…"
    return text


class ConfigOverlay:
    PICKER_POLL_INTERVAL_MS = 5000

    def __init__(self, button_id: str = "config-overlay-button", overlay_id: str = "config-overlay", picker_id: str = "config-overlay-picker"):
        self.button_id = button_id
        self.overlay_id = overlay_id
        self.picker_id = picker_id

    def render_button(self):
        return dbc.Button("config", id=self.button_id, n_clicks=0, color="secondary", outline=True, size="sm")

    def render_overlay(self):
        return dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("trial_config"), close_button=True),
                dbc.ModalBody(
                    [
                        html.Div("check properties to pin them in the top bar.", style={"fontSize": "0.75em", "opacity": 0.6, "marginBottom": "10px"}),
                        dcc.Checklist(
                            id=self.picker_id,
                            options=[], value=[],
                            className="toggle-switch-list",
                            labelStyle={"display": "block", "fontFamily": "var(--font-mono, monospace)", "fontSize": "0.82em", "marginBottom": "4px"},
                        ),
                        dcc.Interval(id=f"{self.picker_id}-poll", interval=self.PICKER_POLL_INTERVAL_MS),
                    ]
                ),
            ],
            id=self.overlay_id, is_open=False, size="lg", scrollable=True,
        )

    def render_selected(self, config: Optional[Dict[str, Any]] = None, selected_keys: Optional[List[str]] = None):
        config = config or {}
        selected_keys = selected_keys or []
        rows = [
            html.Div(f"{k}: {_format_config_value(config[k])}", style={"fontFamily": "var(--font-mono, monospace)", "fontSize": "0.82em"})
            for k in sorted(selected_keys) if k in config
        ]
        if not rows:
            return html.Div("config: none pinned", style={"opacity": 0.5, "fontSize": "0.85em"})
        return html.Div(rows)

    def register_callbacks(self, app: Dash, ui: "ModularUI") -> None:
        @app.callback(
            Output(self.overlay_id, "is_open"),
            Input(self.button_id, "n_clicks"),
            State(self.overlay_id, "is_open"),
            prevent_initial_call=True,
        )
        def _toggle(_n_clicks, is_open):
            return not is_open

        @app.callback(
            Output(self.picker_id, "options"), Output(self.picker_id, "value"),
            Input(f"{self.picker_id}-poll", "n_intervals"), State(self.picker_id, "value"),
        )
        def _refresh_picker(_n, current_value):
            config = ui.get_config()
            known_keys = sorted(config.keys())
            options = [{"label": f"{k}: {_format_config_value(config[k])}", "value": k} for k in known_keys]
            value = [k for k in (current_value or []) if k in known_keys]
            return options, value


class ModelDAGDisplay(Widget):
    """Replaces the old checklist with an interactive Cytoscape DAG.
    Selections sync to a hidden dcc.Store for upstream consumption.
    """
    POLL_INTERVAL_MS = 5000

    def __init__(self, ui: "ModularUI", css_id: str = "modularnet-dag", toggle_id: str = "modularnet-dag-toggle"):
        self.ui = ui
        self.css_id = css_id
        # toggle_id is now the ID of the hidden dcc.Store holding selected nodes.
        self.toggle_id = toggle_id
        self.cyto_id = f"{css_id}-cyto"
        self._last_graph_version = -1

    def _get_elements(self):
        graph = self.ui.get_model_graph()
        nodes = graph.get("nodes", {})
        edges = graph.get("edges", [])
        elements = []

        for nid, node in nodes.items():
            class_name = node.get("class_name", "")
            label = f"{nid}\n({class_name})" if class_name else nid
            elements.append({
                'data': {'id': nid, 'label': label},
                'classes': 'model-node'
            })

        for edge in edges:
            elements.append({
                'data': {'source': edge['source'], 'target': edge['target']}
            })
            
        return elements

    def render(self, checked: Optional[List[str]] = None) -> html.Div:
        return dbc.Card(
            dbc.CardBody([
                html.Div("model DAG", style={"opacity": 0.7, "marginBottom": "8px", "fontSize": "0.85em"}),
                cyto.Cytoscape(
                    id=self.cyto_id,
                    elements=[],
                    layout={'name': 'breadthfirst', 'directed': True, 'spacingFactor': 0.8},
                    style={'width': '100%', 'height': '500px', 'backgroundColor': 'transparent'},
                    stylesheet=[
                        {
                            'selector': 'node',
                            'style': {
                                'content': 'data(label)',
                                'text-wrap': 'wrap',
                                'text-valign': 'center',
                                'text-halign': 'center',
                                'font-size': '10px',
                                'width': '50px',
                                'height': '50px',
                                'background-color': '#4C78A8',
                                'color': '#fff',
                                'text-outline-width': 1,
                                'text-outline-color': '#000'
                            }
                        },
                        {
                            'selector': ':selected',
                            'style': {
                                'background-color': '#E45756',
                                'border-width': '2px',
                                'border-color': '#fff'
                            }
                        },
                        {
                            'selector': 'edge',
                            'style': {
                                'curve-style': 'bezier',
                                'target-arrow-shape': 'triangle',
                                'line-color': '#666',
                                'target-arrow-color': '#666'
                            }
                        }
                    ]
                ),
                dcc.Store(id=self.toggle_id, data=[]),
                html.Div(
                    "Click nodes to toggle visibility.",
                    style={"marginTop": "10px", "color": "var(--text-muted, #888)", "fontSize": "0.72em"},
                ),
                dcc.Interval(id=f"{self.css_id}-poll", interval=self.POLL_INTERVAL_MS),
            ]),
            id=f"{self.css_id}-wrapper",
            className="mb-0",
            style={"width": "260px", "flexShrink": 0, "position": "sticky", "top": "12px"},
        )

    def register_callbacks(self, app: Dash) -> None:
        @app.callback(
            Output(self.cyto_id, "elements"),
            Output(self.toggle_id, "data"),
            Input(f"{self.css_id}-poll", "n_intervals"),
            Input(self.cyto_id, "tapNodeData"), # Changed from selectedNodeData
            State(self.cyto_id, "elements"),
            State(self.toggle_id, "data"),
        )
        def _sync(_n, tapped_node, current_elements, current_data):
            trigger = ctx.triggered_id
            current_selected = current_data if current_data is not None else []

            # 1. Handle Independent Node Toggling
            if trigger == self.cyto_id and tapped_node:
                tapped_id = tapped_node['id']
                
                # Toggle the specific node independently
                if tapped_id in current_selected:
                    current_selected.remove(tapped_id)
                else:
                    current_selected.append(tapped_id)
                
                # Apply selection back to elements to update the UI
                elements = current_elements or self._get_elements()
                for el in elements:
                    if 'id' in el.get('data', {}) and 'source' not in el['data']:
                        el['selected'] = (el['data']['id'] in current_selected)
                        
                return elements, current_selected

            # 2. Handle Polling (existing logic)
            graph_version = self.ui.get_model_graph_version()
            if self._last_graph_version == graph_version and current_elements:
                return no_update, no_update
                
            self._last_graph_version = graph_version
            elements = self._get_elements()
            
            known_ids = [e['data']['id'] for e in elements if 'id' in e['data'] and 'source' not in e['data']]
            previously_known = {e['data']['id'] for e in (current_elements or []) if 'id' in e['data'] and 'source' not in e['data']}
            newly_discovered = [nid for nid in known_ids if nid not in previously_known]
            
            new_selection = [nid for nid in current_selected if nid in known_ids] + newly_discovered

            for el in elements:
                if 'id' in el['data'] and 'source' not in el['data']:
                    el['selected'] = (el['data']['id'] in new_selection)

            return elements, new_selection