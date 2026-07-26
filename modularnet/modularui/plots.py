from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
from dash import dcc, html

from modularnet.modularui.components import Widget

# themes some day ? 
PLOTLY_TEMPLATE="plotly_dark"
PLOTLY_PAPER_BGCOLOR="rgba(0,0,0,0)"
PLOTLY_PLOT_BGCOLOR="rgba(0,0,0,0)"

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
    """One scatter plot per layer. Points are colored chronologically by step."""
    
    def __init__(self, css_id: str = "modularnet-positions"):
        self.css_id = css_id

    def render(self, positions: Optional[Dict[str, List[np.ndarray]]] = None):
        positions = positions or {}
        graphs = []
        
        for layer_id, time_steps in sorted(positions.items()):
            if not time_steps:
                continue
                
            fig = go.Figure()
            
            # Flatten lists to plot all time steps at once, coloring by 'step_idx'
            x_vals, y_vals, step_indices = [], [], []
            for step_idx, pts in enumerate(time_steps):
                x_vals.extend(pts[:, 0])
                y_vals.extend(pts[:, 1])
                step_indices.extend([step_idx] * len(pts))
                
            fig.add_trace(go.Scatter(
                x=x_vals, 
                y=y_vals, 
                mode="markers",
                marker=dict(
                    color=step_indices,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Step")
                ),
                name=layer_id
            ))
            
            fig.update_layout(
                title=f"Positions Evolution: {layer_id}",
                margin=dict(l=20, r=20, t=40, b=20),
                height=350,
                template=PLOTLY_TEMPLATE,
                paper_bgcolor=PLOTLY_PAPER_BGCOLOR, 
                plot_bgcolor=PLOTLY_PLOT_BGCOLOR
            )
            graphs.append(dcc.Graph(figure=fig))
            
        return html.Div(graphs, id=self.css_id)