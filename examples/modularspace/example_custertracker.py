"""Dash UI for animating batch-evolving embeddings.

This demo uses the process from the provided script:

* embeddings live in a small 2D plane but are normalized into cosine space
* the 2D state is mapped onto the sphere with spherical angles
* clusters are organized with ``window``, ``graph``, or ``tree`` partitions
* each tick samples activation gates and updates positions with an AdamW step
* the UI shows the current embedding cloud and a compact training summary

Run the file directly to launch the Dash app.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import count

import numpy as np
import plotly.graph_objects as go
import torch
from dash import Dash, Input, Output, State, ctx, dcc, html
from skimage.util import view_as_windows


def group_activation(num_groups: int, activation_pattern: str) -> torch.Tensor:
    weights = None
    if activation_pattern == "always":
        weights = torch.ones(num_groups)
    elif activation_pattern == "uniform":
        weights = torch.rand(num_groups)
    elif activation_pattern == "normal":
        weights = torch.randn(num_groups)
        weights = (weights - weights.min() + 0.1) / (weights.max() - weights.min())
    elif activation_pattern == "skewed":
        weights = torch.rand(num_groups) ** 2

    weights = weights / weights.sum()
    weights *= num_groups
    return weights


def _normalize_rows(tensor: torch.Tensor) -> torch.Tensor:
    return tensor / tensor.norm(dim=1, keepdim=True).clamp_min(1e-8)


def _project_plane_to_sphere(points: np.ndarray) -> np.ndarray:
    """Map a 2D plane to the unit sphere with inverse stereographic projection."""

    plane = points.astype(np.float64)
    x = plane[:, 0]
    y = plane[:, 1]
    denom = 1.0 + x * x + y * y
    sphere = np.column_stack([2.0 * x / denom, 2.0 * y / denom, (x * x + y * y - 1.0) / denom])
    norms = np.linalg.norm(sphere, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return sphere / norms


def _project_uv_to_sphere(points: np.ndarray) -> np.ndarray:
    """Map u,v coordinates to the sphere with spherical angles."""

    uv = np.tanh(points.astype(np.float64) / 2.0)
    theta = np.pi * (uv[:, 0] + 1.0)
    phi = 0.5 * np.pi * (uv[:, 1] + 1.0)
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    sphere = np.column_stack([x, y, z])
    norms = np.linalg.norm(sphere, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return sphere / norms


def _sphere_surface(resolution: int = 40) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(0.0, 2.0 * np.pi, resolution)
    v = np.linspace(0.0, np.pi, resolution)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def _compute_group_centroids(positions: np.ndarray, groups: Sequence[Sequence[int]]) -> tuple[np.ndarray, np.ndarray]:
    centroids = []
    sizes = []
    for group in groups:
        group_positions = positions[np.asarray(group, dtype=np.int64)]
        centroids.append(group_positions.mean(axis=0))
        sizes.append(len(group))
    return np.asarray(centroids, dtype=np.float64), np.asarray(sizes, dtype=np.float64)


def group_partition_window(num_units: int, num_groups: int, overlapping: float, clip_groups: bool = True):
    positions_idxs = np.array(list(range(num_units)))

    group_size = math.ceil(num_units / num_groups)
    overlap_size = math.ceil(group_size * overlapping)
    stride_size = group_size - math.ceil(overlap_size / 4)
    group_size += overlap_size

    groups = view_as_windows(positions_idxs, int(group_size), int(stride_size))
    if clip_groups:
        groups = groups[:num_groups]
    return groups.tolist()


def group_partition_graph(num_units: int, num_groups: int, overlap_size: float, overlap_num: int):
    positions_idxs = np.array(list(range(num_units)))
    groups = np.array_split(positions_idxs, num_groups)

    group_size = math.ceil(num_units / num_groups)
    ols = math.ceil(group_size * overlap_size)
    shared = np.array(groups)[:, -ols:]

    gs = []
    for i, s in enumerate(shared):
        gs.append([i])
        for k in range(overlap_num):
            g = (i + k + 1) % num_groups
            gs[i].append(g)
            ng = np.concatenate([groups[g], s])
            groups[g] = ng
    return groups


class Node:
    def __init__(self, num_units, parent=None) -> None:
        self.children = []
        self.parent = parent
        self.all_nodes = [] if parent is None else parent.all_nodes
        self.counter = count(0) if parent is None else parent.counter
        self.depth = 0 if parent is None else parent.depth + 1
        self.data = [next(self.counter) for _ in range(num_units)]
        if parent is not None:
            self.data = parent.data + self.data
        self.all_nodes.append(self)


def make_tree(depth, num_branch, num_units, parent=None):
    if depth == 0:
        return None

    node = Node(num_units, parent)
    for _ in range(num_branch):
        child = make_tree(depth - 1, num_branch, num_units, node)
        node.children.append(child)
    return node


def group_partition_tree(depth, num_branch, num_units, leafs_only=True):
    root = make_tree(depth, num_branch, num_units)
    groups = [node.data for node in root.all_nodes if (node.depth == (depth - 1) or not leafs_only)]
    return groups


def update_positions(optim, good, bad):
    if good.shape[0] < 2 or bad.shape[0] < 1:
        optim.zero_grad(set_to_none=True)
        return 0.0

    avg_good = torch.mean(good, dim=0)
    avg_bad = torch.mean(bad, dim=0)

    diff_good = good - avg_good
    dist_good = diff_good.pow(2).sum(dim=1).sqrt()
    dist_min = dist_good.pow(2).sum()

    diff_bad = bad - avg_good
    dist_bad = diff_bad.pow(2).sum(dim=1).sqrt()
    dist_max = torch.log2(1 + dist_bad).sum()

    distances = torch.cdist(good, good).pow(2)
    mask = torch.eye(good.shape[0], device=good.device).bool()
    distances = distances.masked_fill(mask, float("inf"))
    overlap_loss = torch.relu(0.1 - distances).sum()

    center_diff = (avg_good - avg_bad.detach()).pow(2).sum()
    center_loss = torch.log2(1 + center_diff)

    loss = dist_min - dist_max * 10 - center_loss + 0.02 * overlap_loss
    loss.backward()
    optim.step()
    optim.zero_grad(set_to_none=True)
    return float(loss.detach().item())


@dataclass
class ProcessConfig:
    space_size: int = 2
    activation_pattern: str = "always"
    group_pattern: str = "tree"
    steps: int = 500
    lr_steps: Sequence[int] = (300,)
    num_units: int = 100
    num_groups: int = 10
    overlap_size: float = 0.2
    num_shared: int = 4
    tree_depth: int = 3
    tree_num_branch: int = 3
    tree_num_units: int = 5
    tree_leafs_only: bool = True
    lr: float = 0.2
    seed: int = 7


class BatchEmbeddingProcess:
    def __init__(self, config: ProcessConfig, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)
        self.reset()

    def reset(self) -> None:
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        self.step_index = 0
        self.loss_history: list[float] = []
        self.activation_history: list[np.ndarray] = []

        self._build_groups()
        self.group_weights = group_activation(len(self.groups), self.config.activation_pattern).to(self.device)

        self.positions = (torch.rand((self.config.num_units, self.config.space_size), device=self.device) * 2.0 - 1.0) * 2.0
        self.positions.requires_grad = True
        self.optim = torch.optim.AdamW([self.positions], lr=self.config.lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=list(self.config.lr_steps), gamma=0.5)

        self.unit_labels = self._build_unit_labels()
        self.last_loss = 0.0
        self.last_acts = np.zeros(len(self.groups), dtype=np.float32)
        self.last_active_groups: list[int] = []

    def _build_groups(self) -> None:
        if self.config.group_pattern == "window":
            self.groups = group_partition_window(self.config.num_units, self.config.num_groups, self.config.overlap_size)
        elif self.config.group_pattern == "graph":
            self.groups = group_partition_graph(self.config.num_units, self.config.num_groups, self.config.overlap_size, self.config.num_shared)
        else:
            self.groups = group_partition_tree(self.config.tree_depth, self.config.tree_num_branch, self.config.tree_num_units, self.config.tree_leafs_only)
            all_groups = []
            for group in self.groups:
                all_groups += group
            self.config.num_units = len(set(all_groups))
            self.config.num_groups = len(self.groups)

    def _build_unit_labels(self) -> np.ndarray:
        labels = np.full(self.config.num_units, -1, dtype=np.int32)
        for group_index, group in enumerate(self.groups):
            for unit in group:
                if 0 <= unit < len(labels) and labels[unit] == -1:
                    labels[unit] = group_index
        missing = np.where(labels < 0)[0]
        for unit in missing:
            labels[unit] = unit % max(len(self.groups), 1)
        return labels

    def step(self) -> dict:
        acts = torch.rand((len(self.groups)), device=self.device)
        active_groups = []
        losses = []

        for group_index, group in enumerate(self.groups):
            if acts[group_index] > self.group_weights[group_index]:
                continue
            idxs = torch.zeros((self.config.num_units,), dtype=torch.bool, device=self.device)
            idxs[group] = True
            good = self.positions[idxs]
            bad = self.positions[~idxs]
            loss = update_positions(self.optim, good, bad)
            active_groups.append(group_index)
            losses.append(loss)

        self.scheduler.step()
        self.step_index += 1

        if losses:
            self.last_loss = float(np.mean(losses))
        self.loss_history.append(self.last_loss)
        self.activation_history.append(acts.detach().cpu().numpy())
        self.last_acts = acts.detach().cpu().numpy()
        self.last_active_groups = active_groups

        with torch.no_grad():
            self.positions.clamp_(-2.5, 2.5)

        return {
            "step": self.step_index,
            "loss": self.last_loss,
            "active_groups": active_groups,
            "acts": self.last_acts,
            "positions": self.positions.detach().cpu().numpy(),
            "labels": self.unit_labels,
        }


def _figure_embedding(positions: np.ndarray, labels: np.ndarray) -> go.Figure:
    colors = [
        "#6EE7B7",
        "#60A5FA",
        "#F472B6",
        "#FBBF24",
        "#A78BFA",
        "#34D399",
        "#F87171",
        "#22D3EE",
        "#C084FC",
        "#F59E0B",
    ]
    point_colors = [colors[int(label) % len(colors)] for label in labels]

    sphere = _project_plane_to_sphere(positions)
    surface_x, surface_y, surface_z = _sphere_surface()

    figure = go.Figure()
    figure.add_trace(
        go.Surface(
            x=surface_x,
            y=surface_y,
            z=surface_z,
            opacity=0.08,
            showscale=False,
            colorscale=[[0, "#1E293B"], [1, "#334155"]],
            hoverinfo="skip",
            name="sphere",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=sphere[:, 0],
            y=sphere[:, 1],
            z=sphere[:, 2],
            mode="markers",
            marker=dict(size=6, color=point_colors, opacity=0.92, line=dict(width=0.4, color="#0F172A")),
            text=[f"unit {i}<br>group {int(labels[i])}" for i in range(len(labels))],
            hovertemplate="%{text}<extra></extra>",
            name="embeddings",
        )
    )
    figure.update_layout(
        template="plotly_dark",
        title="Underlying embeddings projected onto a sphere",
        margin=dict(l=12, r=12, t=50, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-1.1, 1.1]),
            yaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-1.1, 1.1]),
            zaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-1.1, 1.1]),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.7, y=1.7, z=1.2)),
        ),
    )
    return figure


def _figure_centroids(positions: np.ndarray, groups: Sequence[Sequence[int]]) -> go.Figure:
    colors = [
        "#F59E0B",
        "#34D399",
        "#60A5FA",
        "#F472B6",
        "#A78BFA",
        "#22D3EE",
        "#F87171",
        "#C084FC",
        "#6EE7B7",
        "#FBBF24",
    ]
    centroids, sizes = _compute_group_centroids(positions, groups)
    sphere = _project_plane_to_sphere(centroids)
    surface_x, surface_y, surface_z = _sphere_surface()
    marker_sizes = np.clip(6.0 + sizes * 1.25, 8.0, 30.0)

    figure = go.Figure()
    figure.add_trace(
        go.Surface(
            x=surface_x,
            y=surface_y,
            z=surface_z,
            opacity=0.08,
            showscale=False,
            colorscale=[[0, "#1E293B"], [1, "#334155"]],
            hoverinfo="skip",
            name="sphere",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=sphere[:, 0],
            y=sphere[:, 1],
            z=sphere[:, 2],
            mode="markers+text",
            marker=dict(size=marker_sizes.tolist(), color=colors * ((len(centroids) + len(colors) - 1) // len(colors)), opacity=0.92, line=dict(width=0.4, color="#0F172A")),
            text=[str(i) for i in range(len(centroids))],
            textposition="top center",
            hovertemplate="group %{text}<br>size=%{marker.size}<extra></extra>",
            name="centroids",
        )
    )
    figure.update_layout(
        template="plotly_dark",
        title="Group centroids in regular space",
        margin=dict(l=12, r=12, t=50, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-1.1, 1.1]),
            yaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-1.1, 1.1]),
            zaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-1.1, 1.1]),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.5, y=1.4, z=1.1)),
        ),
    )
    return figure


def _figure_uv(positions: np.ndarray, labels: np.ndarray) -> go.Figure:
    colors = [
        "#93C5FD",
        "#FCD34D",
        "#FDBA74",
        "#A7F3D0",
        "#F9A8D4",
        "#C4B5FD",
        "#67E8F9",
        "#FCA5A5",
        "#DDD6FE",
        "#86EFAC",
    ]
    point_colors = [colors[int(label) % len(colors)] for label in labels]
    sphere = _project_uv_to_sphere(positions)
    surface_x, surface_y, surface_z = _sphere_surface()

    figure = go.Figure()
    figure.add_trace(
        go.Surface(
            x=surface_x,
            y=surface_y,
            z=surface_z,
            opacity=0.08,
            showscale=False,
            colorscale=[[0, "#1E293B"], [1, "#334155"]],
            hoverinfo="skip",
            name="sphere",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=sphere[:, 0],
            y=sphere[:, 1],
            z=sphere[:, 2],
            mode="markers",
            marker=dict(size=5, color=point_colors, opacity=0.92, line=dict(width=0.4, color="#0F172A")),
            text=[f"u={positions[i, 0]:.3f}<br>v={positions[i, 1]:.3f}<br>group={int(labels[i])}" for i in range(len(labels))],
            hovertemplate="%{text}<extra></extra>",
            name="u,v",
        )
    )
    figure.update_layout(
        template="plotly_dark",
        title="u,v coordinates on the sphere",
        margin=dict(l=12, r=12, t=50, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-1.1, 1.1]),
            yaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-1.1, 1.1]),
            zaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-1.1, 1.1]),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.65, y=1.45, z=1.15)),
        ),
    )
    return figure


def _figure_metrics(loss_history: Sequence[float], acts: np.ndarray, group_weights: torch.Tensor) -> go.Figure:
    x = list(range(1, len(loss_history) + 1))
    if len(acts) == 0:
        acts = np.zeros(len(group_weights), dtype=np.float32)

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=x,
            y=list(loss_history),
            mode="lines+markers",
            line=dict(color="#60A5FA", width=2),
            marker=dict(size=4),
            name="loss",
        )
    )
    figure.add_trace(
        go.Bar(
            x=[f"g{i}" for i in range(len(acts))],
            y=list(acts),
            marker_color="#F472B6",
            opacity=0.75,
            name="activation gate",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[f"g{i}" for i in range(len(group_weights))],
            y=group_weights.detach().cpu().numpy().tolist(),
            mode="lines+markers",
            line=dict(color="#34D399", width=2),
            marker=dict(size=6),
            name="group weights",
            yaxis="y2",
        )
    )
    figure.update_layout(
        template="plotly_dark",
        title="Training signals",
        margin=dict(l=12, r=12, t=50, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(title="loss", showgrid=False),
        yaxis2=dict(title="weights", overlaying="y", side="right", showgrid=False),
        barmode="overlay",
        legend=dict(orientation="h"),
    )
    return figure


def _panel_style() -> dict:
    return {
        "padding": "0.9rem 1rem",
        "border": "1px solid rgba(148,163,184,0.18)",
        "borderRadius": "16px",
        "background": "rgba(15, 23, 42, 0.72)",
        "boxShadow": "0 18px 45px rgba(15, 23, 42, 0.28)",
    }


def _button_style(color: str) -> dict:
    return {
        "background": color,
        "border": "0",
        "color": "white",
        "padding": "0.7rem 1.1rem",
        "borderRadius": "999px",
        "fontWeight": 700,
        "cursor": "pointer",
        "boxShadow": "0 10px 24px rgba(15, 23, 42, 0.22)",
    }


def create_app(seed: int = 7) -> Dash:
    demo = BatchEmbeddingProcess(ProcessConfig(seed=seed), device="cpu")
    app = Dash(__name__)

    app.layout = html.Div(
        [
            dcc.Store(
                id="demo-state",
                data={"running": True, "step": 0, "group_pattern": demo.config.group_pattern, "activation_pattern": demo.config.activation_pattern},
            ),
            dcc.Interval(id="tick", interval=450, n_intervals=0, disabled=False),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Batch Embedding Process", style={"fontSize": "2.3rem", "fontWeight": "700"}),
                            html.Div(
                                "Dash visualization of the embedding update loop from your process.",
                                style={"color": "#94A3B8", "marginTop": "0.35rem"},
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Button("Pause", id="toggle-button", n_clicks=0, style=_button_style("#2563EB")),
                            html.Button("Reset", id="reset-button", n_clicks=0, style=_button_style("#334155")),
                        ],
                        style={"display": "flex", "gap": "0.75rem", "alignItems": "center"},
                    ),
                ],
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "gap": "1rem", "paddingBottom": "1rem"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Group pattern", style={"fontSize": "0.85rem", "color": "#94A3B8", "marginBottom": "0.35rem"}),
                            dcc.Dropdown(
                                id="group-pattern",
                                options=[
                                    {"label": "window", "value": "window"},
                                    {"label": "graph", "value": "graph"},
                                    {"label": "tree", "value": "tree"},
                                ],
                                value=demo.config.group_pattern,
                                clearable=False,
                                style={"color": "#0F172A"},
                            ),
                        ],
                        style={"flex": 1, "minWidth": "180px"},
                    ),
                    html.Div(
                        [
                            html.Div("Activation pattern", style={"fontSize": "0.85rem", "color": "#94A3B8", "marginBottom": "0.35rem"}),
                            dcc.Dropdown(
                                id="activation-pattern",
                                options=[
                                    {"label": "always", "value": "always"},
                                    {"label": "uniform", "value": "uniform"},
                                    {"label": "normal", "value": "normal"},
                                    {"label": "skewed", "value": "skewed"},
                                ],
                                value=demo.config.activation_pattern,
                                clearable=False,
                                style={"color": "#0F172A"},
                            ),
                        ],
                        style={"flex": 1, "minWidth": "180px"},
                    ),
                    html.Div(id="status-line", style={"flex": 1.8, "color": "#E2E8F0", "alignSelf": "end"}),
                ],
                style={**_panel_style(), "display": "flex", "gap": "1rem", "alignItems": "end", "marginBottom": "1rem"},
            ),
            html.Div(
                [
                    dcc.Graph(id="embedding-graph", style={"height": "560px"}, config={"displayModeBar": False}),
                    dcc.Graph(id="centroid-graph", style={"height": "560px"}, config={"displayModeBar": False}),
                    dcc.Graph(id="uv-graph", style={"height": "560px", "gridColumn": "1 / -1"}, config={"displayModeBar": False}),
                ],
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "1rem"},
            ),
            html.Div(id="event-line", style={**_panel_style(), "marginTop": "1rem", "color": "#94A3B8"}),
        ],
        style={
            "minHeight": "100vh",
            "padding": "24px",
            "background": "radial-gradient(circle at top left, #1E293B 0%, #0F172A 42%, #020617 100%)",
            "fontFamily": "ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        },
    )

    @app.callback(
        Output("demo-state", "data"),
        Output("tick", "disabled"),
        Output("toggle-button", "children"),
        Output("embedding-graph", "figure"),
        Output("centroid-graph", "figure"),
        Output("uv-graph", "figure"),
        Output("event-line", "children"),
        Output("status-line", "children"),
        Input("tick", "n_intervals"),
        Input("toggle-button", "n_clicks"),
        Input("reset-button", "n_clicks"),
        Input("group-pattern", "value"),
        Input("activation-pattern", "value"),
        State("demo-state", "data"),
        prevent_initial_call=False,
    )
    def update_dashboard(_tick, _toggle_clicks, _reset_clicks, group_pattern, activation_pattern, state):
        if state is None:
            state = {"running": True, "step": 0, "group_pattern": demo.config.group_pattern, "activation_pattern": demo.config.activation_pattern}

        triggered = ctx.triggered_id
        needs_reset = False

        if state.get("group_pattern") != group_pattern or state.get("activation_pattern") != activation_pattern:
            needs_reset = True

        if triggered == "reset-button":
            needs_reset = True
            state["running"] = True
        elif triggered == "toggle-button":
            state["running"] = not bool(state.get("running", True))

        if needs_reset:
            demo.config.group_pattern = group_pattern
            demo.config.activation_pattern = activation_pattern
            demo.reset()
            state = {"running": True, "step": 0, "group_pattern": group_pattern, "activation_pattern": activation_pattern}

        running = bool(state.get("running", True))
        status = (
            f"running={running} | step={demo.step_index} | groups={len(demo.groups)} | "
            f"pattern={demo.config.group_pattern} | activation={demo.config.activation_pattern} | loss={demo.last_loss:.4f}"
        )

        if triggered == "tick" and running and not needs_reset:
            result = demo.step()
            state = {**state, "step": result["step"]}
            embedding_figure = _figure_embedding(result["positions"], result["labels"])
            centroid_figure = _figure_centroids(result["positions"], demo.groups)
            uv_figure = _figure_uv(result["positions"], result["labels"])
            event_text = f"active groups: {result['active_groups']} | batch loss: {result['loss']:.4f} | lr={demo.scheduler.get_last_lr()[0]:.4f}"
            button_label = "Pause" if running else "Resume"
            return state, not running, button_label, embedding_figure, centroid_figure, uv_figure, event_text, status

        embedding_figure = _figure_embedding(demo.positions.detach().cpu().numpy(), demo.unit_labels)
        centroid_figure = _figure_centroids(demo.positions.detach().cpu().numpy(), demo.groups)
        uv_figure = _figure_uv(demo.positions.detach().cpu().numpy(), demo.unit_labels)
        event_text = (
            f"last active groups: {demo.last_active_groups} | "
            f"current batch loss: {demo.last_loss:.4f} | lr={demo.scheduler.get_last_lr()[0]:.4f}"
        )
        button_label = "Pause" if running else "Resume"
        return state, not running, button_label, embedding_figure, centroid_figure, uv_figure, event_text, status

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dash embedding demo with batch-style training dynamics.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Dash server host.")
    parser.add_argument("--port", type=int, default=8050, help="Dash server port.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(seed=args.seed)
    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()
