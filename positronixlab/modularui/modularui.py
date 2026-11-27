import bisect
import copy
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from dash import Dash
from flask.helpers import get_root_path
from werkzeug.serving import make_server

from positronixlab.modularui.layouts import HomeLayout
from positronixlab.modularui.settings import (
    BOOTSTRAP_THEME,
    DBC_HELPER_CSS,
    GLOBAL_TRACE_NAME,
    SPACELOSS_FAMILY,
)

logger = logging.getLogger(__name__)


def _to_numpy(value: Any) -> np.ndarray:
    """Best-effort conversion of tensors/array-likes to a plain numpy array."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


# ---------------------------------------------------------
# Immutable Lock Structure
# ---------------------------------------------------------
@dataclass(frozen=True)
class UILocks:
    """Immutable collection of fine-grained locks."""
    vars: threading.Lock = field(default_factory=threading.Lock)          # dynamic_vars / dynamic_vars_version
    config: threading.Lock = field(default_factory=threading.Lock)        # trial_config
    model_graph: threading.Lock = field(default_factory=threading.Lock)   # model_graph (DAG nodes/edges)
    stats: threading.Lock = field(default_factory=threading.Lock)         # space_stats
    metrics: threading.Lock = field(default_factory=threading.Lock)       # training_metrics
    spaceloss: threading.Lock = field(default_factory=threading.Lock)     # layer + overall spaceloss
    # Guards positions_plot, position bookkeeping, pca_progress and cluster_history
    # together, since cluster snapshots must stay consistent with the position timeline.
    positions: threading.Lock = field(default_factory=threading.Lock)
    observed_layers: threading.Lock = field(default_factory=threading.Lock)


class ModularUI:
    """View-side store for the web UI.

    Everything here arrives already aggregated from StatsPlugin (reduced,
    PCA-projected, cluster-synced). This class only keeps a bounded display
    window of it (deque maxlen) and serves it to the Dash callbacks. Full
    history and persistence live in StatsPlugin.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8050,
        debug: bool = False,
        reloader: bool = False,
        poll_interval_ms: int = 3000,
        positions_maxlen: int = 1000,
        metrics_maxlen: int = 2000,
        cluster_maxlen: int = 500,
        open_browser: bool = True,
    ):
        self.host = host
        self.port = port
        self.debug = debug
        self.reloader = reloader
        self.positions_maxlen = positions_maxlen
        self.metrics_maxlen = metrics_maxlen
        self.cluster_maxlen = cluster_maxlen
        self.open_browser = open_browser

        if not self.debug:
            logging.getLogger("werkzeug").setLevel(logging.ERROR)

        self.locks = UILocks()

        # Protected by locks.vars: generic store for env vars discovered by
        # UIPlugin (e.g. "epoch"), with a per-key version counter.
        self.dynamic_vars: dict[str, Any] = {}
        self.dynamic_vars_version: dict[str, int] = {}

        # Protected by locks.config
        self.trial_config: dict[str, Any] = {}

        # Protected by locks.model_graph. {"nodes": {id: node_dict}, "edges": [...]}
        self.model_graph: dict[str, Any] = {"nodes": {}, "edges": []}
        self.model_graph_version: int = 0

        # Protected by locks.stats
        self.stats: Any = {}

        # Protected by locks.observed_layers: {layer_id: {meta...}} or a flat list of ids.
        self.observed_layers: Any = {}

        # Protected by locks.metrics. ONE store for every scalar metric:
        #   family ("train", "val", "spaceloss", ...) -> layer_id ("global" or a layer) -> label -> values
        # Families, layers and labels are all discovered from the incoming messages.
        self.series: dict[str, dict[str, dict[str, deque[float]]]] = {}

        # Protected by locks.positions.
        # positions_plot holds only the last `positions_maxlen` steps per layer;
        # _position_last_index is the ABSOLUTE index (assigned by StatsPlugin) of
        # the newest retained step, so absolute = relative + offset where
        # offset = last_index - len(window) + 1. Cluster snapshots are tagged with
        # absolute indices and converted in get_cluster_for_index.
        self.positions_plot: dict[str, deque[np.ndarray]] = {}
        self._position_indices: dict[str, deque[int]] = {}
        self._position_last_index: dict[str, int] = {}
        self.position_progress: dict[str, Any] = {"required": 0, "layers": {}}
        self.cluster_history: dict[str, deque[dict[str, Any]]] = {}

        self.poll_interval_ms = poll_interval_ms

        # HomeLayout needs `self` up front to construct LayerBoardDisplay(ui=...)
        # and ModelDAGDisplay(ui=...).
        self.home_layout = HomeLayout(ui=self, poll_interval_ms=self.poll_interval_ms)

        self.app = Dash(
            __name__,
            external_stylesheets=[BOOTSTRAP_THEME, DBC_HELPER_CSS] + self.home_layout.css,
            # Panels can be collapsed/rebuilt, so some callback targets legitimately vanish.
            suppress_callback_exceptions=True,
        )
        self.server = None
        self.server_thread = None

        self.reset()

        self.app.layout = self.home_layout.page_layout()
        self.home_layout.register_callbacks(self.app, self)


    # ---------------------------------------------------------
    # Server lifecycle
    # ---------------------------------------------------------

    def run(self):
        if self.debug:
            self.app.enable_dev_tools(debug=True)
            
        self.server = make_server(self.host, self.port, self.app.server, threaded=True)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()

        if self.open_browser:
            import webbrowser
            webbrowser.open(f"http://{self.host}:{self.port}", new=2) 

    def is_running(self):
        return self.server_thread is not None and self.server_thread.is_alive()

    def stop(self):
        if self.server:
            self.server.shutdown()
        
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=5)


    # ---------------------------------------------------------
    # helper functions
    # ---------------------------------------------------------


    def homedir(self):
        return get_root_path(__name__)

    def reset(self):
        with self.locks.vars:
            self.dynamic_vars.clear()
            self.dynamic_vars_version.clear()
        with self.locks.config:
            self.trial_config = {}
        with self.locks.model_graph:
            self.model_graph = {"nodes": {}, "edges": []}
            self.model_graph_version = 0
        with self.locks.stats:
            self.stats = {}
        with self.locks.observed_layers:
            self.observed_layers = {}
        with self.locks.metrics:
            self.series.clear()
        with self.locks.positions:
            self.positions_plot.clear()
            self._position_indices.clear()
            self._position_last_index.clear()
            self.position_progress = {"required": 0, "layers": {}}
            self.cluster_history.clear()

    # ---------------------------------------------------------
    # Generic dynamic-var store
    # ---------------------------------------------------------

    def set_var(self, name: str, value: Any) -> None:
        with self.locks.vars:
            self.dynamic_vars[name] = value
            self.dynamic_vars_version[name] = self.dynamic_vars_version.get(name, 0) + 1

    def get_var(self, name: str, default: Any = None) -> Any:
        with self.locks.vars:
            return self.dynamic_vars.get(name, default)

    def get_var_version(self, name: str) -> int:
        with self.locks.vars:
            return self.dynamic_vars_version.get(name, 0)

    def get_all_vars_snapshot(self) -> dict[str, Any]:
        with self.locks.vars:
            return dict(self.dynamic_vars)

    # ---------------------------------------------------------
    # Dedicated env-var setters / getters
    # ---------------------------------------------------------

    def set_config(self, config: Any) -> None:
        if hasattr(config, "to_dict"):
            config = config.to_dict()
        with self.locks.config:
            self.trial_config = config or {}

    def get_config(self) -> dict[str, Any]:
        with self.locks.config:
            return dict(self.trial_config) if self.trial_config else {}

    def set_model_graph(self, graph: dict[str, Any]) -> None:
        with self.locks.model_graph:
            self.model_graph = graph or {"nodes": {}, "edges": []}
            self.model_graph_version += 1

    def get_model_graph(self) -> dict[str, Any]:
        with self.locks.model_graph:
            return copy.deepcopy(self.model_graph)

    def get_model_graph_version(self) -> int:
        with self.locks.model_graph:
            return self.model_graph_version

    def set_space_stats(self, stats: Any) -> None:
        with self.locks.stats:
            self.stats = stats

    def set_observed_layers(self, observed_layers: Any) -> None:
        with self.locks.observed_layers:
            self.observed_layers = observed_layers

    def set_position_progress(self, progress: dict[str, Any] | None) -> None:
        """{"required": n, "layers": {layer_id: collected}} published by StatsPlugin."""
        with self.locks.positions:
            self.position_progress = progress or {"required": 0, "layers": {}}

    # ---------------------------------------------------------
    # Data ingestion (called with already-aggregated StatsPlugin output)
    # ---------------------------------------------------------

    def add_point(self, family: str, layer_id: str, label: str, value: Any) -> None:
        """One scalar of one metric. layer_id is "global" for non-per-layer metrics."""
        if value is None:
            return
        arr = _to_numpy(value)
        if arr.size != 1:
            return
        try:
            v = float(arr.reshape(-1)[0])
        except (TypeError, ValueError):
            return
        with self.locks.metrics:
            dq = self.series.setdefault(family, {}).setdefault(layer_id, {}).setdefault(
                label, deque(maxlen=self.metrics_maxlen))
            dq.append(v)

    def ingest_stat(self, family: str, msg: dict[str, Any]) -> None:
        """Entry point for a `stats_<family>` message published by UIMetricsPlugin:
        {"layer_id", "label", "value", "seq", "t"}. Call this from the stats subscriber."""
        self.add_point(family, str(msg.get("layer_id", "global")), str(msg["label"]), msg["value"])

    # --- thin compatibility shims for the old call sites -------------------
    def add_training_point(self, name: str, value: Any) -> None:
        family, _, label = name.rpartition(".")
        self.add_point(family or "metrics", "global", label, value)

    def add_spaceloss_point(self, layer_id: str, names: Any, values: Any) -> None:
        if isinstance(values, dict):
            names, values = list(values.keys()), list(values.values())
        for n, v in zip(names, _to_numpy(values).reshape(-1), strict=False):
            self.add_point(SPACELOSS_FAMILY, layer_id, n, v)

    def add_overall_spaceloss_point(self, names: Any, values: Any) -> None:
        self.add_spaceloss_point("global", names, values)

    def add_position_step(self, layer_id: str, points: Any, index: int | None = None) -> None:
        """One already-PCA-projected step (n, 2) with its absolute timeline index."""
        pts = _to_numpy(points)
        with self.locks.positions:
            window = self.positions_plot.get(layer_id)
            if window is None:
                window = deque(maxlen=self.positions_maxlen)
                self.positions_plot[layer_id] = window
            idx_window = self._position_indices.get(layer_id)
            if idx_window is None:
                idx_window = deque(maxlen=self.positions_maxlen)
                self._position_indices[layer_id] = idx_window
            last = self._position_last_index.get(layer_id, -1)
            abs_idx = index if index is not None else last + 1
            window.append(pts)
            idx_window.append(abs_idx)
            self._position_last_index[layer_id] = abs_idx

    def add_cluster(
        self,
        layer_id: str,
        position_index: int,
        labels: Any,
        best_k: Any,
        best_score: Any,
        centroids: Any,
    ) -> None:
        """Snapshot already projected and tagged with an absolute position_index by StatsPlugin."""
        snap = {
            "position_index": position_index,
            "labels": _to_numpy(labels) if labels is not None else None,
            "best_k": best_k,
            "best_score": best_score,
            "centroids": _to_numpy(centroids) if centroids is not None else None,
        }
        with self.locks.positions:
            self.cluster_history.setdefault(layer_id, deque(maxlen=self.cluster_maxlen)).append(snap)

    # ---------------------------------------------------------
    # Accessors used by the widgets (components.py / plots.py / layouts.py)
    # ---------------------------------------------------------

    def get_series(self, family: str, layer_id: str = "global") -> dict[str, list[float]]:
        with self.locks.metrics:
            return {k: list(v) for k, v in self.series.get(family, {}).get(layer_id, {}).items()}

    def get_labels(self, family: str) -> list[str]:
        """Labels seen for a family, in first-seen order (so "loss" stays first)."""
        with self.locks.metrics:
            labels: dict[str, None] = {}
            for per_label in self.series.get(family, {}).values():
                for label in per_label:
                    labels.setdefault(label, None)
            return list(labels)

    @staticmethod
    def _history_scalar(entry: Any) -> float | None:
        """metrics_history entries look like [timestamp, value] (or [value] / value)."""
        try:
            arr = np.asarray(entry, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return None
        if arr.size == 1:
            return float(arr[0])
        if arr.size == 2:                     # [t, value]
            return float(arr[1])
        return None

    def _training_history_groups(self) -> dict[str, dict[str, list[float]]]:
        """Epoch-level train/val/... curves from the `metrics_history` env var that the trainer
        already publishes ({family: {label: [[t, value], ...]}}). Works even when no stats
        message reaches add_point(). Per-layer space loss is skipped (handled by the series store)."""
        history = self.get_var("metrics_history")
        groups: dict[str, dict[str, list[float]]] = {}
        if not isinstance(history, dict):
            return groups
        for family, labels in history.items():
            if family == SPACELOSS_FAMILY or not isinstance(labels, dict):
                continue
            for label, entries in labels.items():
                if not isinstance(entries, (list, tuple)):
                    continue
                values = [v for v in (self._history_scalar(e) for e in entries) if v is not None]
                if values:
                    groups.setdefault(str(label), {})[str(family)] = values
        return groups

    def get_training_groups(self) -> dict[str, dict[str, list[float]]]:
        """{label: {family: values}} for every global-only family except space loss
        (so train.loss and val.loss land on the same chart). Live series win over the
        metrics_history fallback."""
        groups = self._training_history_groups()
        with self.locks.metrics:
            for family, layers in self.series.items():
                if family == SPACELOSS_FAMILY or set(layers) != {"global"}:
                    continue
                for label, values in layers["global"].items():
                    current = groups.setdefault(label, {}).get(family, [])
                    if len(values) >= len(current):      # never let a late-starting live series truncate the history
                        groups[label][family] = list(values)
        return groups

    def debug_state(self) -> dict[str, Any]:
        """Quick look at what actually reached the UI (handy from the debugger)."""
        with self.locks.metrics:
            series = {f: {layer: {lbl: len(v) for lbl, v in labels.items()} for layer, labels in layers.items()}
                      for f, layers in self.series.items()}
        return {"series": series, "vars": list(self.get_all_vars_snapshot()),
                "layers": self.get_known_layer_ids(), "graph_nodes": len(self.get_model_graph().get("nodes", {}))}

    def get_family_groups(self, family: str) -> dict[str, dict[str, list[float]]]:
        """{label: {layer_id: values}} for a per-layer family. The "global" series (mean over
        layers, computed by UIMetricsPlugin) is exposed under GLOBAL_TRACE_NAME."""
        groups: dict[str, dict[str, list[float]]] = {}
        with self.locks.metrics:
            for layer_id, per_label in self.series.get(family, {}).items():
                trace = GLOBAL_TRACE_NAME if layer_id == "global" else layer_id
                for label, values in per_label.items():
                    groups.setdefault(label, {})[trace] = list(values)
        return groups

    @property
    def overall_spaceloss(self) -> dict[str, list[float]]:
        return self.get_series(SPACELOSS_FAMILY, "global")

    def get_layer_spaceloss(self, layer_id: str) -> dict[str, list[float]]:
        return self.get_series(SPACELOSS_FAMILY, layer_id)

    def get_training_metrics(self) -> dict[str, list[float]]:
        """Flat "family.label" view, kept for old callers."""
        return {f"{fam}.{label}": v for label, fams in self.get_training_groups().items() for fam, v in fams.items()}

    def get_observed_layer_ids(self) -> list[str]:
        with self.locks.observed_layers:
            data = self.observed_layers
        if isinstance(data, dict):
            return list(data.keys())
        if isinstance(data, (list, tuple, set)):
            return list(data)
        return []

    def get_observed_layer_info(self, layer_id: str) -> dict[str, Any]:
        with self.locks.observed_layers:
            data = self.observed_layers
        if isinstance(data, dict):
            info = data.get(layer_id, {})
            return info if isinstance(info, dict) else {"value": info}
        return {}

    def get_known_layer_ids(self) -> list[str]:
        """Observed layers plus layers that already have positions or are collecting PCA samples."""
        observed = set(self.get_observed_layer_ids())
        with self.locks.positions:
            seen = set(self.positions_plot.keys()) | set(self.position_progress.get("layers", {}).keys())
        return sorted(observed | seen)

    def get_position_steps(self, layer_id: str) -> int:
        with self.locks.positions:
            return len(self.positions_plot.get(layer_id, ()))

    def _position_offset_locked(self, layer_id: str) -> int:
        idx_window = self._position_indices.get(layer_id)
        if idx_window:
            return idx_window[0]
        window = self.positions_plot.get(layer_id)
        if not window:
            return 0
        return self._position_last_index.get(layer_id, len(window) - 1) - len(window) + 1

    def get_position_offset(self, layer_id: str) -> int:
        """Absolute timeline index of the first retained step (0 until the window starts rolling)."""
        with self.locks.positions:
            return self._position_offset_locked(layer_id)

    def get_position_at(self, layer_id: str, index: int) -> np.ndarray | None:
        """`index` is relative to the retained window."""
        with self.locks.positions:
            steps = self.positions_plot.get(layer_id)
            if not steps or index < 0 or index >= len(steps):
                return None
            return steps[index]

    def get_layer_stats(self, layer_id: str) -> Any | None:
        with self.locks.stats:
            if isinstance(self.stats, dict):
                return self.stats.get(layer_id)
            return None

    def get_pca_progress(self, layer_id: str) -> dict[str, int] | None:
        """None once the layer has position steps, otherwise {"collected", "required"}."""
        with self.locks.positions:
            if self.positions_plot.get(layer_id):
                return None
            return {
                "collected": self.position_progress.get("layers", {}).get(layer_id, 0),
                "required": self.position_progress.get("required", 0),
            }

    def get_cluster_for_index(self, layer_id: str, position_index: int) -> dict[str, Any] | None:
        """Most recent cluster snapshot valid as of the given *relative* position index
        (cluster_history is ordered by absolute index, so a binary search suffices)."""
        with self.locks.positions:
            history = self.cluster_history.get(layer_id)
            if not history:
                return None
            idx_window = self._position_indices.get(layer_id)
            if idx_window and 0 <= position_index < len(idx_window):
                absolute = idx_window[position_index]
            else:
                absolute = position_index + self._position_offset_locked(layer_id)
            idxs = [h["position_index"] for h in history]
            pos = bisect.bisect_right(idxs, absolute) - 1
            if pos < 0:
                return None
            return history[pos]