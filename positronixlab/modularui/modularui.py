import bisect
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

import numpy as np
import torch
from dash import Dash
from flask.helpers import get_root_path

from positronixlab.modularui.layouts import HomeLayout
from positronixlab.modularui.settings import BOOTSTRAP_THEME, DBC_HELPER_CSS

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
    ):
        self.host = host
        self.port = port
        self.debug = debug
        self.reloader = reloader
        self.positions_maxlen = positions_maxlen
        self.metrics_maxlen = metrics_maxlen
        self.cluster_maxlen = cluster_maxlen

        self.locks = UILocks()

        # Protected by locks.vars: generic store for env vars discovered by
        # UIPlugin (e.g. "epoch"), with a per-key version counter.
        self.dynamic_vars: Dict[str, Any] = {}
        self.dynamic_vars_version: Dict[str, int] = {}

        # Protected by locks.config
        self.trial_config: Dict[str, Any] = {}

        # Protected by locks.model_graph. {"nodes": {id: node_dict}, "edges": [...]}
        self.model_graph: Dict[str, Any] = {"nodes": {}, "edges": []}
        self.model_graph_version: int = 0

        # Protected by locks.stats
        self.stats: Any = {}

        # Protected by locks.observed_layers: {layer_id: {meta...}} or a flat list of ids.
        self.observed_layers: Any = {}

        # Protected by locks.metrics
        self.training_metrics: Dict[str, Deque[float]] = {}

        # Protected by locks.spaceloss
        self.layer_spaceloss: Dict[str, Dict[str, Deque[float]]] = {}
        self._overall_spaceloss: Dict[str, Deque[float]] = {}

        # Protected by locks.positions.
        # positions_plot holds only the last `positions_maxlen` steps per layer;
        # _position_last_index is the ABSOLUTE index (assigned by StatsPlugin) of
        # the newest retained step, so absolute = relative + offset where
        # offset = last_index - len(window) + 1. Cluster snapshots are tagged with
        # absolute indices and converted in get_cluster_for_index.
        self.positions_plot: Dict[str, Deque[np.ndarray]] = {}
        self._position_last_index: Dict[str, int] = {}
        self.position_progress: Dict[str, Any] = {"required": 0, "layers": {}}
        self.cluster_history: Dict[str, Deque[Dict[str, Any]]] = {}

        self.poll_interval_ms = poll_interval_ms

        # HomeLayout needs `self` up front to construct LayerBoardDisplay(ui=...)
        # and ModelDAGDisplay(ui=...).
        self.home_layout = HomeLayout(ui=self, poll_interval_ms=self.poll_interval_ms)

        self.app = Dash(
            __name__,
            external_stylesheets=[BOOTSTRAP_THEME, DBC_HELPER_CSS] + self.home_layout.css,
        )
        self.server_thread = None

        self.reset()

        self.app.layout = self.home_layout.page_layout()
        self.home_layout.register_callbacks(self.app, self)

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
            self.training_metrics.clear()
        with self.locks.spaceloss:
            self.layer_spaceloss.clear()
            self._overall_spaceloss.clear()
        with self.locks.positions:
            self.positions_plot.clear()
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

    def get_all_vars_snapshot(self) -> Dict[str, Any]:
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

    def get_config(self) -> Dict[str, Any]:
        with self.locks.config:
            return dict(self.trial_config) if self.trial_config else {}

    def set_model_graph(self, graph: Dict[str, Any]) -> None:
        with self.locks.model_graph:
            self.model_graph = graph or {"nodes": {}, "edges": []}
            self.model_graph_version += 1

    def get_model_graph(self) -> Dict[str, Any]:
        with self.locks.model_graph:
            return self.model_graph

    def get_model_graph_version(self) -> int:
        with self.locks.model_graph:
            return self.model_graph_version

    def set_space_stats(self, stats: Any) -> None:
        with self.locks.stats:
            self.stats = stats

    def set_observed_layers(self, observed_layers: Any) -> None:
        with self.locks.observed_layers:
            self.observed_layers = observed_layers

    def set_position_progress(self, progress: Optional[Dict[str, Any]]) -> None:
        """{"required": n, "layers": {layer_id: collected}} published by StatsPlugin."""
        with self.locks.positions:
            self.position_progress = progress or {"required": 0, "layers": {}}

    # ---------------------------------------------------------
    # Data ingestion (called with already-aggregated StatsPlugin output)
    # ---------------------------------------------------------

    def add_training_point(self, name: str, value: Any) -> None:
        v = float(_to_numpy(value).reshape(-1)[0])
        with self.locks.metrics:
            self.training_metrics.setdefault(name, deque(maxlen=self.metrics_maxlen)).append(v)

    def add_spaceloss_point(self, layer_id: str, names: List[str], values: Any) -> None:
        vals = _to_numpy(values).reshape(-1)
        with self.locks.spaceloss:
            series = self.layer_spaceloss.setdefault(layer_id, {})
            for n, v in zip(names, vals):
                series.setdefault(n, deque(maxlen=self.metrics_maxlen)).append(float(v))

    def add_overall_spaceloss_point(self, names: List[str], values: Any) -> None:
        vals = _to_numpy(values).reshape(-1)
        with self.locks.spaceloss:
            for n, v in zip(names, vals):
                self._overall_spaceloss.setdefault(n, deque(maxlen=self.metrics_maxlen)).append(float(v))

    def add_position_step(self, layer_id: str, points: Any, index: Optional[int] = None) -> None:
        """One already-PCA-projected step (n, 2) with its absolute timeline index."""
        pts = _to_numpy(points)
        with self.locks.positions:
            window = self.positions_plot.get(layer_id)
            if window is None:
                window = deque(maxlen=self.positions_maxlen)
                self.positions_plot[layer_id] = window
            window.append(pts)
            last = self._position_last_index.get(layer_id, -1)
            self._position_last_index[layer_id] = index if index is not None else last + 1

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

    @property
    def overall_spaceloss(self) -> Dict[str, List[float]]:
        with self.locks.spaceloss:
            return {k: list(v) for k, v in self._overall_spaceloss.items()}

    def get_layer_spaceloss(self, layer_id: str) -> Dict[str, List[float]]:
        with self.locks.spaceloss:
            metrics = self.layer_spaceloss.get(layer_id, {})
            return {k: list(v) for k, v in metrics.items()}

    def get_training_metrics(self) -> Dict[str, List[float]]:
        with self.locks.metrics:
            return {k: list(v) for k, v in self.training_metrics.items()}

    def get_observed_layer_ids(self) -> List[str]:
        with self.locks.observed_layers:
            data = self.observed_layers
        if isinstance(data, dict):
            return list(data.keys())
        if isinstance(data, (list, tuple, set)):
            return list(data)
        return []

    def get_observed_layer_info(self, layer_id: str) -> Dict[str, Any]:
        with self.locks.observed_layers:
            data = self.observed_layers
        if isinstance(data, dict):
            info = data.get(layer_id, {})
            return info if isinstance(info, dict) else {"value": info}
        return {}

    def get_known_layer_ids(self) -> List[str]:
        """Observed layers plus layers that already have positions or are collecting PCA samples."""
        observed = set(self.get_observed_layer_ids())
        with self.locks.positions:
            seen = set(self.positions_plot.keys()) | set(self.position_progress.get("layers", {}).keys())
        return sorted(observed | seen)

    def get_position_steps(self, layer_id: str) -> int:
        with self.locks.positions:
            return len(self.positions_plot.get(layer_id, ()))

    def _position_offset_locked(self, layer_id: str) -> int:
        window = self.positions_plot.get(layer_id)
        if not window:
            return 0
        return self._position_last_index.get(layer_id, len(window) - 1) - len(window) + 1

    def get_position_offset(self, layer_id: str) -> int:
        """Absolute timeline index of the first retained step (0 until the window starts rolling)."""
        with self.locks.positions:
            return self._position_offset_locked(layer_id)

    def get_position_at(self, layer_id: str, index: int) -> Optional[np.ndarray]:
        """`index` is relative to the retained window."""
        with self.locks.positions:
            steps = self.positions_plot.get(layer_id)
            if not steps or index < 0 or index >= len(steps):
                return None
            return steps[index]

    def get_layer_stats(self, layer_id: str) -> Optional[Any]:
        with self.locks.stats:
            if isinstance(self.stats, dict):
                return self.stats.get(layer_id)
            return None

    def get_pca_progress(self, layer_id: str) -> Optional[Dict[str, int]]:
        """None once the layer has position steps, otherwise {"collected", "required"}."""
        with self.locks.positions:
            if self.positions_plot.get(layer_id):
                return None
            return {
                "collected": self.position_progress.get("layers", {}).get(layer_id, 0),
                "required": self.position_progress.get("required", 0),
            }

    def get_cluster_for_index(self, layer_id: str, position_index: int) -> Optional[Dict[str, Any]]:
        """Most recent cluster snapshot valid as of the given *relative* position index
        (cluster_history is ordered by absolute index, so a binary search suffices)."""
        with self.locks.positions:
            history = self.cluster_history.get(layer_id)
            if not history:
                return None
            absolute = position_index + self._position_offset_locked(layer_id)
            idxs = [h["position_index"] for h in history]
            pos = bisect.bisect_right(idxs, absolute) - 1
            if pos < 0:
                return None
            return history[pos]

    # ---------------------------------------------------------
    # Server lifecycle
    # ---------------------------------------------------------

    def run(self):
        self.server_thread = threading.Thread(target=self._run, daemon=True)
        self.server_thread.start()

    def _run(self):
        self.app.run(host=self.host, port=self.port, debug=self.debug, use_reloader=self.reloader)

    def is_running(self):
        return self.server_thread is not None and self.server_thread.is_alive()

    def stop(self):
        if self.server_thread and self.server_thread.is_alive():
            import os
            import signal
            os.kill(os.getpid(), signal.SIGINT)
