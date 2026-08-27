import bisect
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import dash_bootstrap_components as dbc
import numpy as np
import torch
from dash import Dash
from flask.helpers import get_root_path
from sklearn.decomposition import PCA

from modularnet.modularui.layouts import HomeLayout

from modularnet.modularui.settings import BOOTSTRAP_THEME
from modularnet.utils.utils import save_data

logger = logging.getLogger(__name__)

# Bootswatch dark theme -- must match plots.BOOTSTRAP_TEMPLATE_NAME (e.g.
# CYBORG <-> "cyborg") so the Plotly figures and the surrounding Dash chrome
# use the same palette. Other dark Bootswatch options: SUPERHERO, VAPOR,
# SOLAR, DARKLY -- swap this one constant to change the whole app's theme.


# Styles plain dcc components (Checklist, Graph, etc.) to match the dbc
# theme when their container has className="dbc" (see HomeLayout). Without
# this, only actual dbc.* components would pick up the theme.
DBC_HELPER_CSS = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V1.0.2/dbc.min.css"


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
    model: threading.Lock = field(default_factory=threading.Lock)         # model_summary (legacy; nothing currently sets this)
    model_graph: threading.Lock = field(default_factory=threading.Lock)   # model_graph (DAG nodes/edges)
    stats: threading.Lock = field(default_factory=threading.Lock)
    metrics: threading.Lock = field(default_factory=threading.Lock)
    spaceloss: threading.Lock = field(default_factory=threading.Lock)
    # Guards positions_plot, pca, pca_history, cluster_history and the pending
    # cluster buffer together, since cluster snapshots must stay consistent
    # with the position-index timeline they're synced against.
    positions: threading.Lock = field(default_factory=threading.Lock)
    observed_layers: threading.Lock = field(default_factory=threading.Lock)


class ModularUI:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8050,
        debug: bool = False,
        reloader: bool = False,
        save_path: str = None,
        poll_interval_ms: int = 3000,
    ):
        self.host = host
        self.port = port
        self.debug = debug
        self.reloader = reloader
        self.save_path = save_path
        self.pca_dim = 2
        self.pca_samples = 100

        # Initialize the frozen dataclass.
        # The default factories create a new Lock for each field automatically.
        self.locks = UILocks()

        # Protected by locks.vars. Generic store for env vars discovered by
        # UIPlugin's key-discovery loop that don't have a dedicated handler
        # (e.g. "epoch"). Each key has its own version counter so a Dash
        # callback can tell "did *this* var change" without redrawing on
        # every poll tick regardless of that var's actual update rate.
        self.dynamic_vars: Dict[str, Any] = {}
        self.dynamic_vars_version: Dict[str, int] = {}

        # Protected by locks.config / locks.model
        self.trial_config: Dict[str, Any] = {}
        self.model_summary: Any = None  # legacy; no dedicated env var currently populates this

        # Protected by locks.model_graph. Shape: {"nodes": {id: node_dict},
        # "edges": [{"source":..., "target":...}, ...]} -- exactly
        # GraphFormatter.to_dict(graph)'s output (modelmapper.py), published
        # on the "model_graph" env var by ObservableTrainPlugin.on_graph_update.
        self.model_graph: Dict[str, Any] = {"nodes": {}, "edges": []}
        self.model_graph_version: int = 0

        self.stats: Dict[str, np.ndarray] = {}

        # Protected by locks.observed_layers
        # Metadata describing which layers are being observed. Expected shape is
        # either {layer_id: {meta...}} or a flat list/set of layer_id strings;
        # both are supported by get_observed_layer_ids/get_observed_layer_info.
        # In practice this is now GraphFormatter.nodes_dict(graph): {id: node_dict}.
        self.observed_layers: Any = {}

        # Protected by locks.metrics
        self.training_metrics: Dict[str, List[float]] = {}

        # Protected by locks.spaceloss
        self.layer_spaceloss: Dict[str, Dict[str, List[float]]] = {}

        # Protected by locks.positions
        self.positions_plot: Dict[str, List[np.ndarray]] = {}
        self.pca: Dict[str, PCA] = {}
        self.pca_history: Dict[str, List[np.ndarray]] = {}

        # Decimation front-end for add_positions(). Guarded by its own
        # cheap _ema_lock, deliberately separate from locks.positions:
        # the producer can be publishing at very high frequency (e.g.
        # once per training step) while the UI only needs to actually
        # commit a frame at ~poll rate. Filtering (EMA) before subsampling
        # avoids aliasing what a hard time-gate would just silently drop.
        # This is a pure visualization decimation -- add_positions still
        # receives every message, "positions" as observed by other
        # subscribers on the pub/sub topic is untouched.
        self._position_ema: Dict[str, np.ndarray] = {}
        self._ema_lock = threading.Lock()
        self.ema_alpha: float = 0.3
        self._last_flush_ts: Dict[str, float] = {}
        self.flush_interval_s: float = 0.1  # ~10Hz;

        # Protected by locks.positions
        # Cluster snapshots per layer, each tagged with the position_index (index
        # into positions_plot[layer_id]) that was current when the snapshot
        # arrived. Clustering runs at a lower rate than position updates, so
        # this list is sparse relative to positions_plot; get_cluster_for_index
        # resolves "the cluster state valid as of position step N".
        self.cluster_history: Dict[str, List[Dict[str, Any]]] = {}

        # Protected by locks.positions
        # Cluster messages that arrived before PCA was fit for their layer.
        # Drained (and transformed) the moment PCA becomes available.
        self._cluster_pending: Dict[str, List[Dict[str, Any]]] = {}

        self.poll_interval_ms = poll_interval_ms

        # `self` is already a usable object at this point (locks, positions_plot,
        # etc. are all set above), so it's safe to hand it to HomeLayout now —
        # HomeLayout needs it up front to construct LayerBoardDisplay(ui=...)
        # and ModelDAGDisplay(ui=...).
        self.home_layout = HomeLayout(ui=self, poll_interval_ms=self.poll_interval_ms)

        self.app = Dash(
            __name__,
            # Bootswatch theme + dbc helper CSS load first; home_layout.css
            # (the project's own themes/{theme}.css, incl. the toggle-switch
            # styling) loads after, so it can still override where needed.
            external_stylesheets=[BOOTSTRAP_THEME, DBC_HELPER_CSS] + self.home_layout.css,
        )
        self.server_thread = None

        self.reset()

        # Layout + callbacks are set up once.
        self.app.layout = self.home_layout.page_layout()
        self.home_layout.register_callbacks(self.app, self)

    def homedir(self):
        return get_root_path(__name__)

    @property
    def overall_spaceloss(self) -> Dict[str, List[float]]:
        """Dynamically computes the average space loss across all layers at each timestep."""
        with self.locks.spaceloss:
            if not self.layer_spaceloss:
                return {}

            # Find all unique metric names (e.g., 'loss', 'loss_on', 'loss_off')
            metric_names = set()
            for metrics in self.layer_spaceloss.values():
                metric_names.update(metrics.keys())

            overall = {}
            for name in metric_names:
                layer_lists = [
                    metrics[name]
                    for metrics in self.layer_spaceloss.values()
                    if name in metrics
                ]

                if not layer_lists:
                    continue

                min_len = min(len(l) for l in layer_lists)

                avg_values = []
                for i in range(min_len):
                    avg = sum(l[i] for l in layer_lists) / len(layer_lists)
                    avg_values.append(float(avg))

                overall[name] = avg_values

            return overall

    def save_state(self) -> bool:
        if self.save_path is None:
            return False

        # Sequential lock acquisition copies data quickly to avoid deadlocks/blocking
        with self.locks.vars:
            dynamic_vars = dict(self.dynamic_vars)
        with self.locks.config:
            trial_config = dict(self.trial_config) if self.trial_config else {}
        with self.locks.model:
            model_summary = self.model_summary
        with self.locks.model_graph:
            model_graph = dict(self.model_graph)
        with self.locks.stats:
            stats = dict(self.stats)
        with self.locks.observed_layers:
            observed_layers = (
                dict(self.observed_layers)
                if isinstance(self.observed_layers, dict)
                else list(self.observed_layers)
            )
        with self.locks.metrics:
            training_metrics = dict(self.training_metrics)
        with self.locks.spaceloss:
            layer_spaceloss = dict(self.layer_spaceloss)
        with self.locks.positions:
            positions_plot = dict(self.positions_plot)
            pca_history = dict(self.pca_history)
            cluster_history = dict(self.cluster_history)

        # Re-inject config and model summary into dynamic_vars for backwards
        # compatibility in saved data (they used to live in "all_vars").
        if trial_config:
            dynamic_vars["trial_config"] = trial_config
        if model_summary is not None:
            dynamic_vars["model_summary"] = model_summary

        data = {
            "dynamic_vars": dynamic_vars,
            "model_graph": model_graph,
            "stats": stats,
            "observed_layers": observed_layers,
            "training_metrics": training_metrics,
            "layer_spaceloss": layer_spaceloss,
            "positions_plot": positions_plot,
            "pca_history": pca_history,
            "cluster_history": cluster_history,
        }

        payload = {"header": {"timestamp": time.time()}, "data": data}
        save_data(self.save_path, payload)
        logger.info(f"UIPlugin state saved to {self.save_path}")
        return True

    def reset(self):
        with self.locks.vars:
            self.dynamic_vars.clear()
            self.dynamic_vars_version.clear()
        with self.locks.config:
            self.trial_config = {}
        with self.locks.model:
            self.model_summary = None
        with self.locks.model_graph:
            self.model_graph = {"nodes": {}, "edges": []}
            self.model_graph_version = 0
        with self.locks.stats:
            self.stats.clear()
        with self.locks.observed_layers:
            self.observed_layers = {}
        with self.locks.metrics:
            self.training_metrics.clear()
        with self.locks.spaceloss:
            self.layer_spaceloss.clear()
        with self.locks.positions:
            self.positions_plot.clear()
            self.pca_history.clear()
            self.pca.clear()
            self.cluster_history.clear()
            self._cluster_pending.clear()
        with self._ema_lock:
            self._position_ema.clear()
            self._last_flush_ts.clear()

    # ---------------------------------------------------------
    # Generic dynamic-var store (poll-discovery -> per-var subscribe)
    # ---------------------------------------------------------

    def set_var(self, name: str, value: Any) -> None:
        """Generic setter for a dynamically-discovered env var, called from
        a per-var env_subscribe callback wired up by UIPlugin's key-discovery
        loop (see UIPlugin.main / _make_generic_var_handler). Bumps a
        per-key version counter -- not a single global "something changed"
        flag -- so each var's UI refresh can, in principle, be judged on its
        own update cadence rather than the slowest one in the batch."""
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

    def set_config(self, config: Any) -> None:
        """Direct setter for trial_config, meant to be called from an
        env_subscribe("trial_config", ...) callback (via env_get) rather
        than through the old update_vars/all_vars path."""
        if hasattr(config, "to_dict"):
            config = config.to_dict()
        with self.locks.config:
            self.trial_config = config or {}

    def get_config(self) -> Dict[str, Any]:
        with self.locks.config:
            return dict(self.trial_config) if self.trial_config else {}

    def set_model_graph(self, graph: Dict[str, Any]) -> None:
        """Direct setter for model_graph (GraphFormatter.to_dict(...) shape:
        {"nodes": {...}, "edges": [...]}), meant to be called from an
        env_subscribe("model_graph", ...) callback."""
        with self.locks.model_graph:
            self.model_graph = graph or {"nodes": {}, "edges": []}
            self.model_graph_version += 1

    def get_model_graph(self) -> Dict[str, Any]:
        with self.locks.model_graph:
            return self.model_graph

    def get_model_graph_version(self) -> int:
        with self.locks.model_graph:
            return self.model_graph_version

    def set_space_stats(self, stats: Dict[str, Any]) -> None:
        """Direct setter for space_stats, meant to be called from an
        env_subscribe("space_stats", ...) callback (via env_get) rather than
        through update_vars/all_vars."""
        with self.locks.stats:
            self.stats = stats

    def set_observed_layers(self, observed_layers: Any) -> None:
        """Direct setter for observed_layers, meant to be called from an
        env_subscribe("observed_layers", ...) callback (via env_get) rather
        than through update_vars/all_vars."""
        with self.locks.observed_layers:
            self.observed_layers = observed_layers

    def get_layer_spaceloss(self, layer_id: str) -> Dict[str, List[float]]:
        with self.locks.spaceloss:
            metrics = self.layer_spaceloss.get(layer_id, {})
            return {k: list(v) for k, v in metrics.items()}

    def get_training_metrics(self) -> Dict[str, List[float]]:
        with self.locks.metrics:
            return {k: list(v) for k, v in self.training_metrics.items()}

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

    # ---------------------------------------------------------
    # Chart Data Ingestion Methods
    # ---------------------------------------------------------

    def add_metric_training(self, metrics: Dict[str, float]) -> None:
        with self.locks.metrics:
            for name, value in metrics.items():
                self.training_metrics.setdefault(name, []).append(float(value))

    def add_metric_spaceloss(self, layer_id: str, metrics: List[float]) -> None:
        with self.locks.spaceloss:
            loss, loss_on, loss_off, dict_center = metrics
            self.layer_spaceloss.setdefault(layer_id, {}).setdefault("loss", []).append(float(loss))
            self.layer_spaceloss.setdefault(layer_id, {}).setdefault("loss_on", []).append(float(loss_on))
            self.layer_spaceloss.setdefault(layer_id, {}).setdefault("loss_off", []).append(float(loss_off))
            self.layer_spaceloss.setdefault(layer_id, {}).setdefault("dict_center", []).append(float(dict_center))

    def add_positions(self, layer_id: str, positions: np.ndarray) -> None:
        """Entry point for the (possibly very high frequency, e.g. one call
        per training step) "positions" subscription. This is a decimation
        front-end: every call cheaply folds into a per-layer EMA under
        _ema_lock (never locks.positions, so it can't contend with or
        starve the Dash reader thread), and only every flush_interval_s
        does the accumulated, filtered snapshot actually get committed via
        _commit_positions -- which is the only place touching locks.positions
        or doing PCA work. Filtering before subsampling (rather than a bare
        time-gate) avoids throwing away signal between flushes.

        This only affects what ModularUI renders; other "positions"
        subscribers on the pub/sub topic still see every raw message.
        """
        positions = _to_numpy(positions)

        with self._ema_lock:
            prev = self._position_ema.get(layer_id)
            if prev is None or prev.shape != positions.shape:
                # first sample for this layer, or shape changed (e.g. layer
                # resized) -- can't blend, so just reset the accumulator.
                self._position_ema[layer_id] = positions.astype(np.float64, copy=True)
            else:
                prev *= (1.0 - self.ema_alpha)
                prev += self.ema_alpha * positions

            now = time.monotonic()
            should_flush = (now - self._last_flush_ts.get(layer_id, 0.0)) >= self.flush_interval_s
            if not should_flush:
                return
            self._last_flush_ts[layer_id] = now
            snapshot = self._position_ema[layer_id].copy()

        self._commit_positions(layer_id, snapshot)

    def _commit_positions(self, layer_id: str, positions: np.ndarray) -> None:
        """Heavy path: PCA fit/transform + the actual positions_plot /
        cluster_history mutation. Called at the decimated rate only (from
        add_positions()), so locks.positions -- and any PCA.fit() that
        happens to land on this call -- is now acquired at ~1/flush_interval_s
        instead of once per raw producer message. That's what actually
        relieves the reader-thread starvation; the critical section itself
        is unchanged."""
        with self.locks.positions:
            pca = self.pca.get(layer_id)
            if pca is None:
                self.pca_history.setdefault(layer_id, []).append(positions)
                if len(self.pca_history[layer_id]) > self.pca_samples:
                    train = np.concatenate(self.pca_history[layer_id], axis=0)
                    pca = PCA(n_components=self.pca_dim)
                    pca.fit(train)

                    self.positions_plot.setdefault(layer_id, []).append(pca.transform(train))
                    self.pca_history[layer_id] = []
                    self.pca[layer_id] = pca

                    # PCA just became available for this layer: transform and
                    # commit any cluster snapshots that arrived while we were
                    # still collecting samples, tagging them with the position
                    # index that now exists (there's only one so far, index 0).
                    pending = self._cluster_pending.pop(layer_id, [])
                    if pending:
                        position_index = len(self.positions_plot[layer_id]) - 1
                        for msg in pending:
                            centroids_2d = pca.transform(msg["centroids"])
                            self.cluster_history.setdefault(layer_id, []).append({
                                "position_index": position_index,
                                "labels": msg["labels"],
                                "best_k": msg["best_k"],
                                "best_score": msg["best_score"],
                                "centroids": centroids_2d,
                            })
            else:
                self.positions_plot.setdefault(layer_id, []).append(pca.transform(positions))

    def add_cluster(
        self,
        layer_id: str,
        labels: Any,
        best_k: int,
        best_score: float,
        centroids: Any,
    ) -> None:
        """Handle a snapshot published on the ClusterPlugin's "clusters" topic.

        Cluster updates arrive less frequently than position updates. Each
        snapshot is stamped with the position_index it corresponds to (the
        latest index in positions_plot[layer_id] at the time it's processed),
        so the UI can resolve "what did clustering look like as of position
        step N" consistently, including when the user scrubs backwards.
        """
        labels_np = _to_numpy(labels)
        centroids_np = _to_numpy(centroids)

        with self.locks.positions:
            pca = self.pca.get(layer_id)

            if pca is None:
                # PCA isn't fit yet for this layer: stash the raw centroids and
                # transform them once add_positions() fits PCA for this layer.
                self._cluster_pending.setdefault(layer_id, []).append({
                    "labels": labels_np,
                    "best_k": best_k,
                    "best_score": best_score,
                    "centroids": centroids_np,
                })
                return

            position_index = max(len(self.positions_plot.get(layer_id, [])) - 1, 0)
            centroids_2d = pca.transform(centroids_np)
            self.cluster_history.setdefault(layer_id, []).append({
                "position_index": position_index,
                "labels": labels_np,
                "best_k": best_k,
                "best_score": best_score,
                "centroids": centroids_2d,
            })

    # ---------------------------------------------------------
    # Accessors used by the layer board widget (modularui/plots.py)
    # ---------------------------------------------------------

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

    def get_position_steps(self, layer_id: str) -> int:
        with self.locks.positions:
            return len(self.positions_plot.get(layer_id, []))

    def get_position_at(self, layer_id: str, index: int) -> Optional[np.ndarray]:
        with self.locks.positions:
            steps = self.positions_plot.get(layer_id)
            if not steps or index < 0 or index >= len(steps):
                return None
            return steps[index]

    def get_layer_stats(self, layer_id: str) -> Optional[Dict[str, Any]]:
        """Returns the layer's space_stats entry: expected keys count_on,
        count_off, score_on, score_off, each an array aligned by embedding
        index (same ordering as positions for that layer)."""
        with self.locks.stats:
            return self.stats.get(layer_id)

    def get_pca_progress(self, layer_id: str) -> Optional[Dict[str, int]]:
        """Returns None once PCA is ready for this layer, otherwise
        {"collected": n, "required": n} describing sample-collection progress."""
        with self.locks.positions:
            if layer_id in self.pca:
                return None
            collected = len(self.pca_history.get(layer_id, []))
            return {"collected": collected, "required": self.pca_samples}

    def get_cluster_for_index(self, layer_id: str, position_index: int) -> Optional[Dict[str, Any]]:
        """Returns the most recent cluster snapshot whose position_index is
        <= the requested one (cluster_history is append-ordered by increasing
        position_index, so a binary search suffices)."""
        with self.locks.positions:
            history = self.cluster_history.get(layer_id)
            if not history:
                return None
            idxs = [h["position_index"] for h in history]
            pos = bisect.bisect_right(idxs, position_index) - 1
            if pos < 0:
                return None
            return history[pos]
