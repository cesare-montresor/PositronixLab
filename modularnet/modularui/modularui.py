import logging
import threading
import time
from typing import Any, Dict, List, Union

import numpy as np
from dash import Dash
from flask.helpers import get_root_path
from sklearn.decomposition import PCA

from modularnet.modularui.layouts import HomeLayout
from modularnet.utils.utils import save_data

logger = logging.getLogger(__name__)


class ModularUI:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8050,
        debug: bool = False,
        reloader: bool = False,
        save_path: str = None,
        poll_interval_ms: int = 1000,
    ):
        self.host = host
        self.port = port
        self.debug = debug
        self.reloader = reloader
        self.save_path = save_path
        self.pca_dim = 2
        self.pca_samples = 1000

        # Live state: written from worker/training threads via add_* /
        # update_vars, read from Dash's callback thread on every poll tick.
        # A single lock protects all of it.
        self._lock = threading.Lock()
        
        self.all_vars: Dict[str, Any] = {}
        self.stats: Dict[str, np.ndarray] = {}
        
        # Stored data streams
        self.training_metrics: Dict[str, List[float]] = {}
        self.layer_spaceloss: Dict[str, Dict[str, List[float]]] = {}
        self.positions_plot: Dict[str, List[np.ndarray]] = {}

        # Internal state for PCA processing
        self.pca: Dict[str, PCA] = {}
        self.pca_history: Dict[str, List[np.ndarray]] = {}


        self.home_layout = HomeLayout()

        self.app = Dash(__name__, external_stylesheets=self.home_layout.css)
        self.server_thread = None
        

        self.reset()
        
        # Layout + callbacks are set up once. 
        self.app.layout = self.home_layout.page_layout(poll_interval_ms=poll_interval_ms)
        self.home_layout.register_callbacks(self.app, self)

    def homedir(self):
        return get_root_path(__name__)

    @property
    def overall_spaceloss(self) -> Dict[str, List[float]]:
        """Dynamically computes the average space loss across all layers at each timestep."""
        with self._lock:
            if not self.layer_spaceloss:
                return {}
            
            # Find all unique metric names (e.g., 'loss', 'loss_on', 'loss_off')
            metric_names = set()
            for metrics in self.layer_spaceloss.values():
                metric_names.update(metrics.keys())
                
            overall = {}
            for name in metric_names:
                # Gather the lists for this specific metric from all layers that track it
                layer_lists = [
                    metrics[name] 
                    for metrics in self.layer_spaceloss.values() 
                    if name in metrics
                ]
                
                if not layer_lists:
                    continue
                    
                # Depending on threading, layers might be off by 1 step temporarily.
                # Take the min length so we only average complete timesteps.
                min_len = min(len(l) for l in layer_lists)
                
                # Compute the mean across all layers for each timestep
                avg_values = []
                for i in range(min_len):
                    avg = sum(l[i] for l in layer_lists) / len(layer_lists)
                    avg_values.append(float(avg))
                    
                overall[name] = avg_values
                
            return overall

    def save_state(self) -> bool:
        if self.save_path is None:
            return False
        with self._lock:
            data = {
                "all_vars": self.all_vars,
                "stats": self.stats,
                "training_metrics": self.training_metrics, 
                "layer_spaceloss": self.layer_spaceloss,
                "positions_plot": self.positions_plot,
                "pca_history": self.pca_history,
            }
        payload = {"header": {"timestamp": time.time()}, "data": data}
        save_data(self.save_path, payload)
        logger.info(f"UIPlugin state saved to {self.save_path}")
        return True

    def reset(self):
        with self._lock:
            self.stats.clear()
            self.training_metrics.clear()
            self.layer_spaceloss.clear()
            self.positions_plot.clear()
            self.pca_history.clear()
            self.pca.clear()

    def update_vars(self, all_vars: dict):
        if all_vars is None:
            return
        all_vars = dict(all_vars)  # don't mutate the caller's dict
        if "trial_config" in all_vars and hasattr(all_vars["trial_config"], "to_dict"):
            all_vars["trial_config"] = all_vars["trial_config"].to_dict()
        with self._lock:
            if "space_stats" in all_vars:
                self.stats = all_vars.pop("space_stats")
            self.all_vars = all_vars

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
        """Plot 1: Overall training metrics (e.g., train_loss, val_loss)."""
        with self._lock:
            for name, value in metrics.items():
                self.training_metrics.setdefault(name, []).append(float(value))

    def add_metric_spaceloss(self, layer_id: str, metrics: List[float]) -> None:
        """Plot 3: Space loss specific to a given layer. 
        (Plot 4 will automatically compute averages from this)."""
        with self._lock:
            loss, loss_on, loss_off, dict_center = metrics
            self.layer_spaceloss.setdefault(layer_id, {}).setdefault("loss", []).append(float(loss))
            self.layer_spaceloss.setdefault(layer_id, {}).setdefault("loss_on", []).append(float(loss_on))
            self.layer_spaceloss.setdefault(layer_id, {}).setdefault("loss_off", []).append(float(loss_off))
            self.layer_spaceloss.setdefault(layer_id, {}).setdefault("dict_center", []).append(float(dict_center))

    def add_positions(self, layer_id: str, positions: np.ndarray) -> None:
        """Plot 2: Positions per layer, evolving in time."""
        with self._lock:
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
            else:
                self.positions_plot.setdefault(layer_id, []).append(pca.transform(positions))