import logging
import time
from typing import Any, Dict, Set

from positronixlab.modularui.modularui import ModularUI

from parallelazy import Plugin, PluginDescriptor, PluginWorker


logger = logging.getLogger(__name__)


class UIPlugin(Plugin):
    """Pure view layer. Raw data ingestion, decimation, PCA/UMAP, cluster sync,
    history and persistence all live in StatsPlugin; this plugin only
    consumes its aggregated output (the `stats_*` topics / env vars) plus the
    low-rate env vars that never needed aggregation (graph, config, ...)."""

    DISCOVERY_INTERVAL_S = 10.0

    _DEDICATED_KEYS = {"space_stats", "observed_layers", "model_graph", "trial_config", "stats_pca_progress"}

    _IGNORED_PREFIXES = ("stats_",)

    @classmethod
    def default_options(cls) -> Dict[str, Any]:
        return {
            "host": None,
            "port": None,
            "positions_maxlen": 1000,
            "metrics_maxlen": 2000,
            "cluster_maxlen": 500,
        }

    @classmethod
    def Descriptor(cls, options: Dict[str, Any] = None) -> PluginDescriptor:
        return super().Descriptor(
            version="2.0.0",
            author="PositronixLab Team",
            description="Displays the aggregated data published by StatsPlugin in the web UI.",
            options=options
        )

    def __init__(self, worker: PluginWorker, plugin_id: str, options: Dict[str, Any]):
        super().__init__(worker, plugin_id, options)
        self.ui: ModularUI = None
        self.host = options.get("host")
        self.port = options.get("port")
        self.positions_maxlen = options.get("positions_maxlen", 1000)
        self.metrics_maxlen = options.get("metrics_maxlen", 2000)
        self.cluster_maxlen = options.get("cluster_maxlen", 500)

        self._subscribed_keys: Set[str] = set()

    def on_start(self) -> None:
        self.ui = ModularUI(
            host=self.host,
            port=self.port,
            positions_maxlen=self.positions_maxlen,
            metrics_maxlen=self.metrics_maxlen,
            cluster_maxlen=self.cluster_maxlen,
        )
        self.ui.run()

        self.topic_subscribe("stats_positions", self._on_positions)
        self.topic_subscribe("stats_spaceloss", self._on_spaceloss)
        self.topic_subscribe("stats_spaceloss_overall", self._on_spaceloss_overall)
        self.topic_subscribe("stats_training", self._on_training)
        self.topic_subscribe("stats_clusters", self._on_clusters)

        self.env_subscribe("stats_pca_progress", self._on_pca_progress_changed)
        self.env_subscribe("space_stats", self._on_space_stats_changed)
        self.env_subscribe("observed_layers", self._on_observed_layers_changed)
        self.env_subscribe("model_graph", self._on_model_graph_changed)
        self.env_subscribe("trial_config", self._on_config_changed)
        
        self._subscribed_keys |= self._DEDICATED_KEYS

    def main(self) -> None:
        current_keys = set(self.env_get().keys())
        for key in current_keys - self._subscribed_keys:
            if key.startswith(self._IGNORED_PREFIXES):
                continue
            self.env_subscribe(key, self._make_generic_var_handler(key))
            self._subscribed_keys.add(key)

        time.sleep(self.DISCOVERY_INTERVAL_S)

    def on_stop(self) -> None:
        self.ui.stop()

    def _make_generic_var_handler(self, key: str):
        def _handler(var_name: str) -> None:
            self.ui.set_var(var_name, self.env_get(var_name))
        return _handler

    # -----------------------------------------------------
    # StatsPlugin topics
    # -----------------------------------------------------
    def _on_positions(self, msg: Dict[str, Any]) -> None:
        self.ui.add_position_step(msg["key"], msg["value"], msg.get("index"))

    def _on_spaceloss(self, msg: Dict[str, Any]) -> None:
        self.ui.add_spaceloss_point(msg["key"], msg["names"], msg["value"])

    def _on_spaceloss_overall(self, msg: Dict[str, Any]) -> None:
        self.ui.add_overall_spaceloss_point(msg["names"], msg["value"])

    def _on_training(self, msg: Dict[str, Any]) -> None:
        self.ui.add_training_point(msg["key"], msg["value"])

    def _on_clusters(self, msg: Dict[str, Any]) -> None:
        self.ui.add_cluster(
            layer_id=msg["layer_id"],
            position_index=msg["position_index"],
            labels=msg.get("labels"),
            best_k=msg.get("best_k"),
            best_score=msg.get("best_score"),
            centroids=msg.get("centroids"),
        )

    # -----------------------------------------------------
    # env vars
    # -----------------------------------------------------
    def _on_pca_progress_changed(self, var_name: str) -> None:
        self.ui.set_pca_progress(self.env_get(var_name))

    def _on_space_stats_changed(self, var_name: str) -> None:
        self.ui.set_space_stats(self.env_get(var_name))

    def _on_observed_layers_changed(self, var_name: str) -> None:
        self.ui.set_observed_layers(self.env_get(var_name))

    def _on_model_graph_changed(self, var_name: str) -> None:
        self.ui.set_model_graph(self.env_get(var_name))

    def _on_config_changed(self, var_name: str) -> None:
        self.ui.set_config(self.env_get(var_name))