from collections import deque
import logging
import os
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any, Dict, Optional, List, Set

import torch

from modularnet.modularui.modularui import ModularUI
from modularnet.plugin.plugin_worker import PluginWorker
from modularnet.trial.trial_config import TrialConfig
from modularnet.plugin import BasePlugin, PluginDescriptor


logger = logging.getLogger(__name__)


class StatsPlugin(BasePlugin):

    # Discovery cadence for main()'s key-scan loop -- NOT a data refresh
    # rate. Once a key is subscribed, its actual UI refresh rate is however
    # often its source calls env_set() on it (via env_subscribe's push),
    # completely decoupled from this sleep().
    DISCOVERY_INTERVAL_S = 10.0

    # Env vars with a dedicated handler + dedicated storage on ModularUI.
    # Anything else discovered lands in the generic dynamic-var store via
    # ui.set_var() instead (see _make_generic_var_handler).
    _DEDICATED_KEYS = {"space_stats", "observed_layers", "model_graph", "trial_config"}

    @classmethod
    def default_options(cls) -> Dict[str, Any]:
        return {
            "host": None,
            "port": None,
            "positions_maxlen": 1000,
            "save_path": None,
        }


    @classmethod
    def Descriptor(cls, options: Dict[str, Any] = None) -> PluginDescriptor:
        return super().Descriptor(
            version="1.0.0",
            author="ModularNet Team",
            description="Receives metrics from the model and publishes them to the UI.",
            options=options
        )

    def __init__(
        self,
        worker: PluginWorker,
        plugin_id: str,
        options: Dict[str, Any],
    ):
        super().__init__(worker, plugin_id, options )
        self.ui: ModularUI = None  # Will be initialized in on_start()
        self.trial_config: TrialConfig = None  # Will be set in on_start via env_get("trial_config")
        self.host = options.get("host")
        self.port = options.get("port")
        self.save_path = options.get("save_path")

        # Keys we've already called env_subscribe() on -- both the dedicated
        # ones (registered in on_start) and any generic ones main() finds.
        self._subscribed_keys: Set[str] = set()

    def on_start(self) -> None:
        self.trial_config = self.env_get("trial_config")
        self.ui = ModularUI(host=self.host, port=self.port, save_path=self.save_path)
        self.ui.run()


        self.topic_subscribe("metrics", self._on_metrics_received)
        self.topic_subscribe("positions", self._on_positions_received)
        self.topic_subscribe("clusters", self._on_clusters_received)

        # space_stats, observed_layers, model_graph and trial_config are
        # each their own env var (set via env_set by whichever plugin
        # computes them) rather than fields nested inside a bigger blob --
        # so we keep each in sync directly via env_subscribe, with its own
        # setter/storage on ModularUI, rather than pulling them out of a
        # generic env_get()/update_vars() flow. env_subscribe also fires the
        # callback immediately with the current value, so this covers the
        # initial load too.
        self.env_subscribe("space_stats", self._on_space_stats_changed)
        self.env_subscribe("observed_layers", self._on_observed_layers_changed)
        self.env_subscribe("model_graph", self._on_model_graph_changed)
        self.env_subscribe("trial_config", self._on_config_changed)
        self._subscribed_keys |= self._DEDICATED_KEYS

    def main(self) -> None:
        """Periodically discovers *new* env var keys and wires a dedicated
        env_subscribe for each one found, instead of pulling the entire env
        dict every tick (the old `env_get() + ui.update_vars(all_vars=...)`
        pattern, which forced every var -- fast or barely-changing -- through
        the same fixed cadence as this loop's sleep()).

        Keys with a dedicated handler (see on_start / _DEDICATED_KEYS) are
        skipped here since they're already subscribed with their own
        storage on ModularUI; everything else lands in the generic
        dynamic-var store via ui.set_var(), refreshing the UI at whatever
        rate its actual source updates it -- e.g. "epoch" ends up pushing a
        handful of times over a whole trial, instead of being re-fetched
        (and re-rendered) on every single poll tick regardless.
        """
        current_keys = set(self.env_get().keys())
        for key in current_keys - self._subscribed_keys:
            self.env_subscribe(key, self._make_generic_var_handler(key))
            self._subscribed_keys.add(key)

        time.sleep(self.DISCOVERY_INTERVAL_S)

    def on_stop(self) -> None:
        self.ui.save_state()
        self.ui.stop()

    def _make_generic_var_handler(self, key: str):
        """Closes over `key` explicitly (rather than relying on the loop
        variable in main()) to avoid the classic late-binding closure bug --
        every handler wired from that loop needs to keep pointing at the key
        it was created for."""
        def _handler(var_name: str) -> None:
            self.ui.set_var(var_name, self.env_get(var_name))
        return _handler

    def _on_metrics_received(self, data: Dict[str, Dict[str, List[float]]]) -> None:
        layer_id: str = data.get("layer_id", None)
        metrics: Dict[str, List[float]] = data.get("metric", None)
        name: str = data.get("name", None)

        if name == 'spaceloss':
            self.ui.add_metric_spaceloss(layer_id, metrics)
        elif name == 'training':
            self.ui.add_metric_training(metrics)

        logger.debug(f"Metrics received: {metrics}")

    def _on_positions_received(self, data: Dict[str, List[torch.Tensor]]) -> None:
        positions: torch.Tensor = data.get("positions", None)
        layer_id: str = data.get("layer_id", None)
        self.ui.add_positions(layer_id, positions)
        logger.debug(f"Positions received: {positions}")

    def _on_clusters_received(self, data: Dict[str, Any]) -> None:
        """ClusterPlugin publishes here via topic_publish("clusters", {...})
        with labels, layer_id, best_k, best_score, and centroids. Forwarded
        straight to ModularUI, which PCA-transforms the centroids and syncs
        the snapshot to a position_index (see ModularUI.add_cluster)."""
        layer_id: str = data.get("layer_id", None)
        self.ui.add_cluster(
            layer_id=layer_id,
            labels=data.get("labels"),
            best_k=data.get("best_k"),
            best_score=data.get("best_score"),
            centroids=data.get("centroids"),
        )
        logger.debug(f"Cluster snapshot received for layer {layer_id}")

    def _on_space_stats_changed(self, var_name: str) -> None:
        value = self.env_get(var_name)
        self.ui.set_space_stats(value)

    def _on_observed_layers_changed(self, var_name: str) -> None:
        value = self.env_get(var_name)
        self.ui.set_observed_layers(value)

    def _on_model_graph_changed(self, var_name: str) -> None:
        value = self.env_get(var_name)
        self.ui.set_model_graph(value)

    def _on_config_changed(self, var_name: str) -> None:
        value = self.env_get(var_name)
        self.ui.set_config(value)
