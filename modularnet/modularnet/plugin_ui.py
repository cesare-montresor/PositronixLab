from collections import deque
import logging
import os
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any, Dict, Optional, List

import torch

from modularnet.modularui.modularui import ModularUI
from modularnet.plugin.plugin_worker import PluginWorker
from modularnet.trial.trial_config import TrialConfig
from modularnet.plugin import BasePlugin, PluginDescriptor


logger = logging.getLogger(__name__)


class UIPlugin(BasePlugin):
    

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

    def on_start(self) -> None:
        self.trial_config = self.env_get("trial_config")
        self.ui = ModularUI(host=self.host, port=self.port, save_path=self.save_path)
        self.ui.run()


        self.topic_subscribe("metrics", self._on_metrics_received)
        self.topic_subscribe("positions", self._on_positions_received)
        

    def main(self) -> None:
        all_vars = self.env_get()
        self.ui.update_vars(all_vars=all_vars)
        self.ui.save_state()
        time.sleep(2.0)  # Allow other plugins to access this plugin's state
        
    def on_stop(self) -> None:
        self.ui.stop() 


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


