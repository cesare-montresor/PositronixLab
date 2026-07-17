from collections import deque
import logging
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
        self.positions_maxlen = options.get("positions_maxlen")

        self.metrics: Dict[str, Dict[ str, List[float] ]] = {}  # Store metrics received from the model
        self.positions: Dict[str, deque[torch.Tensor] ] = {}  # Store positions received from the model
        self.stats: Dict[str, torch.Tensor ] = {}  # Store statistics received from the model


    def on_start(self) -> None:
        self.trial_config = self.env_get("trial_config")
        self.ui = ModularUI(host=self.host, port=self.port)
        self.ui.run()


        self.topic_subscribe("metrics", self._on_metrics_received)
        self.topic_subscribe("positions", self._on_positions_received)
        

    def main(self) -> None:
        all_vars = self.env_get()
        self.ui.update_layout(all_vars=all_vars)
        time.sleep(1.0)  # Allow other plugins to access this plugin's state
        
    def on_stop(self) -> None:
        self.ui.stop() 



    def _on_metrics_received(self, metrics: Dict[str, Any]) -> None:
        layer_id = metrics["layer_id"]
        name = metrics["name"]
        value = metrics["metric"]
        self.metrics.setdefault(layer_id, {}).setdefault(name, []).append(value)

    def _on_positions_received(self, message: Dict[str, Any]) -> None:
        layer_id = message["layer_id"]
        positions = message["positions"]
        self.positions.setdefault(layer_id, deque(maxlen=self.positions_maxlen)).append(positions)
    