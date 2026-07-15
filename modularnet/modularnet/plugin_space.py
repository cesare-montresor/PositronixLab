import logging

from typing import TYPE_CHECKING, Any, Dict, Optional, List

import time

from torch.cuda import Event

from modularnet.deeplearning import DeeplearningCommon
from modularnet.modularobserver.modularobserver import ModularObserver
from modularnet.modularspace.modularspace import ModularSpace
from modularnet.plugin.plugin_worker import PluginWorker
from modularnet.trial.trial_config import TrialConfig
from modularnet.dataprovider.dataprovider import DataProvider
from modularnet.metamodel.metamodel import MetaModel
from modularnet.plugin import BasePlugin, PluginDescriptor


logger = logging.getLogger(__name__)


class ModularSpacePlugin(BasePlugin):
    
    @classmethod
    def default_options(cls) -> Dict[str, Any]:
        return {
            "device": None,
            "debug": False,
            "compute_scores": True,
            "enable_stats": True
        }

    @classmethod
    def Descriptor(cls, options: Dict[str, Any] = None) -> PluginDescriptor:
        return super().Descriptor(
            version="1.0.0",
            author="ModularNet Team",
            description="Receives activations and scores from the model, processes them in the modular space, and publishes updated positions.",
            options=options
        )

    def __init__(
        self,
        worker: PluginWorker,
        plugin_id: str,
        options: Dict[str, Any],        
    ):
        super().__init__(worker, plugin_id, options )
        
        self.device = options.get("device")
        self.debug = options.get("debug")
        self.stats_enabled = options.get("enable_stats")
        
        
        self.trial_config: TrialConfig = None  # Will be set in on_start via env_get("trial_config")
        self.space: Optional[ModularSpace] = None
        self.observed_layers: dict = None




    def on_start(self) -> None:
        self.trial_config: TrialConfig = self.env_get("trial_config")
        self.observed_layers: dict = self.env_get("observed_layers")
        self.space = ModularSpace(self.trial_config, self.observed_layers, self.device, self.stats_enabled)
        self.topic_subscribe("activations", self._on_activations)
        self.topic_subscribe("scores", self._on_scores)

        

    def main(self) -> None:
        self.env_set("space_stats", self.space.stats)  # Make this plugin accessible to other plugins
        time.sleep(1.0)  # Allow other plugins to access this plugin's state
        pass

    def _on_activations(self, data: Any):
        """Handle activations message from the model."""
        layer_id = data.get("layer_id")
        activations = data.get("activations")
        
        self.space.process_activations(layer_id, activations)

    
    def _on_scores(self, data: Any):
        layer_id = data.get("layer_id")
        scores = data.get("scores")
        event: Event = data.get("ready")
        
        event.synchronize()  # wait for GPU, blocks only this subscriber

        loss, (positions, _) = self.space.process_scores(layer_id, scores)

        self.topic_publish("metrics", {"layer_id": layer_id, "metric": loss, "name": "spaceloss"})
        self.topic_publish("positions", {"layer_id": layer_id, "positions": positions})
