import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, List

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


class StrangeNetOverspacePlugin(BasePlugin):
    
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
        self.num_workers_data = options.get("num_workers_data", 0)
        self.debug = options.get("debug", False)
        
        
        self.trial_config: TrialConfig = None  # Will be set in on_start via env_get("trial_config")
        self.space: Optional[ModularSpace] = None
        self.observed_layers: dict = None




    def on_start(self) -> None:
        self.trial_config: TrialConfig = self.env_get("trial_config")
        self.observed_layers: dict = self.env_get("observed_layers")
        self.space = ModularSpace(self.plugin_id, self.trial_config, device=self.device, observed_layers=self.observed_layers)
        self.topic_subscribe("activations", self._on_activations)
        self.topic_subscribe("scores", self._on_scores)

        

    def main(self) -> None:
        pass

    def _on_activations(self, data: Any):
        """Handle activations message from the model."""
        layer_info = data.get("layer")
        activation = data.get("activation")
        
        self.space.process_activations(layer_info, activation)

    
    def _on_scores(self, data: Any):
        layer_info = data.get("layer")
        scores = data.get("scores")
        event: Event = data.get("ready")
        
        event.synchronize()  # wait for GPU, blocks only this subscriber

        metrics, positions = self.space.process_scores(layer_info, scores)


        layer_id = layer_info["layer_id"]
        self.topic_publish("metrics", {"layer_id": layer_id, "metrics": metrics})
        self.topic_publish("positions", {"layer_id": layer_id, "positions": positions})
