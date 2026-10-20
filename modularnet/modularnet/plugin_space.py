import logging

from typing import TYPE_CHECKING, Any, Dict, Optional, List

import time

import torch
from torch.cuda import Event

from modularnet.modularspace.modularspace import ModularSpace
from parallelazy.plugin_worker import PluginWorker
from modularnet.trial.trial_config import TrialConfig
from parallelazy import Plugin, PluginDescriptor, PluginWorker, IpcEventCuda


logger = logging.getLogger(__name__)


class ModularSpacePlugin(Plugin):
    
    @classmethod
    def default_options(cls) -> Dict[str, Any]:
        return {
            "device": None,
            "debug": False,
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
        while self.observed_layers is None:
            self.observed_layers = self.env_get("observed_layers")
            if self.observed_layers is None:
                logger.warning(f"[{self.plugin_id}] Waiting for observed_layers to be set in the environment...")
                time.sleep(1.0)
        logger.info(f"[{self.plugin_id}] observed_layers:\n{self.observed_layers}")
        self.space = ModularSpace(self.trial_config, self.observed_layers, self.device, self.stats_enabled)
        self.topic_subscribe("activations", self._on_activations)
        self.topic_subscribe("scores", self._on_scores)

        

    def main(self) -> None:
        self.env_set("space_stats", {layer_id: stat.cpu().numpy() for layer_id, stat in self.space.stats.items()})  # Make this plugin accessible to other plugins
        time.sleep(1.0)  # Allow other plugins to access this plugin's state    

    def _on_activations(self, data: Any):
        """Handle activations message from the model."""
        layer_id = data.get("layer_id")
        activations = data.get("activations")
        
        losses, positions, stats = self.space.process_activations(layer_id, activations)
        del activations  # Free memory
        self._on_position_update(layer_id, losses, positions)

    
    def _on_scores(self, data: Any):
        layer_id = data.get("layer_id")
        scores = data.get("scores")
        ready: IpcEventCuda = data.get("ready")
        
        ready.wait()  # wait for GPU, blocks only this subscriber

        losses, positions, stats = self.space.process_scores(layer_id, scores)


        ready.clear()  # Signal that the scores have been processed
          # Signal that the scores have been processed
        del scores  # Free memory

        self._on_position_update(layer_id, losses, positions)


    def _on_position_update(self, layer_id: str, losses: torch.Tensor, positions: torch.Tensor):
        self.topic_publish("metrics", {"layer_id": layer_id, "metric": losses, "name": "spaceloss"})
        self.topic_publish("positions", {"layer_id": layer_id, "positions": positions})