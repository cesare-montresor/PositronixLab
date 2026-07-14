import logging
from typing import Any, Dict, Optional

from modularnet.modularspace.modularoverspace import ModularOverSpace
from modularnet.plugin.plugin_worker import PluginWorker
from modularnet.trial.trial_config import TrialConfig
from modularnet.plugin import BasePlugin, PluginDescriptor


logger = logging.getLogger(__name__)


class OverspacePlugin(BasePlugin):
    
    @classmethod
    def Descriptor(cls, options: Dict[str, Any] = None) -> PluginDescriptor:
        return super().Descriptor(
            version="1.0.0",
            author="ModularNet Team",
            description="Receives labels from the model, processes them in the modular space, and publishes updated positions.",
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
        self.debug = options.get("debug", False)
        
        
        self.trial_config: TrialConfig = None  # Will be set in on_start via env_get("trial_config")
        self.space: Optional[ModularOverSpace] = None
        self.observed_layers: dict = None
        self.groups = {}




    def on_start(self) -> None:
        self.trial_config: TrialConfig = self.env_get("trial_config")
        self.observed_layers: dict = self.env_get("observed_layers")
        self.space = ModularOverSpace(self.trial_config, self.observed_layers, device=self.device, stats_enabled=self.debug)
        self.topic_subscribe("labels", self._on_labels)

        

    def main(self) -> None:
        pass
    
    def _on_labels(self, data: Any):
        """Handle labels message from the model.
            data =__  {
                "labels": labels,
                "layer_id": layer_id,
                "best_k": clusterer.best_k,
                "best_score": clusterer.best_score,
                "centroids": clusterer.centroids,
            })
        """

        layer_id = data.get("layer_id")
        labels = data.get("labels")
        centroids = data.get("centroids")

        if layer_id is None or labels is None or centroids is None:
            logger.warning("Overspace input missing layer_id, labels, or centroids")
            return

        state, events = self.space.update(layer_id, labels, centroids)
        positions = state.outer

        metrics = {
            "k": int(state.K),
            "events": len(events),
            "matches": sum(1 for event in events if event.kind == "match"),
            "splits": sum(1 for event in events if event.kind == "split"),
            "merges": sum(1 for event in events if event.kind == "merge"),
            "switches": sum(1 for event in events if event.kind == "switch"),
            "adds": sum(1 for event in events if event.kind == "add"),
            "removes": sum(1 for event in events if event.kind == "remove"),
        }

        self.topic_publish("metrics", {"layer_id": layer_id, "metrics": metrics})
        self.topic_publish(
            "positions",
            {
                "layer_id": layer_id,
                "positions": positions,
                "labels": labels,
                "centroids": centroids,
                "events": events,
            },
        )
