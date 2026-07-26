import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, List

import torch
from torch import cuda

from modularnet.deeplearning import DeeplearningCommon
from modularnet.modularobserver.layerobserver import LayerObserver
from modularnet.modularobserver.modularobserver import ModularObserver


from modularnet.plugin.plugin_worker import PluginWorker
from modularnet.spacemath.spacemath import SpaceMath
from modularnet.trial.trial_config import TrialConfig
from modularnet.dataprovider.dataprovider import DataProvider
from modularnet.metamodel.metamodel import MetaModel
from modularnet.plugin import BasePlugin, PluginDescriptor


logger = logging.getLogger(__name__)


class ObservableTrainPlugin(BasePlugin):
    
    @classmethod
    def default_options(cls) -> Dict[str, Any]:
        return {
            "device": None,
            "debug": False,
            "compute_scores": False,
            "num_workers_data": 2,
        }

    @classmethod
    def Descriptor(cls, options: Dict[str, Any] = None) -> PluginDescriptor:
        return super().Descriptor(
            version="1.0.0",
            author="ModularNet Team",
            description="Plugin that sets up the model and data provider for deep learning trials.",
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
        self.compute_scores: bool = options.get("compute_scores")  # If True, send scores; if False, send activations
        self.num_workers_data: int = options.get("num_workers_data")

        self.trial_config: TrialConfig = None  # Will be set in on_start via env_get("trial_config")
        self.model: MetaModel = None
        self.dataprovider: DataProvider = None
        self.modular_observer: ModularObserver = None  # Placeholder for modular observer if needed
        self.trainer: DeeplearningCommon = None  # Will be initialized in main()

        self.forward_stream = cuda.Stream()


    def on_start(self) -> None:
        logger.info(f"[{self.plugin_id}] Starting deeplearning setup...")

        self.trial_config = self.env_get("trial_config")
        
        if not self.trial_config:
            raise ValueError("trial_config not provided in config")
        
        # Pass model device to dataset
        self.dataprovider = DataProvider.from_config(
            self.trial_config,
            num_workers=self.num_workers_data
        )

        # Create model first to determine device
        self.model = MetaModel(self.plugin_id, self.trial_config, device=self.device)
        # self.env_set("model", self.model)

        
        #self.env_set("dataprovider", self.dataprovider)

        self.modular_observer = ModularObserver(self.trial_config, self.model)
        self.modular_observer.observe(self.model)
        self.modular_observer.on_forward_layer(self.on_layer_forward)

        self.observed_layers = {observer.layer_id: observer.to_dict() for observer in self.modular_observer.observers}

        self.env_set("observed_layers", self.observed_layers)  # Make observed layers available to other plugins
        


        self.trainer = DeeplearningCommon(
            config=self.trial_config,
            model=self.model,
            dp=self.dataprovider
        )

        self.trainer.on_epoch_start(self.on_epoch_start)
        self.trainer.on_epoch_end(self.on_epoch_end)

    def on_epoch_start(self, epoch: int, epochs: int):
        self.env_set("epoch", (epoch, epochs))
    
    def on_epoch_end(self, epoch: int, epochs: int, metrics: Dict[str, float]):
        self.topic_publish("metrics", {"layer_id": "model", "metric": metrics, "name": "training"})
        
    
    def main(self):
        autoencoder = self.trial_config.task  == 'autoencoder'
        self.trainer.train(lr=self.trial_config.learning_rate, epoch_total=self.trial_config.epochs, autoencoder=autoencoder)
        self.stop()


    def on_layer_forward(self, layer: LayerObserver, x_in: torch.Tensor, x_out: torch.Tensor):
        if self.compute_scores:
            self.on_layer_forward_scores(layer, x_in, x_out)
        else:
            self.on_layer_forward_activations(layer, x_in, x_out)

    def on_layer_forward_activations(self, layer: LayerObserver, x_in: torch.Tensor, x_out: torch.Tensor):
        #TODO: Should I send scores? should i use cuda timelines ? 

        data = {
            'layer_id': layer.layer_id,
            "activations": x_out.detach()
        }
        self.topic_publish("activations", data)
        
    

    def on_layer_forward_scores(self, layer: LayerObserver, x_in: torch.Tensor, x_out: torch.Tensor):
        #TODO: Should I send scores? should i use cuda timelines ? 

        self.forward_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.forward_stream):
            scores = SpaceMath.activations_to_scores(
                x_out.detach(), layer.layer_type, layer.activation,
                self.trial_config.modular_space_active_q, cnn_locality_weight=0.5
            )
            event = torch.cuda.Event(interprocess=True)
            event.record()
        data = {
            'layer_id': layer.layer_id,
            'scores': scores,
            'ready': event
        }

        self.topic_publish("scores", data)
        
    