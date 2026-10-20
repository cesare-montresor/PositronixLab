import logging
import time
from typing import Any, Dict, List

import torch

from modularnet.dataprovider.dataprovider_factory import DataProviderFactory
from modularnet.downstream_tasks.compositionality.compositionality import CompositionalityPipeline
from modularnet.metamodel.metamodel import MetaModel
from modularnet.trial.trial_config import TrialConfig
from parallelazy import Plugin, PluginDescriptor, PluginWorker


logger = logging.getLogger(__name__)


class CompositionalityPlugin(Plugin):

    @classmethod
    def default_options(cls) -> Dict[str, Any]:
        return {
            "device": None,
            "debug": False,
            "num_workers_data": 2,
            "methods": ["weightmask", "atlas"],
        }
    
    @classmethod
    def Descriptor(cls, options: Dict[str, Any] = None) -> PluginDescriptor:
        return super().Descriptor(
            version="1.0.0",
            author="ModularNet Team",
            description="Recieves trained model and dataprovider, then runs compositionality tests.",
            options=options
        )

    def __init__(
        self,
        worker: PluginWorker,
        plugin_id: str,
        options: Dict[str, Any],        
    ):
        super().__init__(worker, plugin_id, options )
        # Pass model device to dataset
        self.device = options.get("device")
        self.debug = options.get("debug")
        self.num_workers_data: int = options.get("num_workers_data")
        self.methods: List[str] = options.get("methods")
        self.cluster_stats = {}
        self.clusters = None
        self.checks = []

        self.dataprovider  = None
        self.model = None
        self.dp = None

    def on_start(self) -> None:
        self.trial_config: TrialConfig = self.env_get("trial_config")
        
        self.dp = DataProviderFactory.from_config(
            self.trial_config,
            num_workers=self.num_workers_data
        )
        self.env_set("compositionality_completed", False)
        

    def main(self) -> None:

        while self.model is None:
            model_path = self.env_get("trained_model")
            if model_path is not None:
                self.model = MetaModel(name="trained_model", config=self.trial_config)
                self.model.load(model_path)
                self.model.to(self.device)
            else:
                logger.info("Waiting for trained model to be available...")
                time.sleep(5)  # Wait for 5 seconds before checking again

        self.cluster_stats = self.env_get("cluster_stats")      
        self.clusters = {key:torch.tensor(val['labels'], device=self.model.device) for key,val in self.cluster_stats.items() }
        
        # Assuming your model, dataprovider, config, clusters, and target_layers are already initialized
        runner = CompositionalityPipeline(model=self.model, dataprovider=self.dp, config=self.trial_config)

        # Generates masks for both paradigms and runs inference evaluation
        results = runner.run_full_evaluation(
            clusters=self.clusters, 
            atlas_threshold=0.05,  # Tune your activation distribution threshold
            csordas_epochs=10      # Tune Csordas gradient optimization cycles
        )

        print("Compositionality Evaluation Results:")
        print(results)

        
        self.env_set("compositionality_completed", True)


