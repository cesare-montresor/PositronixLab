import logging
import time
from typing import Any, Dict, List, Optional

from torch import nn
import torch

from modularnet.dataprovider.covertype_dataprovider import DataProvider
from modularnet.dataprovider.dataprovider_factory import DataProviderFactory
from modularnet.deeplearning import DeeplearningCommon
from modularnet.downstream_tasks.compositionality.atlas_check import AtlasCheck
from modularnet.downstream_tasks.compositionality.weightmask_check import WeightmaskCheck
from modularnet.metamodel.metamodel import MetaModel
from modularnet.plugin.plugin_worker import PluginWorker
from modularnet.trial.trial_config import TrialConfig
from modularnet.plugin import BasePlugin, PluginDescriptor


logger = logging.getLogger(__name__)


class CompositionalityPlugin(BasePlugin):

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
                self.model = MetaModel.load(model_path, name="trained_model", config=self.trial_config)
            else:
                logger.info("Waiting for trained model to be available...")
                time.sleep(5)  # Wait for 5 seconds before checking again
                
        
        if "atlas" in self.methods:
            atlas_check = AtlasCheck(self.trial_config, self.model, self.dp, self.cluster_stats)
            atlas_check.run()
        
        if "weightmask" in self.methods:
            weightmask_check = WeightmaskCheck(self.model, self.dp)
            weightmask_check.run()

        self.env_set("compositionality_completed", True)

        pass


