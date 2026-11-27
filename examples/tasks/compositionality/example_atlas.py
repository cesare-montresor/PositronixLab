

import logging

import torch

from positronixlab.dataprovider.dataprovider_factory import DataProviderFactory
from positronixlab.metamodel.metamodel import MetaModel
from positronixlab.pipeline import PipelineConfig
from positronixlab.tasks.compositionality.compositionality import (
    CompositionalityPipeline,
)
from positronixlab.utils import utils

logger = logging.getLogger(__name__)

class AtlasCompositionalityExample:

    def __init__(self, model_path: str = None, vars_path: str = None, methods: list[str] = None):
        self.methods: list[str] = ["atlas", "weightmask"] if methods is None else methods
        self.cluster_stats = {}

        self.dataprovider  = None
        self.model = None
        self.dp = None

        self.trial_config = PipelineConfig()
        
        self.dp = DataProviderFactory.from_config(
            self.trial_config,
        )

        self.model_path = 'trained_model.pth' if model_path is None else model_path
        self.model = MetaModel( "model", self.trial_config)
        self.model.load(self.model_path)

        self.vars_path = 'all_vars.pkl' if vars_path is None else vars_path
        self.all_vars = utils.load_data(self.vars_path) if self.vars_path is not None else None
        self.cluster_stats = self.all_vars.get("cluster_stats")
        self.clusters = {key: torch.tensor(val['labels'], device=self.model.device) for key, val in self.cluster_stats.items()}

        print(self.clusters)


    def main(self) -> None:
  

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
        print("Atlas Cluster Stats:")
        print(self.cluster_stats)



if __name__ == "__main__":
    atlas_example = AtlasCompositionalityExample()
    atlas_example.main()