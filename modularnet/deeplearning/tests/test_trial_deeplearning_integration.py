"""
Integration test for TrialDeepLearningCore with full training.

This test creates a complete TrialDeepLearningCore with standard trial config
and attempts a full training pipeline with real components.

This is NOT a mock test - it uses actual:
- TrialConfig with standard settings
- Real data provider (MNIST)
- Real neural network model
- Real training plugins

The test runs for minimal epochs to complete quickly while validating
the full integration of setup + training pipeline.
"""

import pytest
import torch

from modularnet.trial.trial_config import TrialConfig
from modularnet.trial.trial_manager import TrialManager



from modularnet.deeplearning.deeplearning_train_plugin import DeeplearningTrainPluginDescriptor
from modularnet.deeplearning.deeplearning_setup_plugin import DeeplearningSetupPluginDescriptor
from modularnet.trial import Trial, TrialConfig, TrialManager


class TestTrialDeepLearningIntegration:
    def test_base_integration(self):
        self.config = TrialConfig()
        self.trial_manager = TrialManager()
        self.trial = Trial(self.config, self.trial_manager)
        

        dl_setup = DeeplearningSetupPluginDescriptor(config=self.config)
        dl_train = DeeplearningTrainPluginDescriptor(config=self.config)

        self.trial.load_plugin(dl_setup)
        self.trial.load_plugin(dl_train)


        self.trial.run()
    
    



if __name__ == '__main__':
    pytest.main([__file__, '-v'])
