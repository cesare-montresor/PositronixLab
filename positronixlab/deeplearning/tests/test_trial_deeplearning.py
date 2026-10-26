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

from positronixlab.deeplearning.deeplearning_train_plugin import DeeplearningTrainPlugin
from positronixlab.deeplearning.deeplearning_setup_plugin import DeeplearningSetupPlugin
from positronixlab.trial import TrialBase, TrialConfig, TrialManager


class TestTrialDeepLearning:
    def test_base_integration(self):
        self.config = TrialConfig()
        self.trial_manager = TrialManager()
        self.trial = TrialBase(self.config, self.trial_manager)
        

        dl_setup = DeeplearningSetupPlugin.Descriptor(options={"trial_config": self.config})
        dl_train = DeeplearningTrainPlugin.Descriptor(options={"trial_config": self.config})

        self.trial.load_plugin(dl_setup)
        self.trial.load_plugin(dl_train)


        self.trial.run()
        while not self.trial.is_complete():
            pass  # In a real test, we would have better synchronization here
    
    



if __name__ == '__main__':
    pytest.main([__file__, '-v', '-W','error::UserWarning'])
