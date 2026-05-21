from calendar import c
import re

import pytest

from modularnet.deeplearning.deeplearning_train_plugin import DeeplearningTrainPlugin
from modularnet.deeplearning.deeplearning_setup_plugin import DeeplearningSetupPlugin
from modularnet.trial import BaseTrial, TrialConfig, TrialManager


class TrialTrainSimple(BaseTrial):
    def __init__(self, config: TrialConfig, trial_manager: TrialManager):
        super().__init__(config, trial_manager)
        options = {
            "trial_config": self.config,
            "device": None,  # Use default device
            "num_workers_data": 0,  # No parallel data loading for test
            "debug": True
        }
        self.error = False
        self.dl_setup = DeeplearningSetupPlugin.Descriptor(options=options)
        self.dl_train = DeeplearningTrainPlugin.Descriptor(options=options)

        self.load_plugin(self.dl_setup)
        self.load_plugin(self.dl_train)

    def is_complete(self) -> bool:
        # For testing purposes, we can just check if the training plugin has finished
        statuses = self.plugin_manager.get_all_plugin_status()

        alive = list(filter(lambda s: s.process_alive, statuses.values()))
        if len(alive) == 0: return True

        return False
    
    


class TestTrialTrain:
    def test_setup_plugin_loads(self):
        """Test that setup plugin can be loaded and started."""
        config = TrialConfig()
        trial_manager = TrialManager()
        trial = TrialTrainSimple(config, trial_manager)
        
        # Just verify plugins loaded without error
        assert len(trial.plugin_manager._plugins) == 2
        print("Both plugins loaded successfully")

    def test_base_integration(self):
        """Test basic trial execution with setup plugin."""
        self.config = TrialConfig()
        self.trial_manager = TrialManager()
        self.trial = TrialTrainSimple(self.config, self.trial_manager)
        
        # Start plugins
        self.trial.run()
        
        # Give setup plugin time to complete on_start
        import time
        time.sleep(2)
        
        # Check that setup plugin completed without error
        statuses = self.trial.plugin_manager.get_all_plugin_status()
        for plugin_id, status in statuses.items():
            print(f"{plugin_id}: alive={status.process_alive}, worker_status={status.worker_status}")
            if status.worker_status:
                assert status.worker_status.status != "error", f"Plugin {plugin_id} has errors"
        
        print("Trial setup completed successfully")
        

    

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
