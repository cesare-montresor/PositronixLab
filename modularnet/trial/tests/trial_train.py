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

        alive = list(filter(lambda s: s["alive"], statuses.values()))
        if len(alive) == 0: return True
        workers = {s["worker"] for s in alive}


        return all(not status["alive"] for status in statuses.values())
    
    


class TestTrialTrain:
    def test_base_integration(self):
        self.config = TrialConfig()
        self.trial_manager = TrialManager()
        self.trial = TrialTrainSimple(self.config, self.trial_manager)
        

        

        self.trial.run()
        while not self.trial.is_complete():
            pass  # In a real test, we would have better synchronization here
        print("Trial completed successfully")
        

    

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
