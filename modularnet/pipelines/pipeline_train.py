

import multiprocessing
import sys

# Set fork method before any other multiprocessing operations
# This must be done at module level, not inside if __name__ == "__main__"
# to prevent debugger/spawn issues with semaphores

try:
    multiprocessing.set_start_method('fork', force=True)
except RuntimeError:
    # Already set in this process
    pass

from modularnet.deeplearning import DeeplearningTrainPlugin, DeeplearningSetupPlugin
from modularnet.trial import Trial, TrialConfig, TrialManager


class PipelineTrain:
    def __init__(self):
        self.config = TrialConfig()
        self.trial_manager = TrialManager()
        self.trial = Trial(self.config, self.trial_manager)
        

        dl_setup = DeeplearningSetupPlugin(name="deeplearning_setup", config=self.config)
        dl_train = DeeplearningTrainPlugin(name="deeplearning_train", config=self.config)

        self.trial.load_plugin(dl_setup)
        self.trial.load_plugin(dl_train)
    
    


if __name__ == "__main__":    
    pipeline = PipelineTrain()
    pipeline.trial.run()