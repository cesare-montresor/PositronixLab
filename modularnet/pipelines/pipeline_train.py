

from modularnet.deeplearning.deeplearning_train_plugin import DeeplearningTrainPluginDescriptor
from modularnet.deeplearning.deeplearning_setup_plugin import DeeplearningSetupPluginDescriptor
from modularnet.trial import Trial, TrialConfig, TrialManager


class PipelineTrain:
    def __init__(self):
        self.config = TrialConfig()
        self.trial_manager = TrialManager()
        self.trial = Trial(self.config, self.trial_manager)
        

        dl_setup = DeeplearningSetupPluginDescriptor(config=self.config)
        dl_train = DeeplearningTrainPluginDescriptor(config=self.config)

        self.trial.load_plugin(dl_setup)
        self.trial.load_plugin(dl_train)


        self.trial.run()
    
    


if __name__ == "__main__":    
    pipeline = PipelineTrain()
    pipeline.trial.run()