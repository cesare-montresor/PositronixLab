from modularnet.config.trial_config import TrialConfig
from modularnet.deeplearning_core.deeplearning_core_plugin import DeeplearningSetupPlugin, DeeplearningTrainPlugin
from modularnet.modularobserver.modularobserver_plugin import ModularObserverPlugin
from modularnet.plugin.plugin import Plugin, PluginManager
from modularnet.trial.trial_manager import TrialManager


class TrialBase():
    def __init__(self, config: TrialConfig, trial_manager:TrialManager):
        self.config = config
        self.trial_manager = trial_manager
        
        self.plugin_manager = PluginManager()

    def load_plugin(self, plugin_class:Plugin, *args, **kwargs):
        plugin = plugin_class(*args, **kwargs)
        self.plugin_manager.register_plugin(plugin)
        return plugin
    
    def run(self):
        self.plugin_manager.start_all_plugins()

    def is_complete():
        





class TrialDeepLearningCore(TrialBase):
    def __init__(self, config: TrialConfig, trial_manager:TrialManager):
        super().__init__(config, trial_manager)
        # Load plugins in order of dependencies
        self.load_plugin(DeeplearningSetupPlugin, name="deeplearning_setup", config=config)
        self.load_plugin(DeeplearningTrainPlugin, name="deeplearning_train", config=config)