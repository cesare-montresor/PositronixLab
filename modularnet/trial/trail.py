from modularnet.trial.trial_config import TrialConfig
from modularnet.trial.trial_manager import TrialManager
from modularnet.plugin_old import Plugin, PluginManager


class Trial():
    def __init__(self, config: TrialConfig, trial_manager:TrialManager):
        self.config = config
        self.trial_manager = trial_manager
        
        self.plugin_manager = PluginManager()

    def load_plugin(self, plugin:Plugin):
        self.plugin_manager.register_plugin(plugin)
        return plugin
    
    def run(self):
        self.plugin_manager.start_all_plugins()

    def is_complete(self):
        return self.plugin_manager.all_plugins_completed()
        
        
