from abc import ABC, abstractmethod
import logging
import time
from typing import Dict, Any, Optional

from modularnet.trial.trial_config import TrialConfig
from modularnet.trial.trial_manager import TrialManager
from modularnet.plugin import PluginManager, PluginDescriptor

logger = logging.getLogger(__name__)


class BaseTrial(ABC):
    """Base trial class."""
    
    def __init__(self, config: TrialConfig, trial_manager: TrialManager):
        self.config = config
        self.trial_manager = trial_manager
        
        # Use new Ray-based plugin manager
        self.plugin_manager = PluginManager()
        self.results: Optional[Dict[str, Any]] = None

    def load_plugin(self, descriptor: PluginDescriptor, plugin_config: Dict[str, Any]={}) -> str:
        """
        Load a plugin from descriptor.
        
        Args:
            descriptor: PluginDescriptor with plugin metadata
            plugin_config: Configuration for plugin instantiation
            
        Returns:
            Plugin ID
        """
        plugin_id = descriptor.name
        self.plugin_manager.load_plugin(descriptor, plugin_config)
        logger.info(f"Plugin {plugin_id} loaded")
        return plugin_id
    
    def run(self) -> None:
        """Run all loaded plugins."""
        self.plugin_manager.start_all_plugins()

    def is_error(self) -> bool:
        """Check if any plugin is in error state."""
        statuses = self.plugin_manager.get_all_plugin_status()
        for status in statuses.values():
            if status.worker_status and status.worker_status.status == "error":
                return True
        return False

    @abstractmethod
    def is_complete(self) -> bool:
        """Check if trial is complete."""
        pass


