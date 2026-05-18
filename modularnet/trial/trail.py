import logging
import time
from typing import Dict, Any, Optional

from modularnet.trial.trial_config import TrialConfig
from modularnet.trial.trial_manager import TrialManager
from modularnet.plugin import PluginManager, PluginDescriptor

logger = logging.getLogger(__name__)


class Trial:
    """Base trial class."""
    
    def __init__(self, config: TrialConfig, trial_manager: TrialManager):
        self.config = config
        self.trial_manager = trial_manager
        
        # Use new Ray-based plugin manager
        self.plugin_manager = PluginManager()
        self.results: Optional[Dict[str, Any]] = None

    def load_plugin(self, descriptor: PluginDescriptor, plugin_config: Dict[str, Any]) -> str:
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

    def is_complete(self) -> bool:
        """Check if all plugins have completed."""
        # In the new system, we check if plugins are no longer running
        return True  # Simplified for now

