from abc import ABC, abstractmethod
import logging
import time
from typing import Dict, Any, Optional

from positronixlab.pipeline.pipeline_config import PipelineConfig
from parallelazy import PluginManager, PluginDescriptor

logger = logging.getLogger(__name__)


class Pipeline(ABC):
    """Base trial class."""
    
    def __init__(self, config: PipelineConfig ):
        self.config = config
        
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
        return plugin_id
    
    def run(self) -> None:
        """Run all loaded plugins."""
        self.plugin_manager.start_all_plugins()

    def is_error(self) -> bool:
        """Check if any plugin is in error state."""
        statuses = self.plugin_manager.plugin_status_all()
        for status in statuses.values():
            if status.worker_status and status.worker_status.state == "error":
                return True
        return False

    
    def get_results(self) -> Optional[Dict[str, Any]]:
        """Get results from trial."""
        self.results = self.plugin_manager.env_get()
        return self.results

    @abstractmethod
    def is_complete(self) -> bool:
        """Check if trial is complete."""
        pass

    

    