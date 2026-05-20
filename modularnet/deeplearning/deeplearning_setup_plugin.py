
"""
DeepLearning Setup Plugin: Initializes model and data provider.

This plugin creates a MetaModel and DataProvider based on TrialConfig,
then sets them in the environment for other plugins to use.
"""

import logging
from typing import Any, Dict, Optional, List

from modularnet.trial.trial_config import TrialConfig
from modularnet.dataprovider.dataprovider import DataProvider
from modularnet.metamodel.metamodel import MetaModel
from modularnet.plugin import BasePlugin, PluginDescriptor, Subscription

logger = logging.getLogger(__name__)


class DeeplearningSetupPlugin(BasePlugin):
    """
    Plugin that creates and sets up the model and data provider.
    
    This plugin:
    1. Creates a MetaModel based on TrialConfig
    2. Creates a DataProvider
    3. Sets both in the environment for other plugins
    4. Exits after setup is complete
    """

    @classmethod
    def Descriptor(cls, options: Dict[str, Any] = None) -> PluginDescriptor:
        return super().Descriptor(
            version="1.0.0",
            author="ModularNet Team",
            description="Plugin that sets up the model and data provider for deep learning trials.",
            options=options
        )
    


    def __init__(
        self,
        plugin_id: str,
        config: Dict[str, Any],
        orchestrator_handle: Optional[Any] = None,
    ):
        """
        Initialize the setup plugin.
        
        Args:
            plugin_id: Unique plugin identifier
            config: Configuration containing:
                - trial_config: TrialConfig instance
                - device: Optional torch device
                - num_workers_data: Number of data workers
                - debug: Enable debug logging
            orchestrator_handle: Orchestrator reference for multiprocessing backend (not used directly)
        """
        super().__init__(plugin_id, config, orchestrator_handle)
        
        self.trial_config: Optional[TrialConfig] = config.get("trial_config")
        self.device = config.get("device")
        self.num_workers_data = config.get("num_workers_data", 0)
        self.debug = config.get("debug", False)

    def on_start(self) -> None:
        """Create model and dataprovider, then set them in environment."""
        try:
            logger.info(f"[{self.plugin_id}] Starting deeplearning setup...")
            
            if not self.trial_config:
                raise ValueError("trial_config not provided in config")
            
            # Create model first to determine device
            model = MetaModel(self.plugin_id, self.trial_config, device=self.device)
            if self.debug:
                model.print()

            device = model.device
            
            # Pass model device to dataset
            dataset = DataProvider.from_config(
                self.trial_config, 
                device=device, 
                num_workers=self.num_workers_data
            )
            
            # Set in environment using orchestrator
            if self._orchestrator:
                self._orchestrator.set_env_variable("model", model)
                self._orchestrator.set_env_variable("dataprovider", dataset)
                logger.info(f"[{self.plugin_id}] Model and dataprovider set in environment")
            else:
                logger.warning(f"[{self.plugin_id}] No orchestrator available, cannot set env variables")
            
            logger.info(f"[{self.plugin_id}] Setup complete")
            
        except Exception as e:
            logger.error(f"[{self.plugin_id}] Setup failed: {e}")
            self.on_error(e, "setup")
            raise

    def on_stop(self) -> None:
        """Cleanup."""
        logger.info(f"[{self.plugin_id}] Setup plugin stopped")

    def on_update(self, config: Dict[str, Any]) -> None:
        """Handle configuration updates."""
        super().on_update(config)
        # After setup, additional updates are not expected
        logger.debug(f"[{self.plugin_id}] Config updated: {config}")

    def get_subscriptions(self) -> List[Subscription]:
        """Setup plugin doesn't subscribe to any topics."""
        return []

    def get_env_subscriptions(self) -> List[Subscription]:
        """Setup plugin doesn't watch environment variables."""
        return []

    def get_services(self) -> List[str]:
        """Setup plugin provides no services."""
        return []
