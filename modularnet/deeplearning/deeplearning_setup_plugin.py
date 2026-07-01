
"""
DeepLearning Setup Plugin: Initializes model and data provider.

This plugin creates a MetaModel and DataProvider based on TrialConfig,
then sets them in the environment for other plugins to use.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, List

    from modularnet.plugin.plugin_worker import PluginWorker
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
        worker: PluginWorker,
        plugin_id: str,
        options: Dict[str, Any],        
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
            manager: Optional ManagerChannel for direct manager access
        """
        super().__init__(worker, plugin_id, options )
        
        
        self.device = options.get("device")
        self.num_workers_data = options.get("num_workers_data", 0)
        self.debug = options.get("debug", False)
        self.trial_config: TrialConfig = None  # Will be set in on_start via env_get("trial_config")

    def on_start(self) -> None:
        """Create model and dataprovider, then set them in environment."""
        self.trial_config: TrialConfig = self.env_get("trial_config")

        logger.info(f"[{self.plugin_id}] Starting deeplearning setup...")
        try:
            
            if not self.trial_config:
                raise ValueError("trial_config not provided in config")
            
            # Create model first to determine device
            model = MetaModel(self.plugin_id, self.trial_config, device=self.device)
            if self.debug:
                model.print()

            device = model.device
            
            # Pass model device to dataset
            dataprovider = DataProvider.from_config(
                self.trial_config, 
                device=device, 
                num_workers=self.num_workers_data
            )

            self.env_set("model", model)
            self.env_set("dataprovider", dataprovider)
            
            # Note: Environment variables are set by PluginManager via set_env_variable()
            # This plugin provides setup results; manager handles environment distribution
            logger.info(f"[{self.plugin_id}] Model and dataprovider ready for distribution")
            
            logger.info(f"[{self.plugin_id}] Setup complete")
            
        except Exception as e:
            logger.error(f"[{self.plugin_id}] Setup failed: {e}")
            self.on_error(e, "setup")
            raise

    def on_stop(self) -> None:
        """Cleanup."""
        logger.info(f"[{self.plugin_id}] Setup plugin stopped")

