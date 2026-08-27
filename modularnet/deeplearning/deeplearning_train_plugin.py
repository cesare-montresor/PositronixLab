
"""
DeepLearning Training Plugin: Trains the model using data provider.

This plugin:
1. Waits for model and dataprovider to be set in environment
2. Optionally waits for a ModularSpace
3. Runs training
4. Stops after training completes
"""

import logging
import time
from typing import Any, Dict, Optional 


from modularnet.plugin.plugin_worker import PluginWorker
from modularnet.trial.trial_config import TrialConfig
from modularnet.dataprovider.dataprovider_factory import DataProviderFactory
from modularnet.metamodel.metamodel import MetaModel
from modularnet.deeplearning.deeplearning_common import DeeplearningCommon
from modularnet.plugin import BasePlugin, PluginDescriptor, Subscription

logger = logging.getLogger(__name__)



class DeeplearningTrainPlugin(BasePlugin):
    """
    Plugin that runs training after model and dataprovider are available.
    
    This plugin:
    1. Subscribes to environment variable changes for model, dataprovider, space
    2. Waits for both model and dataprovider to be available
    3. Runs training
    4. Stops after training completes
    """


    @classmethod
    def Descriptor(cls, options: Dict[str, Any]=None) -> PluginDescriptor:
        return super().Descriptor(
            version="1.0.0",
            author="ModularNet Team",
            description="Plugin that runs training after model and dataprovider are available.",
            options=options,
        )
    
   
    def __init__(
        self,
        worker: PluginWorker,
        plugin_id: str,
        options: Dict[str, Any],
        
    ):
        """
        Initialize the training plugin.
        
        Args:
            plugin_id: Unique plugin identifier
            options: Configuration containing:
                - trial_config: TrialConfig instance
                - lr: Learning rate (default 0.001)
                - epochs: Number of epochs (default 20)
                - autoencoder: Use autoencoder mode (default False)
            worker: PluginWorker instance for managing the plugin
        """
        super().__init__(worker, plugin_id, options)
        
        self.trial_config: TrialConfig = self.env_get("trial_config")
        self.lr = options.get("lr", 0.001)
        self.epochs = options.get("epochs", 20)
        self.autoencoder = options.get("autoencoder", False)
        
        self.model: Optional[MetaModel] = None
        self.dataprovider: Optional[DataProviderFactory] = None
        self.space: Optional[Any] = None


    def on_start(self) -> None:
        """Initialize env subscriptions and wait for dependencies."""
        logger.info(f"[{self.plugin_id}] Training plugin started, waiting for environment variables...")
        
        # Initial values will be set via on_model_changed, on_dataprovider_changed callbacks
        # No need to query orchestrator directly - just wait for subscribed env updates
        
        self.model = self.env_get("model")
        self.dataprovider = self.env_get("dataprovider")
        self.space = self.env_get("space")
        # Subscribe to environment variable changes
        self.env_subscribe("model", self._on_model_changed)
        self.env_subscribe("dataprovider", self._on_dataprovider_changed)
        self.env_subscribe("space", self._on_space_changed)

    def _on_model_changed(self) -> None:
        """Callback when model environment variable is updated."""
        self.model = self.env_get("model")
        logger.info(f"[{self.plugin_id}] Model received from environment")

    def _on_dataprovider_changed(self) -> None:
        """Callback when dataprovider environment variable is updated."""
        self.dataprovider = self.env_get("dataprovider")
        logger.info(f"[{self.plugin_id}] DataProvider received from environment")

    def _on_space_changed(self) -> None:
        """Callback when space environment variable is updated."""
        self.space = self.env_get("space")
        if self.space is not None:
            logger.info(f"[{self.plugin_id}] ModularSpace received from environment")
        else:
            logger.info(f"[{self.plugin_id}] No ModularSpace provided (optional)")

    def check_and_start_training(self) -> None:
        """Check if both model and dataprovider are available, then start training."""
        logging.debug(f"[{self.plugin_id}] Checking if model and dataprovider are available for training...")
        while self.model is None or self.dataprovider is None:
             time.sleep(0.01)  # Wait for dependencies to be set in environment

    
    def main(self) -> None:
        self.check_and_start_training()

        """Run the training process."""
        try:
            if not self.trial_config:
                raise ValueError("trial_config not provided in config")
            
            logger.info(f"[{self.plugin_id}] Starting training loop...")
            
            # Create DeeplearningCommon instance with the model and dataprovider
            # Space is optional - use None if not provided
            core = DeeplearningCommon(
                config=self.trial_config,
                model=self.model,
                dp=self.dataprovider,
                draft=False
            )
            
            # Run training
            core.train(lr=self.lr, epoch_total=self.epochs, autoencoder=self.autoencoder)
            
            logger.info(f"[{self.plugin_id}] Training completed successfully")
            
        except Exception as e:
            logger.error(f"[{self.plugin_id}] Training failed: {e}")
            raise

    def on_stop(self) -> None:
        """Cleanup."""
        logger.info(f"[{self.plugin_id}] Training plugin stopped")
