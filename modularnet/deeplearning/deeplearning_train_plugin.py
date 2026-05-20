
"""
DeepLearning Training Plugin: Trains the model using data provider.

This plugin:
1. Waits for model and dataprovider to be set in environment
2. Optionally waits for a ModularSpace
3. Runs training
4. Stops after training completes
"""

import logging
from time import sleep
from typing import Any, Dict, Optional, List
import torch

from modularnet.trial.trial_config import TrialConfig
from modularnet.dataprovider.dataprovider import DataProvider
from modularnet.metamodel.metamodel import MetaModel
from modularnet.deeplearning.deeplearning_common import DeeplearningCommon
from modularnet.plugin import BasePlugin, PluginDescriptor, Subscription, EnvUpdate

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
        plugin_id: str,
        config: Dict[str, Any],
        orchestrator_handle: Optional[Any] = None,
    ):
        """
        Initialize the training plugin.
        
        Args:
            plugin_id: Unique plugin identifier
            config: Configuration containing:
                - trial_config: TrialConfig instance
                - lr: Learning rate (default 0.001)
                - epochs: Number of epochs (default 20)
                - autoencoder: Use autoencoder mode (default False)
            orchestrator_handle: Orchestrator reference for multiprocessing backend (not used directly)
        """
        super().__init__(plugin_id, config, orchestrator_handle)
        
        self.trial_config: Optional[TrialConfig] = config.get("trial_config")
        self.lr = config.get("lr", 0.001)
        self.epochs = config.get("epochs", 20)
        self.autoencoder = config.get("autoencoder", False)
        
        self.model: Optional[MetaModel] = None
        self.dataprovider: Optional[DataProvider] = None
        self.space: Optional[Any] = None
        self.training_started = False

    def on_start(self) -> None:
        """Initialize env subscriptions and wait for dependencies."""
        logger.info(f"[{self.plugin_id}] Training plugin started, waiting for environment variables...")
        
        # Initial values will be set via on_model_changed, on_dataprovider_changed callbacks
        # No need to query orchestrator directly - just wait for subscribed env updates
        self.check_and_start_training()
        self.run_training()

    def on_model_changed(self, update: EnvUpdate) -> None:
        """Callback when model environment variable is updated."""
        self.model = update.value
        logger.info(f"[{self.plugin_id}] Model received from environment")

    def on_dataprovider_changed(self, update: EnvUpdate) -> None:
        """Callback when dataprovider environment variable is updated."""
        self.dataprovider = update.value
        logger.info(f"[{self.plugin_id}] DataProvider received from environment")

    def on_space_changed(self, update: EnvUpdate) -> None:
        """Callback when space environment variable is updated."""
        self.space = update.value
        if update.value is not None:
            logger.info(f"[{self.plugin_id}] ModularSpace received from environment")
        else:
            logger.info(f"[{self.plugin_id}] No ModularSpace provided (optional)")

    def check_and_start_training(self) -> None:
        """Check if both model and dataprovider are available, then start training."""
        logging.debug(f"[{self.plugin_id}] Checking if model and dataprovider are available for training...")
        while self.model is None or self.dataprovider is None:
            sleep(0.01)  # Wait for dependencies to be set in environment

    def run_training(self) -> None:
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
                space=self.space,
                draft=False
            )
            
            # Run training
            core.train(lr=self.lr, epochs=self.epochs, autoencoder=self.autoencoder)
            
            logger.info(f"[{self.plugin_id}] Training completed successfully")
            
        except Exception as e:
            logger.error(f"[{self.plugin_id}] Training failed: {e}")
            raise

    def on_stop(self) -> None:
        """Cleanup."""
        logger.info(f"[{self.plugin_id}] Training plugin stopped")

    def on_update(self, config: Dict[str, Any]) -> None:
        """Handle configuration updates."""
        super().on_update(config)
        logger.debug(f"[{self.plugin_id}] Config updated")

    def get_subscriptions(self) -> List[Subscription]:
        """Training plugin doesn't subscribe to any topics."""
        return []

    def get_env_subscriptions(self) -> List[Subscription]:
        """Subscribe to model, dataprovider, and optional space environment variables."""
        return [
            Subscription(name="model", callback="on_model_changed"),
            Subscription(name="dataprovider", callback="on_dataprovider_changed"),
            Subscription(name="space", callback="on_space_changed"),
        ]

    def get_services(self) -> List[str]:
        """Training plugin provides no services."""
        return []

