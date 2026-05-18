
"""
DeepLearning Training Plugin: Trains the model using data provider.

This plugin:
1. Waits for model and dataprovider to be set in environment
2. Optionally waits for a ModularSpace
3. Runs training
4. Stops after training completes
"""

import logging
from typing import Any, Dict, Optional, List
import torch

from modularnet.trial.trial_config import TrialConfig
from modularnet.dataprovider.dataprovider import DataProvider
from modularnet.metamodel.metamodel import MetaModel
from modularnet.deeplearning.deeplearning_common import DeeplearningCommon
from modularnet.plugin import BasePlugin, PluginDescriptor, Subscription, EnvUpdate

logger = logging.getLogger(__name__)



class DeeplearningTrainPluginDescriptor(PluginDescriptor):
    """
    Descriptor for the DeeplearningTrainPlugin.
    
    Provides metadata for plugin registration and instantiation.
    """
    def __init__(
        self,
        name: str = "deeplearning-train",
        config: Optional[Dict[str, Any]] = None,
        version: str = "1.0",
        metadata: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            package="modularnet.deeplearning.deeplearning_train_plugin",
            class_name="DeeplearningTrainPlugin",
            name=name,
            options=config or {},
            version=version,
            metadata=metadata or {}
        )


class DeeplearningTrainPlugin(BasePlugin):
    """
    Plugin that runs training after model and dataprovider are available.
    
    This plugin:
    1. Subscribes to environment variable changes for model, dataprovider, space
    2. Waits for both model and dataprovider to be available
    3. Runs training
    4. Stops after training completes
    """
   
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
            orchestrator_handle: Ray handle to orchestrator
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
        logger.info(f"[{self.plugin_id}] Training plugin started, subscribing to environment variables...")
        
        # Try to get initial values from orchestrator (may not be set yet)
        if self._orchestrator:
            try:
                import ray
                self.model = ray.get(self._orchestrator.get_env_variable.remote("model"))
                self.dataprovider = ray.get(self._orchestrator.get_env_variable.remote("dataprovider"))
                self.space = ray.get(self._orchestrator.get_env_variable.remote("space"))
            except Exception as e:
                logger.debug(f"[{self.plugin_id}] Initial env values not yet available: {e}")
        
        self.check_and_start_training()

    def on_model_changed(self, update: EnvUpdate) -> None:
        """Callback when model environment variable is updated."""
        self.model = update.value
        logger.info(f"[{self.plugin_id}] Model received from environment")
        self.check_and_start_training()

    def on_dataprovider_changed(self, update: EnvUpdate) -> None:
        """Callback when dataprovider environment variable is updated."""
        self.dataprovider = update.value
        logger.info(f"[{self.plugin_id}] DataProvider received from environment")
        self.check_and_start_training()

    def on_space_changed(self, update: EnvUpdate) -> None:
        """Callback when space environment variable is updated."""
        self.space = update.value
        if update.value is not None:
            logger.info(f"[{self.plugin_id}] ModularSpace received from environment")
        else:
            logger.info(f"[{self.plugin_id}] No ModularSpace provided (optional)")
        self.check_and_start_training()

    def check_and_start_training(self) -> None:
        """Check if both model and dataprovider are available, then start training."""
        if (self.model is not None and self.dataprovider is not None 
                and not self.training_started):
            self.training_started = True
            logger.info(f"[{self.plugin_id}] Both model and dataprovider available. Starting training...")
            try:
                self.run_training()
            except Exception as e:
                logger.error(f"[{self.plugin_id}] Training failed: {e}")
                self.on_error(e, "run_training")

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

