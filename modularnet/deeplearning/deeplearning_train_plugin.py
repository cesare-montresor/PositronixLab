



from modularnet.trial.trial_config import TrialConfig
from modularnet.dataprovider.dataprovider import DataProvider
from modularnet.metamodel.metamodel import MetaModel
from modularnet.deeplearning.deeplearning_common import DeeplearningCommon

import torch
from typing import Optional

from modularnet.plugin_old.plugin import Plugin
from modularnet.plugin_old.plugin_package import PluginDescriptor





class DeeplearningTrainPluginDescriptor(PluginDescriptor):
    """
    Descriptor for the DeeplearningTrainPlugin, which sets up the model and dataprovider in the environment,
    then runs the training process.
    """
    def __init__(self, name: str, options: dict, version: str = "1.0", metadata: Optional[dict] = None):
        super().__init__(
            plugin_class=DeeplearningTrainPlugin,
            name=name,
            options={"options": options},
            version=version,
            metadata=metadata or {}
        )


class DeeplearningTrainPlugin(Plugin):
    """
    Plugin that waits for model and dataprovider to be set in environment,
    then runs the training process.
    """
   
    def __init__(
        self,
        name: str,
        config: TrialConfig,
        lr: float = 0.001,
        epochs: int = 20,
        autoencoder: bool = False,
        fatal_errors: bool = False,
    ):
        super().__init__(name=name, fatal_errors=fatal_errors)
        self.config = config
        self.lr = lr
        self.epochs = epochs
        self.autoencoder = autoencoder
        
        self.model = None
        self.dataprovider = None
        self.space = None
        self.training_started = False

    def onStart(self):
        """Declare the environment variables we depend on."""
        # Declare that we watch for model and dataprovider changes


        self.model = self.env_get("model")
        self.dataprovider = self.env_get("dataprovider")
        self.space = self.env_get("space")  # Space is optional

        self.env_watch("model", on_change=self.on_model_changed)
        self.env_watch("dataprovider", on_change=self.on_dataprovider_changed)
        # Space is optional - can be set by setup plugin or external
        self.env_watch("space", on_change=self.on_space_changed)



        
        self.logger.info("Training plugin started, waiting for model and dataprovider...")

    def on_model_changed(self, old, new):
        """Callback when model is set."""
        self.model = new
        self.logger.info("Model received from environment.")
        self.check_and_start_training()

    def on_dataprovider_changed(self, old, new):
        """Callback when dataprovider is set."""
        self.dataprovider = new
        self.logger.info("DataProvider received from environment.")
        self.check_and_start_training()

    def on_space_changed(self, old, new):
        """Callback when space is set (optional)."""
        self.space = new
        if new is not None:
            self.logger.info("ModularSpace received from environment.")
        else:
            self.logger.info("No ModularSpace provided (optional).")
        self.check_and_start_training()

    def check_and_start_training(self):
        """Check if both model and dataprovider are available, then start training."""
        if self.model is not None and self.dataprovider is not None and not self.training_started:
            self.training_started = True
            self.logger.info("Both model and dataprovider available. Starting training...")
            self.run_training()

    def run_training(self):
        """Run the training process."""
        try:
            # Create a DeeplearningCore instance with the model and dataprovider
            # Space is optional - use None if not provided
            space = self.space if self.space is not None else None
            
            core = DeeplearningCommon(
                config=self.config,
                model=self.model,
                dp=self.dataprovider,
                space=space,
                draft=False
            )
            
            # Run training
            core.train(lr=self.lr, epochs=self.epochs, autoencoder=self.autoencoder)
            
            self.logger.info("Training completed successfully.")
            self.stop()
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            if self.fatal_errors:
                raise
            self.stop()

    def onUpdate(self):
        #TODO: place training loop here
        #TODO: extract and inject into the pub/sub system partial metrics, logs, checkpoints, etc.
        
        pass

    def onStop(self):
        """Cleanup."""
        pass


