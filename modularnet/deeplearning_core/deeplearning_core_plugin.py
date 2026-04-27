



from modularnet.config.trial_config import TrialConfig
from modularnet.dataprovider.dataprovider import DataProvider
from modularnet.metamodel.metamodel import MetaModel
from modularnet.deeplearning_core.deeplearning_core import DeeplearningCore

import torch
from typing import Optional

from modularnet.plugin.plugin import Plugin


class DeeplearningSetupPlugin(Plugin):
    """
    Plugin that creates and registers the model and dataprovider in the environment.
    Runs once during setup, then exits.
    """
   
    def __init__(
        self,
        name: str,
        config: TrialConfig,
        device: Optional[torch.device] = None,
        num_workers_data: int = 0,
        debug: bool = False,
        fatal_errors: bool = False,
    ):
        super().__init__(name=name, fatal_errors=fatal_errors)
        self.config = config
        self.device = device
        self.num_workers_data = num_workers_data
        self.debug = debug

    def onStart(self):
        """Create model and dataprovider, set them in environment, then stop."""
        try:
            # Create model first to determine device
            model = MetaModel(self.name, self.config, device=self.device)
            if self.debug:
                model.print()

            device = model.device
            
            # Pass model device to dataset
            dataset = DataProvider.from_config(
                self.config, 
                device=device, 
                num_workers=self.num_workers_data
            )
            
            # Set environment variables for other plugins
            self.env_set("model", model)
            self.env_set("dataprovider", dataset)
            
            self.logger.info(f"Setup complete. Model and dataprovider initialized.")
            
        except Exception as e:
            self.logger.error(f"Setup failed: {e}")
            if self.fatal_errors:
                raise

    def onUpdate(self):
        """Stop the plugin after setup."""
        self.stop()

    def onStop(self):
        """Cleanup."""
        pass


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
        self.env_declare("model", watch=True, on_change=self.on_model_changed)
        self.env_declare("dataprovider", watch=True, on_change=self.on_dataprovider_changed)
        # Space is optional - can be set by setup plugin or external
        self.env_declare("space", watch=True, on_change=self.on_space_changed, default=None)
        
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
            
            core = DeeplearningCore(
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
        """Idle loop - main work happens in callbacks."""
        pass

    def onStop(self):
        """Cleanup."""
        pass
