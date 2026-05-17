



from modularnet.trial.trial_config import TrialConfig
from modularnet.dataprovider.dataprovider import DataProvider
from modularnet.metamodel.metamodel import MetaModel
from modularnet.deeplearning.deeplearning_common import DeeplearningCommon

import torch
from typing import Optional

from modularnet.plugin_old.plugin import Plugin
from modularnet.plugin_old.plugin_package import PluginDescriptor





class DeeplearningSetupPluginDescriptor(PluginDescriptor):
    """
    Descriptor for the DeeplearningSetupPlugin, which sets up the model and dataprovider in the environment,
    then runs the training process.
    """
    def __init__(self, name: str, options: dict, version: str = "1.0", metadata: Optional[dict] = None):
        super().__init__(
            plugin_class=DeeplearningSetupPlugin,
            name=name,
            options={"options": options},
            version=version,
            metadata=metadata or {}
        )


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
