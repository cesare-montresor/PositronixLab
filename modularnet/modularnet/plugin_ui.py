import logging
import time
from typing import TYPE_CHECKING, Any, Dict, Optional, List

from modularnet.modularspace.modularoverspace import ModularOverSpace

from modularnet.modularui.modularui import ModularUI
from modularnet.plugin.plugin_worker import PluginWorker
from modularnet.trial.trial_config import TrialConfig
from modularnet.plugin import BasePlugin, PluginDescriptor

from dash import Dash, html

logger = logging.getLogger(__name__)


class UIPlugin(BasePlugin):
    

    @classmethod
    def default_options(cls) -> Dict[str, Any]:
        return {
            "host": None,
            "port": None,
        }


    @classmethod
    def Descriptor(cls, options: Dict[str, Any] = None) -> PluginDescriptor:
        return super().Descriptor(
            version="1.0.0",
            author="ModularNet Team",
            description="Receives metrics from the model and publishes them to the UI.",
            options=options
        )

    def __init__(
        self,
        worker: PluginWorker,
        plugin_id: str,
        options: Dict[str, Any],        
    ):
        super().__init__(worker, plugin_id, options )
        self.ui: ModularUI = None  # Will be initialized in on_start()
        self.trial_config: TrialConfig = None  # Will be set in on_start via env_get("trial_config")
        self.host = options.get("host")
        self.port = options.get("port")


    def on_start(self) -> None:
        self.trial_config = self.env_get("trial_config")
        self.ui = ModularUI(host=self.host, port=self.port)
        self.ui.run()

    def main(self) -> None:
        all_vars = self.env_get()
        self.ui.update_layout(all_vars=all_vars)
        time.sleep(1.0)  # Allow other plugins to access this plugin's state
        
    def on_stop(self) -> None:
        self.ui.stop() 
