import logging
import os
from pathlib import Path
import sys
import time


from modularnet.utils import utils
from modularnet.modularnet.plugin_cluster import ClusterPlugin
from modularnet.modularnet.plugin_overspace import OverspacePlugin
from modularnet.modularnet.plugin_space import ModularSpacePlugin
from modularnet.modularnet.plugin_train import ObservableTrainPlugin
from modularnet.modularnet.plugin_ui import UIPlugin
from modularnet.modularnet.plugin_compositionality import CompositionalityPlugin
from modularnet.trial import BaseTrial, TrialConfig, TrialManager
from modularnet.utils.utils import is_debugpy_attached

logger = logging.getLogger(__name__)

class TrailModularNetV1(BaseTrial):
    def __init__(self, config: TrialConfig, trial_manager: TrialManager):
        super().__init__(config, trial_manager)
        self.error = False

        logger.setLevel(logging.DEBUG)
        self.plugin_manager.worker_pydebug = True if is_debugpy_attached() else False  # Set a debug port for worker processes (optional)
        self.plugin_manager.env_set("trial_config", self.config)  # Make config available to plugins
        
        self.workdir = os.path.dirname(os.path.realpath(__file__))
        self.ui_save_path = Path(self.workdir,"dump","data.pickle")

        self.plugins = [
            UIPlugin.Descriptor({ "save_path": self.ui_save_path }),
            ObservableTrainPlugin.Descriptor(),
            ModularSpacePlugin.Descriptor(),
            ClusterPlugin.Descriptor(),
            #CompositionalityPlugin.Descriptor(),
            #OverspacePlugin.Descriptor()
        ]
        
        self.plugin_manager.load_plugins(self.plugins, parallel=True)  # Load plugins in the main process for debugging purposes

    def is_complete(self) -> bool:
        # For testing purposes, we can just check if the training plugin has finished
        statuses = self.plugin_manager.plugin_status_all()
        
        alive = list(filter(lambda s: s.process_alive, statuses.values()))
                
        if len(alive) == 0: 
            logger.error(f"All plugins have terminated:")
            return True  

        if self.plugin_manager.env_get("compositionality_completed"):
            logger.info("Compositionality plugin has completed.")
            return True

        return False

    def shutdown(self):
        all_vars = self.plugin_manager.env_get()
        logger.info(f"Trial environment variables: {all_vars}")
        utils.save_data('all_vars.pkl', all_vars)

        logger.info("Shutting down trial...")
        self.plugin_manager.stop_all_plugins()
        self.plugin_manager.unload_all_plugins()
        logger.info("Trial shutdown complete.")



def main():
    """Main function to run the test."""
    logging.basicConfig(level=logging.INFO)
    test_train()




def test_train():
    """Test basic trial execution with setup and training plugins."""
    #clear_torch_compile_cache(verbose=True)  # Clear torch compile cache before starting the trial
    verbose=False
    config = TrialConfig()
    config.dataset_name = 'mnist'  # Use mnist dataset for quick testing
    #config.epochs = 1  # Use 3 epochs for quick test
    trial_manager = TrialManager()
    trial = TrailModularNetV1(config, trial_manager)
    

    
    
    # Start plugins
    logger.info("Starting plugins...")
    trial.run()
    
    # Give plugins time to initialize
    time.sleep(0.5)
    
    # Monitor plugin execution with timeout
    max_wait = 600  # 60 second timeout
    start_time = time.time()
    last_status_time = time.time()
    
    while time.time() - start_time < max_wait:
        statuses = trial.plugin_manager.plugin_status_all()
        
        # Log status every 3 seconds
        current_time = time.time()
        if current_time - last_status_time >= 3:
            if verbose:
                logger.info(f"Plugin status check ({current_time - start_time:.1f}s):")

            for plugin_id, status in statuses.items():
                msg = f"[{plugin_id}] STATUS: "
                if status.worker_status:
                    if status.worker_status.errors:
                        msg += f" ERRORS: {status.worker_status.errors}"
                    else:
                        msg += f" STATE: {status.worker_status.state}"
                else:
                    if status.process_alive:
                        verbose = True
                        msg += " ALIVE but status unknown"
                    else:
                        msg += " Terminated."

                if verbose:
                    logger.info(msg)
                
                # Check for errors immediately
                if status.worker_status and status.worker_status.state == "error":
                    logger.error(f"Plugin {plugin_id} encountered error: {status.worker_status.errors}")
                    raise RuntimeError(f"Plugin {plugin_id} failed: {status.worker_status.errors}")



            last_status_time = current_time
        
        # Check if complete
        if trial.is_complete():
            logger.info("All plugins completed successfully")
            break
        
        # Small sleep to avoid busy waiting
        time.sleep(0.2)
    
    else:
        # Timeout occurred
        statuses = trial.plugin_manager.plugin_status_all()
        logger.error(f"Test timed out after {max_wait}s")
        for plugin_id, status in statuses.items():
            logger.error(f"  {plugin_id}: alive={status.process_alive}, status={status.worker_status}")
        raise TimeoutError(f"Test execution exceeded {max_wait}s timeout")

    trial.shutdown()
    
    logger.info("Trial execution completed successfully")
    sys.exit(0)
        

    

if __name__ == "__main__":
    main()