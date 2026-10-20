from calendar import c
import re
import logging
import time

import pytest

from positronixlab.deeplearning.deeplearning_train_plugin import DeeplearningTrainPlugin
from positronixlab.deeplearning.deeplearning_setup_plugin import DeeplearningSetupPlugin
from positronixlab.trial import BaseTrial, TrialConfig, TrialManager

logger = logging.getLogger(__name__)


class TrialTrainSimple(BaseTrial):
    def __init__(self, config: TrialConfig, trial_manager: TrialManager):
        super().__init__(config, trial_manager)
        options = {
            "trial_config": self.config,
            "device": None,  # Use default device
            "num_workers_data": 0,  # No parallel data loading for test
            "debug": True
        }
        self.error = False
        self.dl_setup = DeeplearningSetupPlugin.Descriptor(options=options)
        self.dl_train = DeeplearningTrainPlugin.Descriptor(options=options)

        # Pass config to plugins
        self.load_plugin(self.dl_setup, options)
        self.load_plugin(self.dl_train, options)

    def is_complete(self) -> bool:
        # For testing purposes, we can just check if the training plugin has finished
        statuses = self.plugin_manager.plugin_status_all()

        alive = list(filter(lambda s: s.process_alive, statuses.values()))
        if len(alive) == 0: return True

        return False
    
    


class TestTrialTrainDry:
    def test_setup_plugin_loads(self):
        """Test that setup plugin can be loaded and started."""
        config = TrialConfig()
        trial_manager = TrialManager()
        trial = TrialTrainSimple(config, trial_manager)
        
        # Just verify plugins loaded without error
        assert len(trial.plugin_manager._plugins) == 2
        print("Both plugins loaded successfully")

    def test_base_integration(self):
        """Test basic trial execution with setup plugin."""
        self.config = TrialConfig()
        self.trial_manager = TrialManager()
        self.trial = TrialTrainSimple(self.config, self.trial_manager)
        
        # Start plugins
        self.trial.run()
        
        # Give setup plugin time to complete on_start
        import time
        time.sleep(2)
        
        statuses = self.trial.plugin_manager.plugin_status_all()
        for plugin_id, status in statuses.items():
            print(f"{plugin_id}: alive={status.process_alive}, worker_status={status.worker_status}")
            if status.worker_status:
                assert status.worker_status.state != "error", f"Plugin {plugin_id} has errors"

        print("Trial setup completed successfully")
        

class TestTrialTrain:
    def test_train(self):
        """Test basic trial execution with setup and training plugins."""
        config = TrialConfig()
        config.dataset_name = 'iris'  # Use iris dataset for quick testing
        config.epochs = 1  # Use 1 epoch for quick test
        trial_manager = TrialManager()
        trial = TrialTrainSimple(config, trial_manager)
        
        # Start plugins
        logger.info("Starting plugins...")
        trial.run()
        
        # Give plugins time to initialize
        time.sleep(0.5)
        
        # Monitor plugin execution with timeout
        max_wait = 60  # 60 second timeout
        start_time = time.time()
        last_status_time = time.time()
        
        while time.time() - start_time < max_wait:
            statuses = trial.plugin_manager.plugin_status_all()
            
            # Log status every 3 seconds
            current_time = time.time()
            if current_time - last_status_time >= 3:
                logger.info(f"Plugin status check ({current_time - start_time:.1f}s):")
                for plugin_id, status in statuses.items():
                    msg = f"  {plugin_id}: process_alive={status.process_alive}"
                    if status.process_pid:
                        msg += f" pid={status.process_pid}"
                    if status.worker_status:
                        msg += f" worker_status={status.worker_status.state}"
                        if status.worker_status.errors:
                            msg += f" ERRORS: {status.worker_status.errors}"
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
        
        logger.info("Trial execution completed successfully")
        

    

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
