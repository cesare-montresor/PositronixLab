
import pytest
import torch

from modularnet.config.trial_config import TrialConfig
from modularnet.trial.trail import TrialDeepLearningCore
from modularnet.trial.trial_manager import TrialManager


class TestTrialDeepLearningCoreIntegration:
    """Integration tests with real components and training."""
    
    @pytest.mark.integration
    def test_trial_run_iris(self):
        """Test trial initialization with standard trial config."""
        # Create minimal config for quick test
        config = TrialConfig(options={
            'dataset_name': 'iris',  # Smallest dataset for fast testing
            'model_type': 'fc',
            'train_epochs': 1,
            'batch_size': 16,
            'learning_rate': 0.01,
            'input_num_layers': 1,
            'output_num_layers': 1,
            'input_hidden_dim': 16,
            'output_hidden_dim': 8,
        })
        
        # Create trial manager
        trial_manager = TrialManager(TrialDeepLearningCore, config, trial_manager=None)
        
        # Initialize trial - should not raise
        trial = TrialDeepLearningCore(config, trial_manager)
        
        # Run trial - should complete without errors
        trial.run()
        # Verify trial has results
        assert trial.results is not None
        assert 'train_loss' in trial.results
        assert 'train_accuracy' in trial.results
        assert trial.results['train_loss'] > 0
        assert 0 <= trial.results['train_accuracy'] <= 1