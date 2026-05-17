
import pytest
import torch

from modularnet.trial.trial_config import TrialConfig
from modularnet.trial.trail import TrialDeepLearningCore
from modularnet.trial.trial_manager import TrialManager
from modularnet.plugin_old.plugin_package import (
    PluginDescriptor,
    PluginPackage,
    ServicePlugins,
    SequentialPlugins,
    ParallelPlugins,
)


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
    
    @pytest.mark.integration
    def test_trial_run_iris_with_plugin_package(self):
        """Test trial initialization using PluginPackage configuration."""
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
        
        # Create plugin descriptor for the trial core
        trial_plugin = PluginDescriptor(
            plugin_class=TrialDeepLearningCore,
            name="trial_deeplearning_core",
            options={"config": config},
            version="1.0",
        )
        
        # Create plugin package with trial as a sequential step
        plugin_package = PluginPackage()
        plugin_package.add_step(trial_plugin)
        
        # Create trial manager
        trial_manager = TrialManager(TrialDeepLearningCore, config, trial_manager=None)
        
        # Initialize trial from descriptor - should not raise
        trial = TrialDeepLearningCore(config, trial_manager)
        
        # Run trial - should complete without errors
        trial.run()
        
        # Verify trial has results
        assert trial.results is not None
        assert 'train_loss' in trial.results
        assert 'train_accuracy' in trial.results
        assert trial.results['train_loss'] > 0
        assert 0 <= trial.results['train_accuracy'] <= 1
        
        # Verify plugin package structure
        assert len(plugin_package.steps.items) == 1
        assert isinstance(plugin_package.steps.items[0], PluginDescriptor)
        assert plugin_package.steps.items[0].name == "trial_deeplearning_core"
    
    @pytest.mark.integration
    def test_trial_run_iris_plugin_package_serialization(self):
        """Test PluginPackage serialization/deserialization with trial config."""
        # Create minimal config for quick test
        config = TrialConfig(options={
            'dataset_name': 'iris',
            'model_type': 'fc',
            'train_epochs': 1,
            'batch_size': 16,
            'learning_rate': 0.01,
            'input_num_layers': 1,
            'output_num_layers': 1,
            'input_hidden_dim': 16,
            'output_hidden_dim': 8,
        })
        
        # Create plugin descriptor for the trial core
        trial_plugin = PluginDescriptor(
            plugin_class=TrialDeepLearningCore,
            name="trial_deeplearning_core",
            options={"config": config},
            version="1.0",
        )
        
        # Create plugin package with trial
        original_package = PluginPackage()
        original_package.add_step(trial_plugin)
        
        # Serialize to JSON
        json_str = original_package.to_json()
        assert json_str is not None
        assert "trial_deeplearning_core" in json_str
        
        # Deserialize from JSON
        restored_package = PluginPackage.from_json(json_str)
        
        # Verify structure is preserved
        assert len(restored_package.steps.items) == 1
        restored_descriptor = restored_package.steps.items[0]
        assert isinstance(restored_descriptor, PluginDescriptor)
        assert restored_descriptor.name == "trial_deeplearning_core"
        assert restored_descriptor.version == "1.0"
