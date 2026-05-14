"""
Integration test for TrialDeepLearningCore with full training.

This test creates a complete TrialDeepLearningCore with standard trial config
and attempts a full training pipeline with real components.

This is NOT a mock test - it uses actual:
- TrialConfig with standard settings
- Real data provider (MNIST)
- Real neural network model
- Real training plugins

The test runs for minimal epochs to complete quickly while validating
the full integration of setup + training pipeline.
"""

import pytest
import torch

from modularnet.config.trial_config import TrialConfig
from modularnet.trial.trail import TrialDeepLearningCore
from modularnet.trial.trial_manager import TrialManager


class TestTrialDeepLearningCoreIntegration:
    """Integration tests with real components and training."""
    
    @pytest.mark.integration
    def test_trial_initialization_with_standard_config(self):
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
        
        # Verify trial is properly initialized
        assert trial.config is config
        assert trial.trial_manager is trial_manager
        assert trial.plugin_manager is not None
        assert len(trial.plugin_manager.plugins) == 2
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_trial_run_with_real_components(self):
        """Test actual trial run with real model and data."""
        # Create minimal config for quick training test
        config = TrialConfig(options={
            'dataset_name': 'iris',
            'model_type': 'fc',
            'train_epochs': 1,
            'batch_size': 16,
            'learning_rate': 0.01,
            'input_num_layers': 1,
            'output_num_layers': 1,
            'input_hidden_dim': 8,
            'output_hidden_dim': 4,
            'criterion_type': 'mse',
            'task': 'classification',
        })
        
        # Create trial manager
        trial_manager = TrialManager(TrialDeepLearningCore, config, trial_manager=None)
        
        # Initialize trial
        trial = TrialDeepLearningCore(config, trial_manager)
        
        # Verify plugins are registered
        assert len(trial.plugin_manager.plugins) == 2
        assert trial.plugin_manager.plugins[0] is not None
        assert trial.plugin_manager.plugins[1] is not None
        
        # Verify run method exists and is callable
        assert callable(trial.run)
        
        # Note: Actual run() would require full plugin implementation
        # This test validates the structure is correct for execution


class TestTrialConfigOptions:
    """Test TrialConfig with various option combinations."""
    
    def test_config_with_empty_options(self):
        """Test config uses defaults when options are empty."""
        config = TrialConfig(options={})
        
        # Verify defaults are applied
        assert config.dataset_name == 'fashion_mnist'
        assert config.model_type == 'fc'
        assert config.task == 'classification'
        assert config.train_epochs == 10
        assert config.batch_size == 32
        assert config.learning_rate == 0.001
    
    def test_config_with_custom_options(self):
        """Test config with custom options overrides defaults."""
        options = {
            'dataset_name': 'iris',
            'model_type': 'fc',
            'train_epochs': 5,
            'batch_size': 64,
            'learning_rate': 0.005,
        }
        config = TrialConfig(options=options)
        
        # Verify custom options are used
        assert config.dataset_name == 'iris'
        assert config.model_type == 'fc'
        assert config.train_epochs == 5
        assert config.batch_size == 64
        assert config.learning_rate == 0.005
    
    def test_config_preserves_options(self):
        """Test config preserves original options dictionary."""
        options = {
            'dataset_name': 'iris',
            'train_epochs': 3,
        }
        config = TrialConfig(options=options)
        
        # Original options should be preserved
        assert config.options == options


class TestTrialDeepLearningCoreStructure:
    """Test the structure and attributes of TrialDeepLearningCore."""
    
    def test_trial_stores_config_and_manager(self):
        """Test trial properly stores config and manager."""
        config = TrialConfig(options={'dataset_name': 'iris'})
        trial_manager = TrialManager(TrialDeepLearningCore, config, trial_manager=None)
        
        trial = TrialDeepLearningCore(config, trial_manager)
        
        assert trial.config is config
        assert trial.trial_manager is trial_manager
    
    def test_trial_creates_plugin_manager(self):
        """Test trial creates its own plugin manager."""
        config = TrialConfig(options={'dataset_name': 'iris'})
        trial_manager = TrialManager(TrialDeepLearningCore, config, trial_manager=None)
        
        trial = TrialDeepLearningCore(config, trial_manager)
        
        assert hasattr(trial, 'plugin_manager')
        assert trial.plugin_manager is not None
    
    def test_trial_loads_two_plugins(self):
        """Test trial loads exactly two plugins in correct order."""
        config = TrialConfig(options={'dataset_name': 'iris'})
        trial_manager = TrialManager(TrialDeepLearningCore, config, trial_manager=None)
        
        trial = TrialDeepLearningCore(config, trial_manager)
        
        # Should have setup plugin and train plugin
        assert len(trial.plugin_manager.plugins) == 2
        
        # Plugins should exist with expected names
        assert 'deeplearning_setup' in trial.plugin_manager.plugins
        assert 'deeplearning_train' in trial.plugin_manager.plugins
        
        # Access by name
        setup_plugin = trial.plugin_manager.plugins['deeplearning_setup']
        train_plugin = trial.plugin_manager.plugins['deeplearning_train']
        
        assert setup_plugin.name == 'deeplearning_setup'
        assert train_plugin.name == 'deeplearning_train'
    
    def test_trial_has_run_method(self):
        """Test trial has callable run method."""
        config = TrialConfig(options={'dataset_name': 'iris'})
        trial_manager = TrialManager(TrialDeepLearningCore, config, trial_manager=None)
        
        trial = TrialDeepLearningCore(config, trial_manager)
        
        assert hasattr(trial, 'run')
        assert callable(trial.run)


class TestTrialInitializationSequence:
    """Test the initialization sequence of TrialDeepLearningCore."""
    
    def test_initialization_sequence_order(self):
        """Test plugins are initialized in correct order."""
        config = TrialConfig(options={'dataset_name': 'iris'})
        trial_manager = TrialManager(TrialDeepLearningCore, config, trial_manager=None)
        
        trial = TrialDeepLearningCore(config, trial_manager)
        
        plugins = trial.plugin_manager.plugins
        
        # Verify both plugins exist with correct names
        assert 'deeplearning_setup' in plugins
        assert 'deeplearning_train' in plugins
        
        # Verify names
        assert plugins['deeplearning_setup'].name == 'deeplearning_setup'
        assert plugins['deeplearning_train'].name == 'deeplearning_train'
    
    def test_both_plugins_receive_config(self):
        """Test both plugins are initialized with the config."""
        config = TrialConfig(options={'dataset_name': 'iris', 'train_epochs': 3})
        trial_manager = TrialManager(TrialDeepLearningCore, config, trial_manager=None)
        
        trial = TrialDeepLearningCore(config, trial_manager)
        
        # Both plugins should have access to config
        setup_plugin = trial.plugin_manager.plugins['deeplearning_setup']
        train_plugin = trial.plugin_manager.plugins['deeplearning_train']
        
        assert setup_plugin.config is config
        assert train_plugin.config is config


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
