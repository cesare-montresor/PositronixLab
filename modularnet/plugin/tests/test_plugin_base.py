"""
Comprehensive tests for PluginBase and PluginManagerBase.
"""

import pytest
import time
import threading
from torch import multiprocessing as mp

from modularnet.plugin.plugin_base import PluginBase, PluginManagerBase


class SimplePlugin(PluginBase):
    """Basic test plugin that counts iterations."""
    def __init__(self, name="SimplePlugin", shared_list=None):
        super().__init__(name=name)
        self.shared_list = shared_list
        self.counter = 0
        self.started = False
        self.stopped = False
    
    def onStart(self):
        self.started = True
        if self.shared_list is not None:
            self.shared_list.append(f"{self.name}_started")
    
    def onUpdate(self):
        self.counter += 1
        time.sleep(0.01)
        if self.shared_list is not None and self.counter % 10 == 0:
            self.shared_list.append(f"{self.name}_update_{self.counter}")
    
    def onStop(self):
        self.stopped = True
        if self.shared_list is not None:
            self.shared_list.append(f"{self.name}_stopped_{self.counter}")


class FailingPlugin(PluginBase):
    """Plugin that raises an exception."""
    def __init__(self, fail_on_iteration=1, name="FailingPlugin"):
        super().__init__(name=name, fatal_errors=False)
        self.fail_on_iteration = fail_on_iteration
        self.counter = 0
    
    def onUpdate(self):
        self.counter += 1
        if self.counter == self.fail_on_iteration:
            raise RuntimeError(f"Intentional error on iteration {self.counter}")
        time.sleep(0.01)


class ConfigurablePlugin(PluginBase):
    """Plugin that uses options."""
    def __init__(self, name="ConfigurablePlugin"):
        super().__init__(name=name, options={"mode": "test", "timeout": 5})
        self.option_mode = self.options.get("mode")
        self.option_timeout = self.options.get("timeout")


# ============================================================================
# Tests for PluginBase
# ============================================================================

class TestPluginBase:
    """Test PluginBase lifecycle and functionality."""
    
    def test_plugin_initialization(self):
        """Test that plugins initialize correctly."""
        plugin = SimplePlugin(name="test_plugin")
        assert plugin.name == "test_plugin"
        assert plugin.version is not None
        assert plugin.stop_event is not None
        assert plugin.start_event is not None
    
    def test_plugin_manifest(self):
        """Test plugin manifest generation."""
        plugin = SimplePlugin(name="manifest_test")
        manifest = plugin.manifest()
        assert manifest["name"] == "manifest_test"
        assert "version" in manifest
    
    def test_plugin_start_stop(self):
        """Test starting and stopping a plugin."""
        with mp.Manager() as manager:
            shared_list = manager.list()
            plugin = SimplePlugin(name="starter", shared_list=shared_list)
            
            plugin.start(wait=False)
            time.sleep(0.5)  # Let it run
            plugin.stop(wait=True)
            
            assert len(shared_list) > 0
            assert any("started" in str(item) for item in shared_list)
            assert any("stopped" in str(item) for item in shared_list)
    
    def test_plugin_runs_repeatedly(self):
        """Test that plugin onUpdate is called repeatedly."""
        with mp.Manager() as manager:
            shared_list = manager.list()
            plugin = SimplePlugin(name="repeater", shared_list=shared_list)
            
            plugin.start(wait=False)
            time.sleep(0.3)  # Let it run for a bit
            plugin.stop(wait=True)
            
            # Should have multiple update calls
            update_calls = len([item for item in shared_list if "update_" in str(item)])
            assert update_calls > 0
    
    def test_plugin_custom_options(self):
        """Test that custom options are passed to plugins."""
        plugin = ConfigurablePlugin(name="config_test")
        plugin.options = {"mode": "custom", "timeout": 10}
        
        assert plugin.options["mode"] == "custom"
        assert plugin.options["timeout"] == 10
    
    def test_plugin_error_handling_non_fatal(self):
        """Test that non-fatal errors don't stop the plugin."""
        with mp.Manager() as manager:
            shared_list = manager.list()
            plugin = FailingPlugin(fail_on_iteration=2, name="non_fatal")
            plugin.fatal_errors = False
            plugin.shared_list = shared_list
            
            plugin.start(wait=False)
            time.sleep(0.5)
            plugin.stop(wait=True)
            
            # Plugin should have continued after error
            assert plugin.counter > 2
    
    def test_plugin_name_defaults_to_class_name(self):
        """Test that plugin name defaults to class name."""
        plugin = SimplePlugin()  # no name provided
        assert plugin.name == "SimplePlugin"
    
    def test_plugin_set_manager(self):
        """Test set_manager functionality."""
        plugin = SimplePlugin(name="managed")
        manager = PluginManagerBase()
        
        plugin.set_manager(manager)
        assert plugin.manager is manager
    
    def test_plugin_multiple_start_stop_cycles(self):
        """Test multiple start/stop cycles."""
        with mp.Manager() as manager:
            shared_list = manager.list()
            plugin = SimplePlugin(name="cycler", shared_list=shared_list)
            
            # First cycle
            plugin.start(wait=False)
            time.sleep(0.2)
            plugin.stop(wait=True)
            
            first_count = len(shared_list)
            
            # Second cycle
            plugin = SimplePlugin(name="cycler", shared_list=shared_list)
            plugin.start(wait=False)
            time.sleep(0.2)
            plugin.stop(wait=True)
            
            second_count = len(shared_list)
            # Should have accumulated events from both cycles
            assert second_count > first_count


# ============================================================================
# Tests for PluginManagerBase
# ============================================================================

class TestPluginManagerBase:
    """Test PluginManagerBase registration and control."""
    
    def test_manager_initialization(self):
        """Test manager initializes correctly."""
        manager = PluginManagerBase()
        assert len(manager.plugins) == 0
    
    def test_register_plugin(self):
        """Test registering a plugin."""
        manager = PluginManagerBase()
        plugin = SimplePlugin(name="plugin1")
        
        manager.register_plugin(plugin)
        assert "plugin1" in manager.plugins
        assert manager.plugins["plugin1"] is plugin
    
    def test_register_plugin_custom_name(self):
        """Test registering plugin with custom name."""
        manager = PluginManagerBase()
        plugin = SimplePlugin(name="original_name")
        
        manager.register_plugin(plugin, name="custom_name")
        assert "custom_name" in manager.plugins
        assert "original_name" not in manager.plugins
    
    def test_register_multiple_plugins(self):
        """Test registering multiple plugins."""
        manager = PluginManagerBase()
        p1 = SimplePlugin(name="p1")
        p2 = SimplePlugin(name="p2")
        p3 = SimplePlugin(name="p3")
        
        manager.register_plugin(p1)
        manager.register_plugin(p2)
        manager.register_plugin(p3)
        
        assert len(manager.plugins) == 3
        assert "p1" in manager.plugins
        assert "p2" in manager.plugins
        assert "p3" in manager.plugins
    
    def test_unregister_plugin(self):
        """Test unregistering a plugin."""
        manager = PluginManagerBase()
        plugin = SimplePlugin(name="removable")
        
        manager.register_plugin(plugin)
        assert "removable" in manager.plugins
        
        manager.unregister_plugin(plugin)
        assert "removable" not in manager.plugins
    
    def test_unregister_plugin_custom_name(self):
        """Test unregistering with custom name."""
        manager = PluginManagerBase()
        plugin = SimplePlugin(name="p")
        
        manager.register_plugin(plugin, name="custom")
        manager.unregister_plugin(plugin, name="custom")
        assert "custom" not in manager.plugins
    
    def test_get_plugin(self):
        """Test retrieving a plugin."""
        manager = PluginManagerBase()
        plugin = SimplePlugin(name="retrievable")
        
        manager.register_plugin(plugin)
        retrieved = manager.get_plugin("retrievable")
        assert retrieved is plugin
    
    def test_get_nonexistent_plugin(self):
        """Test retrieving nonexistent plugin returns None."""
        manager = PluginManagerBase()
        result = manager.get_plugin("nonexistent")
        assert result is None
    
    def test_list_plugins(self):
        """Test listing all plugins."""
        manager = PluginManagerBase()
        manager.register_plugin(SimplePlugin(name="a"))
        manager.register_plugin(SimplePlugin(name="b"))
        manager.register_plugin(SimplePlugin(name="c"))
        
        plugin_names = manager.list_plugins()
        assert len(plugin_names) == 3
        assert "a" in plugin_names
        assert "b" in plugin_names
        assert "c" in plugin_names
    
    def test_start_all_plugins(self):
        """Test starting all registered plugins."""
        with mp.Manager() as mp_manager:
            shared_list = mp_manager.list()
            manager = PluginManagerBase()
            
            p1 = SimplePlugin(name="p1", shared_list=shared_list)
            p2 = SimplePlugin(name="p2", shared_list=shared_list)
            
            manager.register_plugin(p1)
            manager.register_plugin(p2)
            manager.start_all_plugins()
            
            time.sleep(0.3)
            
            manager.stop_all_plugins()
            
            # Both should have started
            assert any("p1_started" in str(item) for item in shared_list)
            assert any("p2_started" in str(item) for item in shared_list)
    
    def test_start_specific_plugin(self):
        """Test starting a specific plugin."""
        with mp.Manager() as mp_manager:
            shared_list = mp_manager.list()
            manager = PluginManagerBase()
            
            p1 = SimplePlugin(name="p1", shared_list=shared_list)
            p2 = SimplePlugin(name="p2", shared_list=shared_list)
            
            manager.register_plugin(p1)
            manager.register_plugin(p2)
            manager.start_plugin("p1")  # Start only p1
            
            time.sleep(0.2)
            
            manager.stop_plugin("p1")
            
            # Only p1 should have started
            assert any("p1_started" in str(item) for item in shared_list)
            # p2 should not have started
            assert not any("p2_started" in str(item) for item in shared_list)
    
    def test_stop_all_plugins(self):
        """Test stopping all plugins."""
        with mp.Manager() as mp_manager:
            shared_list = mp_manager.list()
            manager = PluginManagerBase()
            
            manager.register_plugin(SimplePlugin(name="p1", shared_list=shared_list))
            manager.register_plugin(SimplePlugin(name="p2", shared_list=shared_list))
            
            manager.start_all_plugins()
            time.sleep(0.2)
            manager.stop_all_plugins()
            
            # Both should have stopped
            assert any("p1_stopped" in str(item) for item in shared_list)
            assert any("p2_stopped" in str(item) for item in shared_list)
    
    def test_stop_specific_plugin(self):
        """Test stopping a specific plugin."""
        with mp.Manager() as mp_manager:
            shared_list = mp_manager.list()
            manager = PluginManagerBase()
            
            p1 = SimplePlugin(name="p1", shared_list=shared_list)
            p2 = SimplePlugin(name="p2", shared_list=shared_list)
            
            manager.register_plugin(p1)
            manager.register_plugin(p2)
            manager.start_all_plugins()
            
            time.sleep(0.2)
            manager.stop_plugin("p1")  # Stop only p1
            time.sleep(0.1)
            manager.stop_plugin("p2")
            
            # Both should have stopped
            assert any("p1_stopped" in str(item) for item in shared_list)
            assert any("p2_stopped" in str(item) for item in shared_list)
    
    def test_manager_sets_plugin_manager_reference(self):
        """Test that registering sets plugin's manager reference."""
        manager = PluginManagerBase()
        plugin = SimplePlugin(name="managed_plugin")
        
        assert plugin.manager is None
        manager.register_plugin(plugin)
        assert plugin.manager is manager
    
    def test_manager_clears_plugin_manager_on_unregister(self):
        """Test that unregistering clears plugin's manager reference."""
        manager = PluginManagerBase()
        plugin = SimplePlugin(name="p")
        
        manager.register_plugin(plugin)
        assert plugin.manager is manager
        
        manager.unregister_plugin(plugin)
        assert plugin.manager is None


# ============================================================================
# Integration Tests
# ============================================================================

class TestPluginBaseIntegration:
    """Integration tests for PluginBase and PluginManagerBase."""
    
    def test_plugin_lifecycle_full(self):
        """Test complete plugin lifecycle."""
        with mp.Manager() as mp_manager:
            shared_list = mp_manager.list()
            manager = PluginManagerBase()
            plugin = SimplePlugin(name="lifecycle", shared_list=shared_list)
            
            # Register
            manager.register_plugin(plugin)
            assert "lifecycle" in manager.list_plugins()
            
            # Start
            manager.start_plugin("lifecycle")
            time.sleep(0.2)
            
            # Check running
            assert any("lifecycle_started" in str(item) for item in shared_list)
            
            # Stop
            manager.stop_plugin("lifecycle")
            assert any("lifecycle_stopped" in str(item) for item in shared_list)
            
            # Unregister
            manager.unregister_plugin(plugin)
            assert "lifecycle" not in manager.list_plugins()
    
    def test_concurrent_plugin_execution(self):
        """Test that plugins run concurrently."""
        with mp.Manager() as mp_manager:
            shared_list = mp_manager.list()
            manager = PluginManagerBase()
            
            # Create slow and fast plugins
            p1 = SimplePlugin(name="fast", shared_list=shared_list)
            p2 = SimplePlugin(name="slow", shared_list=shared_list)
            
            manager.register_plugin(p1)
            manager.register_plugin(p2)
            manager.start_all_plugins()
            
            time.sleep(0.3)
            manager.stop_all_plugins()
            
            # Both should have updates
            fast_updates = len([item for item in shared_list if "fast_update_" in str(item)])
            slow_updates = len([item for item in shared_list if "slow_update_" in str(item)])
            
            assert fast_updates > 0
            assert slow_updates > 0
    
    def test_plugin_manager_with_many_plugins(self):
        """Test manager with many plugins."""
        with mp.Manager() as mp_manager:
            shared_list = mp_manager.list()
            manager = PluginManagerBase()
            
            # Register 10 plugins
            for i in range(10):
                plugin = SimplePlugin(name=f"plugin_{i}", shared_list=shared_list)
                manager.register_plugin(plugin)
            
            assert len(manager.list_plugins()) == 10
            
            manager.start_all_plugins()
            time.sleep(0.3)
            manager.stop_all_plugins()
            
            # All should have started
            for i in range(10):
                assert any(f"plugin_{i}_started" in str(item) for item in shared_list)
                assert any(f"plugin_{i}_stopped" in str(item) for item in shared_list)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
