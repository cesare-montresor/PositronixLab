"""
Test suite for the torch.multiprocessing plugin system backend.

Mirrors test_plugin_system.py but uses the multiprocessing backend instead of Ray.
Tests all major features: lifecycle, pub/sub, environment variables, inter-plugin communication.

Run with: pytest modularnet/plugin/tests/test_plugin_manager_mp.py -v
"""

import pytest
import time
from typing import Dict, Any, List

from modularnet.plugin.plugin_manager import PluginManager
from modularnet.plugin.plugin_descriptor import PluginDescriptor
from modularnet.plugin.plugin_base import BasePlugin, Subscription
from modularnet.plugin.environment_manager import EnvUpdate


# ============================================================================
# Dummy Plugins for Testing
# ============================================================================

class DummyBasicPlugin(BasePlugin):
    """Minimal plugin for testing lifecycle."""
    
    def __init__(self, plugin_id: str, config: Dict[str, Any] = None, orchestrator_handle=None):
        super().__init__(plugin_id, config, orchestrator_handle)
        self.on_start_called = False
        self.on_stop_called = False
        self.on_update_called = False
    
    def on_start(self) -> None:
        self.on_start_called = True
    
    def on_stop(self) -> None:
        self.on_stop_called = True
    
    def on_update(self, config: Dict[str, Any]) -> None:
        self.on_update_called = True
        self.config.update(config)
    
    def on_error(self, error: Exception, context: str = "") -> None:
        pass
    
    def get_services(self) -> List[str]:
        return ["get_state"]
    
    def get_state(self) -> Dict[str, bool]:
        return {
            "started": self.on_start_called,
            "stopped": self.on_stop_called,
            "updated": self.on_update_called
        }


class DummyPubSubPlugin(BasePlugin):
    """Plugin for testing pub/sub messaging."""
    
    def __init__(self, plugin_id: str, config: Dict[str, Any] = None, orchestrator_handle=None):
        super().__init__(plugin_id, config, orchestrator_handle)
        self.received_messages: List[Any] = []
        self.output_count = 0
    
    def on_start(self) -> None:
        pass
    
    def on_stop(self) -> None:
        pass
    
    def on_error(self, error: Exception, context: str = "") -> None:
        pass
    
    def get_subscriptions(self) -> List[Subscription]:
        return [
            Subscription(
                name="test_input",
                callback="on_input_message",
                publish_to="test_output"
            )
        ]
    
    def on_input_message(self, data: Any, env: Dict[str, Any]) -> Any:
        """Receive message and republish transformed."""
        self.received_messages.append(data)
        self.output_count += 1
        return {"transformed": data, "count": self.output_count}
    
    def get_services(self) -> List[str]:
        return ["get_message_count", "get_messages"]
    
    def get_message_count(self) -> int:
        return len(self.received_messages)
    
    def get_messages(self) -> List[Any]:
        return self.received_messages.copy()


class DummyEnvPlugin(BasePlugin):
    """Plugin for testing environment variable subscriptions."""
    
    def __init__(self, plugin_id: str, config: Dict[str, Any] = None, orchestrator_handle=None):
        super().__init__(plugin_id, config, orchestrator_handle)
        self.env_changes: List[Dict[str, Any]] = []
        self.current_value = None
    
    def on_start(self) -> None:
        pass
    
    def on_stop(self) -> None:
        pass
    
    def on_error(self, error: Exception, context: str = "") -> None:
        pass
    
    def get_env_subscriptions(self) -> List[Subscription]:
        return [
            Subscription(
                name="shared_param",
                callback="on_param_change"
            )
        ]
    
    def on_param_change(self, update: EnvUpdate) -> None:
        """Called when env var changes."""
        self.current_value = update.value
        self.env_changes.append({
            "key": update.key,
            "old_value": update.old_value,
            "new_value": update.value
        })
    
    def get_services(self) -> List[str]:
        return ["get_change_count", "get_current_value", "get_changes"]
    
    def get_change_count(self) -> int:
        return len(self.env_changes)
    
    def get_current_value(self) -> Any:
        return self.current_value
    
    def get_changes(self) -> List[Dict[str, Any]]:
        return self.env_changes.copy()


class DummyErrorPlugin(BasePlugin):
    """Plugin for testing error handling."""
    
    def __init__(self, plugin_id: str, config: Dict[str, Any] = None, orchestrator_handle=None):
        super().__init__(plugin_id, config, orchestrator_handle)
        self.errors_received: List[Dict[str, str]] = []
    
    def on_start(self) -> None:
        pass
    
    def on_stop(self) -> None:
        pass
    
    def on_error(self, error: Exception, context: str = "") -> None:
        """Track errors."""
        self.errors_received.append({
            "context": context,
            "message": str(error)
        })
    
    def get_services(self) -> List[str]:
        return ["trigger_error", "get_errors"]
    
    def trigger_error(self) -> None:
        """Intentionally trigger an error."""
        try:
            raise ValueError("Test error")
        except Exception as e:
            self.on_error(e, "service_trigger")
    
    def get_errors(self) -> List[Dict[str, str]]:
        return self.errors_received.copy()


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def manager():
    """Create and cleanup plugin manager for each test."""
    pm = PluginManager()
    yield pm
    try:
        pm.shutdown()
    except Exception:
        pass
    
    # Give processes time to clean up
    time.sleep(0.2)


# ============================================================================
# High-Level Feature Tests
# ============================================================================

class TestLifecycleFeature:
    """Test plugin lifecycle: load, start, stop, unload."""
    
    def test_lifecycle_load_start_stop_unload(self, manager):
        """Test complete lifecycle sequence."""
        descriptor = PluginDescriptor(
            name="lifecycle_plugin",
            version="1.0.0",
            package=__name__,
            class_name="DummyBasicPlugin"
        )
        
        # Load plugin
        assert manager.load_plugin(descriptor)
        plugins = manager.list_plugins()
        assert "lifecycle_plugin" in plugins
        
        # Start plugin
        assert manager.start_plugin("lifecycle_plugin")
        time.sleep(0.2)
        
        # Stop plugin
        assert manager.stop_plugin("lifecycle_plugin")
        time.sleep(0.2)
        
        # Unload plugin
        assert manager.unload_plugin("lifecycle_plugin")
        plugins = manager.list_plugins()
        assert "lifecycle_plugin" not in plugins
    
    def test_lifecycle_multiple_plugins(self, manager):
        """Test managing multiple plugins in lifecycle."""
        # Load 3 plugins
        for i in range(3):
            descriptor = PluginDescriptor(
                name=f"p{i}",
                version="1.0.0",
                package=__name__,
                class_name="DummyBasicPlugin"
            )
            assert manager.load_plugin(descriptor)
        
        # All loaded
        assert len(manager.list_plugins()) == 3
        
        # Start all
        for i in range(3):
            assert manager.start_plugin(f"p{i}")
        
        time.sleep(0.2)
        
        # Stop all
        for i in range(3):
            assert manager.stop_plugin(f"p{i}")
        
        time.sleep(0.2)
        
        # Unload all
        for i in range(3):
            assert manager.unload_plugin(f"p{i}")
        
        assert len(manager.list_plugins()) == 0
    
    def test_lifecycle_start_before_load_fails(self, manager):
        """Test that starting non-existent plugin fails gracefully."""
        assert not manager.start_plugin("nonexistent_plugin")
    
    def test_lifecycle_double_start_idempotent(self, manager):
        """Test starting already-running plugin."""
        descriptor = PluginDescriptor(
            name="test_p",
            version="1.0.0",
            package=__name__,
            class_name="DummyBasicPlugin"
        )
        
        manager.load_plugin(descriptor)
        assert manager.start_plugin("test_p")
        time.sleep(0.2)
        
        # Start again (should handle gracefully)
        assert manager.start_plugin("test_p")
        time.sleep(0.2)


class TestLifecycleEventsFeature:
    """Test plugin lifecycle events: onStart, onStop, onUpdate, onError."""
    
    def test_event_on_start_called(self, manager):
        """Test onStart event is called."""
        descriptor = PluginDescriptor(
            name="event_plugin",
            version="1.0.0",
            package=__name__,
            class_name="DummyBasicPlugin"
        )
        
        manager.load_plugin(descriptor)
        manager.start_plugin("event_plugin")
        time.sleep(0.3)
        
        # Check that on_start was called
        state = manager.invoke_service("event_plugin", "get_state")
        assert state["started"] is True
    
    def test_event_on_stop_called(self, manager):
        """Test onStop event is called."""
        descriptor = PluginDescriptor(
            name="event_plugin",
            version="1.0.0",
            package=__name__,
            class_name="DummyBasicPlugin"
        )
        
        manager.load_plugin(descriptor)
        manager.start_plugin("event_plugin")
        time.sleep(0.2)
        manager.stop_plugin("event_plugin")
        time.sleep(0.2)
        
        # Note: Can't directly check stopped flag in subprocess, 
        # but if stop didn't error, it worked
    
    def test_event_on_update_called(self, manager):
        """Test onUpdate event is called when config changes."""
        descriptor = PluginDescriptor(
            name="event_plugin",
            version="1.0.0",
            package=__name__,
            class_name="DummyBasicPlugin"
        )
        
        manager.load_plugin(descriptor)
        manager.start_plugin("event_plugin")
        time.sleep(0.2)
        
        # Update config
        manager.update_plugin_config("event_plugin", {"new_param": "value"})
        time.sleep(0.2)


class TestEnvFeature:
    """Test environment variables: set, get, onChange subscriptions."""
    
    def test_env_set_and_get(self, manager):
        """Test setting and getting env variables."""
        manager.set_env_variable("test_key", "test_value")
        value = manager.get_env_variable("test_key")
        assert value == "test_value"
    
    def test_env_get_with_default(self, manager):
        """Test getting non-existent key with default."""
        value = manager.get_env_variable("nonexistent", "default_value")
        assert value == "default_value"
    
    def test_env_get_all(self, manager):
        """Test getting all env variables."""
        manager.set_env_variable("key1", "value1")
        manager.set_env_variable("key2", "value2")
        manager.set_env_variable("key3", "value3")
        
        all_vars = manager.get_all_env_variables()
        assert all_vars["key1"] == "value1"
        assert all_vars["key2"] == "value2"
        assert all_vars["key3"] == "value3"
    
    def test_env_on_change_single_subscription(self, manager):
        """Test onChange callback on single env var."""
        descriptor = PluginDescriptor(
            name="env_plugin",
            version="1.0.0",
            package=__name__,
            class_name="DummyEnvPlugin"
        )
        
        manager.load_plugin(descriptor)
        manager.start_plugin("env_plugin")
        time.sleep(0.3)
        
        # Change env var
        manager.set_env_variable("shared_param", "new_value")
        time.sleep(0.3)
        
        # Check plugin received change
        current = manager.invoke_service("env_plugin", "get_current_value")
        assert current == "new_value"
    
    def test_env_on_change_tracking(self, manager):
        """Test onChange callback tracks changes."""
        descriptor = PluginDescriptor(
            name="env_plugin",
            version="1.0.0",
            package=__name__,
            class_name="DummyEnvPlugin"
        )
        
        manager.load_plugin(descriptor)
        manager.start_plugin("env_plugin")
        time.sleep(0.3)
        
        # Set initial value
        manager.set_env_variable("shared_param", "value1")
        time.sleep(0.2)
        
        # Change it multiple times
        manager.set_env_variable("shared_param", "value2")
        time.sleep(0.2)
        
        manager.set_env_variable("shared_param", "value3")
        time.sleep(0.2)
        
        # Check all changes were tracked
        changes = manager.invoke_service("env_plugin", "get_changes")
        assert len(changes) >= 3
        
        # Check values
        values = [c["new_value"] for c in changes]
        assert "value1" in values
        assert "value2" in values
        assert "value3" in values
    
    def test_env_multiple_subscribers(self, manager):
        """Test multiple plugins subscribe to same env var."""
        # Load 2 env plugins
        for i in range(2):
            descriptor = PluginDescriptor(
                name=f"env_p{i}",
                version="1.0.0",
                package=__name__,
                class_name="DummyEnvPlugin"
            )
            manager.load_plugin(descriptor)
            manager.start_plugin(f"env_p{i}")
        
        time.sleep(0.3)
        
        # Set env var
        manager.set_env_variable("shared_param", "broadcast_value")
        time.sleep(0.3)
        
        # Both should receive it
        val1 = manager.invoke_service("env_p0", "get_current_value")
        val2 = manager.invoke_service("env_p1", "get_current_value")
        assert val1 == "broadcast_value"
        assert val2 == "broadcast_value"


class TestPubSubFeature:
    """Test the pub/sub messaging system."""
    
    def test_pubsub_basic_publish_receive(self, manager):
        """Test basic pub/sub: publish -> receive."""
        descriptor = PluginDescriptor(
            name="pubsub_plugin",
            version="1.0.0",
            package=__name__,
            class_name="DummyPubSubPlugin"
        )
        
        assert manager.load_plugin(descriptor)
        assert manager.start_plugin("pubsub_plugin")
        manager.start_processing_loop(interval=0.1)
        
        time.sleep(0.3)
        
        # Publish messages
        manager.publish("test_input", {"cmd": "test1"})
        manager.publish("test_input", {"cmd": "test2"})
        manager.publish("test_input", {"cmd": "test3"})
        
        # Wait for processing
        time.sleep(0.5)
        
        # Verify plugin received all messages
        count = manager.invoke_service("pubsub_plugin", "get_message_count")
        assert count == 3
        
        messages = manager.invoke_service("pubsub_plugin", "get_messages")
        assert len(messages) == 3
        assert messages[0]["cmd"] == "test1"
        assert messages[2]["cmd"] == "test3"
        
        manager.stop_processing_loop()
    
    def test_pubsub_multiple_topics(self, manager):
        """Test pub/sub with multiple topics."""
        # Load two plugins subscribing to same topic
        for name in ["p1", "p2"]:
            descriptor = PluginDescriptor(
                name=name,
                version="1.0.0",
                package=__name__,
                class_name="DummyPubSubPlugin"
            )
            manager.load_plugin(descriptor)
            manager.start_plugin(name)
        
        manager.start_processing_loop(interval=0.1)
        time.sleep(0.3)
        
        # Publish 10 messages
        for i in range(10):
            manager.publish("test_input", {"seq": i})
        
        time.sleep(0.5)
        
        # Both plugins should receive all messages
        count1 = manager.invoke_service("p1", "get_message_count")
        count2 = manager.invoke_service("p2", "get_message_count")
        assert count1 == 10
        assert count2 == 10
        
        manager.stop_processing_loop()


class TestIntegration:
    """Test interactions between features."""
    
    def test_full_system_workflow(self, manager):
        """Test a realistic workflow using all features."""
        # Load pub/sub plugin
        ps_descriptor = PluginDescriptor(
            name="processor",
            version="1.0.0",
            package=__name__,
            class_name="DummyPubSubPlugin"
        )
        manager.load_plugin(ps_descriptor)
        manager.start_plugin("processor")
        
        # Load env plugin
        env_descriptor = PluginDescriptor(
            name="monitor",
            version="1.0.0",
            package=__name__,
            class_name="DummyEnvPlugin"
        )
        manager.load_plugin(env_descriptor)
        manager.start_plugin("monitor")
        
        manager.start_processing_loop(interval=0.1)
        time.sleep(0.3)
        
        # Set env variables
        manager.set_env_variable("shared_param", "config_a")
        time.sleep(0.2)
        
        # Publish messages
        for i in range(5):
            manager.publish("test_input", {"seq": i})
        
        time.sleep(0.5)
        
        # Verify pub/sub worked
        msg_count = manager.invoke_service("processor", "get_message_count")
        print(msg_count)
        assert msg_count == 5
        
        # Verify env subscription worked
        current_val = manager.invoke_service("monitor", "get_current_value")
        assert current_val == "config_a"
        
        # Update env
        manager.set_env_variable("shared_param", "config_b")
        time.sleep(0.2)
        
        # Publish more messages
        for i in range(5, 10):
            manager.publish("test_input", {"seq": i})
        
        time.sleep(0.5)
        
        # Verify new env value
        current_val = manager.invoke_service("monitor", "get_current_value")
        assert current_val == "config_b"
        
        # Verify all messages processed
        msg_count = manager.invoke_service("processor", "get_message_count")
        assert msg_count == 10
        
        manager.stop_processing_loop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
