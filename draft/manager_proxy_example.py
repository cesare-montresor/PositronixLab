"""
Example: Using Manager Proxy in Plugins

This example demonstrates how plugins can directly access the manager
during runtime, not just during initialization.
"""

from modularnet.plugin import BasePlugin, Subscription, SubscriptionType, PluginDescriptor
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ProducerPlugin(BasePlugin):
    """Plugin that publishes data and sets environment variables."""
    
    @classmethod
    def Descriptor(cls, options: Dict[str, Any] = None) -> PluginDescriptor:
        return super().Descriptor(
            version="1.0.0",
            author="Example",
            description="Plugin that produces data",
            options=options
        )
    
    def on_start(self) -> None:
        """Initialize and start publishing."""
        logger.info(f"[{self.plugin_id}] Starting producer")
        
        # Set initial state in environment
        self._worker.env_set("producer_status", "initialized")
        
        # Simulate some work and publish results
        for i in range(3):
            # Publish to topic for subscribers
            self._worker.publish("data_topic", {"value": i, "source": self.plugin_id})
            
            # Update environment for other plugins to read
            self._worker.env_set("latest_value", i)
            
            logger.info(f"[{self.plugin_id}] Published value: {i}")
    
    def on_stop(self) -> None:
        """Cleanup."""
        self._worker.env_set("producer_status", "stopped")
        logger.info(f"[{self.plugin_id}] Producer stopped")
    
    def get_subscriptions(self) -> List[Subscription]:
        """This plugin doesn't subscribe to anything."""
        return []
    
    def get_services(self) -> List[str]:
        """Provide a service method."""
        return ["get_last_value"]
    
    def get_last_value(self) -> int:
        """Service method that can be called by other plugins."""
        return self._worker.env_get("latest_value", default=-1)


class ConsumerPlugin(BasePlugin):
    """Plugin that subscribes to data and consumes it."""
    
    @classmethod
    def Descriptor(cls, options: Dict[str, Any] = None) -> PluginDescriptor:
        return super().Descriptor(
            version="1.0.0",
            author="Example",
            description="Plugin that consumes data",
            options=options
        )
    
    def __init__(self, plugin_id: str, config: Optional[Dict[str, Any]] = None, manager=None):
        super().__init__(plugin_id, config, manager)
        self.data_received = []
    
    def on_start(self) -> None:
        """Start consuming."""
        logger.info(f"[{self.plugin_id}] Starting consumer")
        self._worker.env_set("consumer_status", "running")
    
    def on_stop(self) -> None:
        """Cleanup."""
        logger.info(f"[{self.plugin_id}] Consumer stopped")
    
    def on_data_received(self, data: Any, env_state: Dict[str, Any]) -> None:
        """
        Callback when data is published to subscribed topic.
        
        This is called by the plugin system when data arrives.
        """
        logger.info(f"[{self.plugin_id}] Received data: {data}")
        self.data_received.append(data)
    
    def on_latest_value_changed(self, env_update) -> None:
        """
        Callback when subscribed environment variable changes.
        
        This is called by the plugin system when env var is updated.
        """
        logger.info(
            f"[{self.plugin_id}] Latest value changed: "
            f"{env_update.old_value} → {env_update.value}"
        )
        
        # Now directly invoke a service on producer via manager proxy
        try:
            value = self._worker.invoke_service(
                "producer",  # target plugin_id
                "get_last_value"  # method name
            )
            logger.info(f"[{self.plugin_id}] Producer reports last value: {value}")
        except Exception as e:
            logger.error(f"Failed to invoke producer service: {e}")
    
    def get_subscriptions(self) -> List[Subscription]:
        """Subscribe to topic and environment variable."""
        return [
            Subscription.topic("data_topic", "on_data_received"),
            Subscription.env_var("latest_value", "on_latest_value_changed"),
        ]
    
    def get_services(self) -> List[str]:
        """This plugin has no public services."""
        return []


class CoordinatorPlugin(BasePlugin):
    """Plugin that coordinates other plugins using manager proxy."""
    
    @classmethod
    def Descriptor(cls, options: Dict[str, Any] = None) -> PluginDescriptor:
        return super().Descriptor(
            version="1.0.0",
            author="Example",
            description="Coordinator plugin",
            options=options
        )
    
    def on_start(self) -> None:
        """Coordinate plugin execution."""
        logger.info(f"[{self.plugin_id}] Coordinator starting")
        
        # Check status of other plugins via manager proxy
        try:
            producer_status = self._worker.get_plugin_status("producer")
            consumer_status = self._worker.get_plugin_status("consumer")
            
            if producer_status:
                logger.info(f"Producer alive: {producer_status['process_alive']}")
            if consumer_status:
                logger.info(f"Consumer alive: {consumer_status['process_alive']}")
        except Exception as e:
            logger.error(f"Failed to get plugin status: {e}")
        
        # Query all environment variables
        all_env = self._worker.get_all_env()
        logger.info(f"Current environment: {all_env}")
        
        # Set coordination state
        self._worker.env_set("coordinator_active", True)
    
    def on_stop(self) -> None:
        """Cleanup."""
        logger.info(f"[{self.plugin_id}] Coordinator stopped")
    
    def get_subscriptions(self) -> List[Subscription]:
        """Subscribe to overall status updates."""
        return []
    
    def get_services(self) -> List[str]:
        """Provide coordination methods."""
        return ["get_coordination_status"]
    
    def get_coordination_status(self) -> Dict[str, Any]:
        """Service method to get current coordination state."""
        try:
            return {
                "plugins": self._worker.list_plugins(),
                "env": self._worker.get_all_env(),
                "coordinator": self.plugin_id
            }
        except Exception as e:
            logger.error(f"Failed to get coordination status: {e}")
            return {}


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    from modularnet.plugin import PluginManager
    
    logging.basicConfig(level=logging.INFO)
    
    # Create manager
    manager = PluginManager()
    
    # Load plugins
    manager.load_plugin(ProducerPlugin.Descriptor())
    manager.load_plugin(ConsumerPlugin.Descriptor())
    manager.load_plugin(CoordinatorPlugin.Descriptor())
    
    # Start plugins
    manager.start_plugin("producer")
    manager.start_plugin("consumer")
    manager.start_plugin("coordinator")
    
    # Let them run
    import time
    time.sleep(2)
    
    # Stop plugins
    manager.stop_plugin("producer")
    manager.stop_plugin("consumer")
    manager.stop_plugin("coordinator")
    
    # Unload
    manager.unload_plugin("producer")
    manager.unload_plugin("consumer")
    manager.unload_plugin("coordinator")
