"""
Plugin Worker: Subprocess handler for plugin execution.

PluginWorker.run() is the entry point for each plugin subprocess.
It creates plugin instance, handles incoming requests, and routes to handlers.

Command dispatch:
  - All incoming requests routed via @command_handler decorator
  - Services invoked via reflection (self.plugin.method_name(*args, **kwargs))
  - No manual if/elif chains
"""

from __future__ import annotations

import logging
import inspect
import threading
from typing import Any, Dict, Optional, Callable
from functools import wraps

from modularnet.plugin.plugin_messages import Request, Response, Channel, DataExchange, EnvUpdate, SubscriptionType, WorkerStatus
from modularnet.plugin.plugin_base import BasePlugin
from modularnet.plugin.plugin_descriptor import PluginDescriptor

logger = logging.getLogger(__name__)


# ============================================================================
# COMMAND HANDLER DECORATOR
# ============================================================================

_command_handlers: Dict[str, Callable] = {}


def command_handler(command_name: str) -> Callable:
    """
    Decorator to register command handlers.
    
    Usage:
        @command_handler("start")
        def handle_start(self, req):
            ...
            return result  # automatic Response wrapping
    """
    def decorator(func: Callable) -> Callable:
        _command_handlers[command_name] = func
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ============================================================================
# PLUGIN WORKER
# ============================================================================

class PluginWorker:
    """
    Subprocess worker for running a plugin.
    
    PluginWorker.run() is called in a new subprocess and:
    1. Instantiates the plugin
    2. Runs an event loop waiting for requests
    3. Routes requests to command handlers
    4. Sends responses back
    """
    
    @staticmethod
    def run(
        plugin_id: str,
        descriptor_dict: Dict[str, Any],
        config: Dict[str, Any],
        cmd_queue: Any,
        resp_queue: Any
    ) -> None:
        """
        Entry point for plugin subprocess.
        
        Args:
            plugin_id: Plugin instance identifier
            descriptor_dict: Serialized plugin descriptor
            config: Plugin configuration
            cmd_queue: Queue for incoming requests
            resp_queue: Queue for outgoing responses
        """
        worker = PluginWorker(plugin_id, descriptor_dict, config, cmd_queue, resp_queue)
        worker.run_loop()
    
    def __init__(
        self,
        plugin_id: str,
        descriptor_dict: Dict[str, Any],
        config: Dict[str, Any],
        cmd_queue: Any,
        resp_queue: Any
    ):
        """Initialize worker."""
        self.plugin_id = plugin_id
        self.descriptor = PluginDescriptor.from_dict(descriptor_dict)
        self.config = config
        self.channel = Channel(resp_queue, cmd_queue)  # Note: queues swapped for worker
        
        # Instantiate the plugin
        self.plugin: Optional[BasePlugin] = None
        self._init_plugin()
        
        # Topic subscriptions: topic_name -> queue
        self.topic_subscriptions: Dict[str, Any] = {}
        
        # Error log
        self.error_log: list = []
    
    def _init_plugin(self) -> None:
        """Instantiate plugin from descriptor."""
        try:
            # Import plugin class
            import importlib
            module = importlib.import_module(self.descriptor.package)
            plugin_class = getattr(module, self.descriptor.class_name)
            
            # Instantiate
            self.plugin = plugin_class(self.plugin_id, self.config)
            logger.info(f"Plugin {self.plugin_id} instantiated")
        except Exception as e:
            logger.error(f"Failed to init plugin {self.plugin_id}: {e}")
            raise
    
    def run_loop(self) -> None:
        """Main event loop: wait for requests and topic messages."""
        logger.info(f"Plugin {self.plugin_id} worker started")
        
        # Start topic listener thread
        topic_thread = threading.Thread(target=self._topic_listener_loop, daemon=True)
        topic_thread.start()
        
        try:
            while True:
                req = self.channel.receive(timeout=1.0)
                if req is None:
                    continue
                
                self._handle_request(req)
        except KeyboardInterrupt:
            logger.info(f"Plugin {self.plugin_id} interrupted")
        except Exception as e:
            logger.error(f"Plugin {self.plugin_id} error: {e}", exc_info=True)
        finally:
            self._cleanup()
    
    def _topic_listener_loop(self) -> None:
        """Background thread: listen for messages on topic subscriptions."""
        while True:
            try:
                # Poll each topic subscription
                for topic, queue in list(self.topic_subscriptions.items()):
                    try:
                        # Non-blocking get with tiny timeout
                        data = queue.get(timeout=0.01)
                        self._deliver_topic_message(topic, data)
                    except Exception:
                        # Timeout or empty queue, continue
                        continue
                
                # Small sleep to avoid busy loop
                threading.Event().wait(0.01)
            except Exception as e:
                logger.error(f"Topic listener error: {e}")
    
    def _deliver_topic_message(self, topic: str, data: Any) -> None:
        """Deliver a topic message to the appropriate callback."""
        try:
            # Find subscription for this topic
            subs = self.plugin.get_subscriptions()
            for sub in subs:
                if sub.sub_type == SubscriptionType.TOPIC and sub.name == topic:
                    # Call the callback
                    callback_method = getattr(self.plugin, sub.callback)
                    # Get current env state
                    env_state = {}  # Could pass actual env if needed
                    result = callback_method(data, env_state)
                    
                    # Republish if configured
                    if sub.republish_to and result is not None:
                        # Store for potential republish (manager will handle via publish())
                        pass
                    
                    break
        except Exception as e:
            logger.error(f"Error delivering topic message: {e}", exc_info=True)
            self.error_log.append(str(e))
    
    def _handle_request(self, req: Request) -> None:
        """Dispatch request to appropriate handler."""
        try:
            command = req.command
            
            # Try registered handler first
            if command in _command_handlers:
                handler = _command_handlers[command]
                result = handler(self, req)
                self.channel.reply(req.cmd_id, result=result)
            
            # Fall back to service invocation
            elif hasattr(self.plugin, command):
                result = self._invoke_service(req)
                self.channel.reply(req.cmd_id, result=result)
            
            else:
                raise ValueError(f"Unknown command: {command}")
        
        except Exception as e:
            logger.error(f"Request handler error: {e}", exc_info=True)
            self.error_log.append(str(e))
            self.channel.reply(req.cmd_id, error=str(e))
    
    def _invoke_service(self, req: Request) -> Any:
        """Invoke service method on plugin via reflection."""
        command = req.command
        args = req.args
        kwargs = req.kwargs
        
        method = getattr(self.plugin, command)
        return method(*args, **kwargs)
    
    def _cleanup(self) -> None:
        """Clean up resources."""
        try:
            if self.plugin:
                self.plugin.on_stop()
                logger.info(f"Plugin {self.plugin_id} stopped")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    # ========================================================================
    # COMMAND HANDLERS
    # ========================================================================
    
    @command_handler("start")
    def handle_start(self, req: Request) -> bool:
        """Start plugin."""
        try:
            self.plugin.on_start()
            return True
        except Exception as e:
            logger.error(f"Failed to start plugin: {e}")
            raise
    
    @command_handler("stop")
    def handle_stop(self, req: Request) -> bool:
        """Stop plugin."""
        try:
            self.plugin.on_stop()
            return True
        except Exception as e:
            logger.error(f"Failed to stop plugin: {e}")
            raise
    
    @command_handler("update_config")
    def handle_update_config(self, req: Request) -> bool:
        """Update plugin config."""
        try:
            config = req.args[0] if req.args else {}
            self.plugin.on_update(config)
            self.config.update(config)
            return True
        except Exception as e:
            logger.error(f"Failed to update config: {e}")
            raise
    
    @command_handler("get_data_exchanges")
    def handle_get_data_exchanges(self, req: Request) -> Dict[str, Any]:
        """
        Get all data exchange subscriptions (unified topic + env).
        
        Returns:
            {
                "topics": [topic_name, ...],
                "env_vars": [env_key, ...]
            }
        """
        subs = self.plugin.get_subscriptions()
        topics = [s.name for s in subs if s.sub_type == SubscriptionType.TOPIC]
        env_vars = [s.name for s in subs if s.sub_type == SubscriptionType.ENV_VAR]
        
        return {
            "topics": topics,
            "env_vars": env_vars,
            "subscriptions": subs  # Return full subscription objects for manager
        }
    
    @command_handler("register_topic_subscription")
    def handle_register_topic_subscription(self, req: Request) -> bool:
        """Register a topic subscription queue."""
        topic = req.args[0]
        queue = req.args[1]
        self.topic_subscriptions[topic] = queue
        return True
    
    @command_handler("receive_env_update")
    def handle_receive_env_update(self, req: Request) -> None:
        """Receive environment variable change notification."""
        update: EnvUpdate = req.args[0]
        
        # Find env subscription for this key
        subs = self.plugin.get_subscriptions()
        for sub in subs:
            if sub.sub_type == SubscriptionType.ENV_VAR and sub.name == update.key:
                callback_method = getattr(self.plugin, sub.callback)
                callback_method(update)
    
    @command_handler("get_status")
    def handle_get_status(self, req: Request) -> Dict[str, Any]:
        """Get plugin status."""
        subs = self.plugin.get_subscriptions()
        return {
            "plugin_id": self.plugin_id,
            "status": "running",
            "is_running": True,
            "subscriptions": len(subs),
            "errors": len(self.error_log)
        }
    
    @command_handler("get_errors")
    def handle_get_errors(self, req: Request) -> list:
        """Get error log."""
        return self.error_log.copy()
    
    @command_handler("clear_errors")
    def handle_clear_errors(self, req: Request) -> bool:
        """Clear error log."""
        self.error_log.clear()
        return True


# ============================================================================
# TOPIC SUBSCRIPTION MESSAGE DELIVERY
# ============================================================================

def deliver_topic_message(queue: Any, data: Any) -> bool:
    """
    Try to deliver a topic message.
    
    Args:
        queue: Subscriber's message queue
        data: Message data
    
    Returns:
        True if delivered, False if queue full/error
    """
    try:
        queue.put_nowait(data)
        return True
    except Exception:
        return False
