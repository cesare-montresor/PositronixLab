"""
Plugin Handler: Ray actor that manages individual plugins.

Each plugin runs in isolation. The handler manages:
1. Plugin lifecycle (start, stop, update)
2. Pub/sub message dispatch
3. Environment variable subscriptions
4. Service method calls
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path
import importlib
import sys
import traceback
import ray

from .plugin_base import BasePlugin, Subscription
from .environment_manager import EnvironmentManager, EnvUpdate

logger = logging.getLogger(__name__)


@ray.remote
class PluginHandler:
    """
    Manages a single plugin instance in isolation.
    
    Handles plugin lifecycle, message dispatch, environment variables, and services.
    """
    
    def __init__(self, plugin_id: str, orchestrator_handle: Any):
        """
        Initialize handler.
        
        Args:
            plugin_id: Unique plugin identifier
            orchestrator_handle: Ray handle to orchestrator
        """
        self.plugin_id = plugin_id
        self.orchestrator = orchestrator_handle
        self.plugin: Optional[BasePlugin] = None
        self.status = "INITIALIZED"
        self._subscriptions: List[Dict[str, Any]] = []
        self._env_subscriptions: List[Dict[str, Any]] = []
        self._local_env = {}  # Local cache of env vars for O(1) access
        self.error_log: List[str] = []
    
    def load_plugin(self, descriptor_dict: Dict[str, Any], config: Dict[str, Any]) -> bool:
        """
        Load plugin from descriptor.
        
        Args:
            descriptor_dict: Plugin descriptor as dictionary
            config: Configuration options for the plugin
            
        Returns:
            True if successful, False otherwise
        """
        try:
            package = descriptor_dict["package"]
            class_name = descriptor_dict["class_name"]
            
            # Import module dynamically
            module = importlib.import_module(package)
            plugin_class = getattr(module, class_name)
            
            # Instantiate plugin with orchestrator handle for communication
            merged_config = {**descriptor_dict.get("options", {}), **config}
            self.plugin = plugin_class(self.plugin_id, merged_config, orchestrator_handle=self.orchestrator)
            
            # Set up subscriptions
            for sub_desc in self.plugin.get_subscriptions():
                name = sub_desc.name
                callback_name = sub_desc.callback
                publish_to = sub_desc.publish_to
                
                self._subscriptions.append({
                    "name": name,
                    "callback": callback_name,
                    "publish_to": publish_to,
                    "queue": None  # Will be set by orchestrator
                })
            
            # Set up env subscriptions
            for sub_desc in self.plugin.get_env_subscriptions():
                name = sub_desc.name
                callback_name = sub_desc.callback
                
                self._env_subscriptions.append({
                    "name": name,
                    "callback": callback_name
                })
            
            self.status = "LOADED"
            logger.info(f"Plugin {self.plugin_id} loaded successfully")
            return True
            
        except Exception as e:
            error_msg = f"Failed to load plugin {self.plugin_id}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.error_log.append(error_msg)
            self.status = "ERROR"
            return False
    
    def start(self) -> bool:
        """
        Start the plugin.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.plugin:
            error_msg = f"Plugin {self.plugin_id} not loaded"
            logger.error(error_msg)
            self.error_log.append(error_msg)
            return False
        
        try:
            self.plugin.on_start()
            self.plugin._is_running = True
            self.status = "RUNNING"
            
            # Register subscriptions with orchestrator
            for sub in self._subscriptions:
                q_ref = ray.get(self.orchestrator.subscribe.remote(sub["name"]))
                sub["queue"] = q_ref
            
            # Register env subscriptions with orchestrator
            for env_sub in self._env_subscriptions:
                self.orchestrator.subscribe_env.remote(env_sub["name"], self.plugin_id)
            
            logger.info(f"Plugin {self.plugin_id} started")
            return True
            
        except Exception as e:
            error_msg = f"Plugin {self.plugin_id} failed to start: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.error_log.append(error_msg)
            self.plugin._is_running = False
            self.status = "ERROR"
            if self.plugin:
                self.plugin.on_error(Exception(error_msg), "on_start")
            return False
    
    def stop(self) -> bool:
        """
        Stop the plugin.
        
        Returns:
            True if successful
        """
        if not self.plugin:
            return True
        
        try:
            self.plugin.on_stop()
            self.plugin._is_running = False
            self.status = "STOPPED"
            
            # Unregister subscriptions
            self._subscriptions.clear()
            self._env_subscriptions.clear()
            
            logger.info(f"Plugin {self.plugin_id} stopped")
            return True
            
        except Exception as e:
            error_msg = f"Plugin {self.plugin_id} error during stop: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.error_log.append(error_msg)
            self.status = "ERROR"
            if self.plugin:
                self.plugin.on_error(Exception(error_msg), "on_stop")
            return False
    
    def update_config(self, config: Dict[str, Any]) -> bool:
        """
        Update plugin configuration.
        
        Args:
            config: New configuration
            
        Returns:
            True if successful
        """
        if not self.plugin:
            return False
        
        try:
            self.plugin.on_update(config)
            self.status = "RUNNING"
            logger.info(f"Plugin {self.plugin_id} configuration updated")
            return True
            
        except Exception as e:
            error_msg = f"Plugin {self.plugin_id} error during config update: {str(e)}"
            logger.error(error_msg)
            self.error_log.append(error_msg)
            self.status = "ERROR"
            if self.plugin:
                self.plugin.on_error(Exception(error_msg), "on_update")
            return False
    
    def receive_env_update(self, update: EnvUpdate) -> None:
        """
        Receive environment variable update.
        
        Args:
            update: EnvUpdate object with key and value
        """
        self._local_env[update.key] = update.value
        
        # Call env subscriptions
        for env_sub in self._env_subscriptions:
            if env_sub["name"] == update.key:
                callback_name = env_sub["callback"]
                if hasattr(self.plugin, callback_name):
                    try:
                        callback = getattr(self.plugin, callback_name)
                        callback(update)
                    except Exception as e:
                        logger.error(f"Error in env callback {callback_name}: {e}")
                        if self.plugin:
                            self.plugin.on_error(e, f"env_callback_{callback_name}")
    
    def execute_service(self, method_name: str, *args, **kwargs) -> Any:
        """
        Execute a service method on the plugin.
        
        Args:
            method_name: Name of the method to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Method return value
        """
        if not self.plugin:
            raise RuntimeError(f"Plugin {self.plugin_id} not loaded")
        
        if method_name not in self.plugin.get_services():
            raise ValueError(f"Service method '{method_name}' not available")
        
        try:
            method = getattr(self.plugin, method_name)
            return method(*args, **kwargs)
        except Exception as e:
            logger.error(f"Service method {method_name} error: {e}")
            if self.plugin:
                self.plugin.on_error(e, f"service_{method_name}")
            raise
    
    def process_messages(self, max_messages: int = 10) -> int:
        """
        Process pending messages from all subscribed topics.
        
        Args:
            max_messages: Maximum messages to process per call
            
        Returns:
            Number of messages processed
        """
        if self.status != "RUNNING" or not self.plugin:
            return 0
        
        messages_processed = 0
        
        for sub in self._subscriptions:
            q = sub.get("queue")
            if not q:
                continue
            
            callback_name = sub["callback"]
            publish_to = sub["publish_to"]
            
            while not q.empty() and messages_processed < max_messages:
                try:
                    msg = q.get(block=False)
                    
                    # Resolve ObjectRef if present
                    if isinstance(msg, dict) and "data_ref" in msg:
                        data = ray.get(msg["data_ref"])
                    else:
                        data = msg
                    
                    # Call plugin's callback
                    if hasattr(self.plugin, callback_name):
                        callback = getattr(self.plugin, callback_name)
                        result = callback(data, self._local_env)
                        
                        # Republish if configured
                        if result is not None and publish_to:
                            self.orchestrator.publish.remote(publish_to, result)
                    
                    messages_processed += 1
                    
                except Exception as e:
                    logger.error(f"Error processing message in {callback_name}: {e}")
                    if self.plugin:
                        self.plugin.on_error(e, f"callback_{callback_name}")
        
        return messages_processed
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get plugin status information.
        
        Returns:
            Status dictionary
        """
        return {
            "plugin_id": self.plugin_id,
            "status": self.status,
            "is_running": self.plugin._is_running if self.plugin else False,
            "subscriptions": len(self._subscriptions),
            "env_subscriptions": len(self._env_subscriptions),
            "errors": len(self.error_log)
        }
    
    def get_errors(self) -> List[str]:
        """Get recent error messages."""
        return self.error_log.copy()
    
    def clear_errors(self) -> None:
        """Clear error log."""
        self.error_log.clear()
