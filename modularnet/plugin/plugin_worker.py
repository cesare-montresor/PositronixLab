"""
Plugin Worker (torch.multiprocessing backend): Manages individual plugins in separate processes.

Each plugin runs in isolation in its own process. The worker manages:
1. Plugin lifecycle (start, stop, update)
2. Pub/sub message dispatch
3. Environment variable subscriptions
4. Service method calls
5. Command processing
"""

import logging
import traceback
import importlib
import uuid
import time
from typing import Any, Dict, List, Optional

import torch.multiprocessing as mp

from modularnet.plugin.plugin_base import BasePlugin, PluginState
from modularnet.plugin.plugin_messages import Responses, Channel, WorkerStatus
from modularnet.plugin.environment_manager import EnvUpdate

logger = logging.getLogger(__name__)


class PluginWorker:
    """
    Manages a single plugin instance in isolation (runs in subprocess).
    
    Handles plugin lifecycle, message dispatch, environment variables, and services.
    """
    
    def __init__(
        self,
        plugin_id: str,
        max_messages_per_cycle: int = 5,
        cmd_receive_timeout: float = 0.1,
        process_sleep_interval: float = 0.01
    ):
        """
        Initialize worker.
        
        Args:
            plugin_id: Unique plugin identifier
            max_messages_per_cycle: Max messages to process per poll cycle
            cmd_receive_timeout: Timeout for receiving commands (seconds)
            process_sleep_interval: Sleep time when no command (seconds)
        """
        self.plugin_id = plugin_id
        self.plugin: Optional[BasePlugin] = None
        self.status = PluginState.INITIALIZED
        self._subscriptions: List[Dict[str, Any]] = []
        self._env_subscriptions: List[Dict[str, Any]] = []
        self._local_env: Dict[str, Any] = {}  # Local cache of env vars for O(1) access
        self.error_log: List[str] = []
        
        # Configuration
        self.max_messages_per_cycle = max_messages_per_cycle
        self.cmd_receive_timeout = cmd_receive_timeout
        self.process_sleep_interval = process_sleep_interval
    
    def _safe_arg(self, args: tuple, index: int, default: Any = None) -> Any:
        """Safely extract argument from args tuple."""
        return args[index] if len(args) > index else default
    
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
            
            # Instantiate plugin
            merged_config = {**descriptor_dict.get("options", {}), **config}
            self.plugin = plugin_class(self.plugin_id, merged_config)
            
            # Set up subscriptions
            for sub_desc in self.plugin.get_subscriptions():
                name = sub_desc.name
                callback_name = sub_desc.callback
                publish_to = sub_desc.publish_to
                
                self._subscriptions.append({
                    "name": name,
                    "callback": callback_name,
                    "publish_to": publish_to,
                    "queue": None  # Will be set on start
                })
            
            # Set up env subscriptions
            for sub_desc in self.plugin.get_env_subscriptions():
                name = sub_desc.name
                callback_name = sub_desc.callback
                
                self._env_subscriptions.append({
                    "name": name,
                    "callback": callback_name
                })
            
            self.status = PluginState.LOADED
            logger.info(f"Plugin {self.plugin_id} loaded successfully")
            return True
            
        except Exception as e:
            error_msg = f"Failed to load plugin {self.plugin_id}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.error_log.append(error_msg)
            self.status = PluginState.ERROR
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
            self.status = PluginState.RUNNING
            
            logger.info(f"Plugin {self.plugin_id} started")
            return True
            
        except Exception as e:
            error_msg = f"Plugin {self.plugin_id} failed to start: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.error_log.append(error_msg)
            self.plugin._is_running = False
            self.status = PluginState.ERROR
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
            self.status = PluginState.STOPPED
            
            # Unregister subscriptions
            self._subscriptions.clear()
            self._env_subscriptions.clear()
            
            logger.info(f"Plugin {self.plugin_id} stopped")
            return True
            
        except Exception as e:
            error_msg = f"Plugin {self.plugin_id} error during stop: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.error_log.append(error_msg)
            self.status = PluginState.ERROR
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
            self.status = PluginState.RUNNING
            logger.info(f"Plugin {self.plugin_id} configuration updated")
            return True
            
        except Exception as e:
            error_msg = f"Plugin {self.plugin_id} error during config update: {str(e)}"
            logger.error(error_msg)
            self.error_log.append(error_msg)
            self.status = PluginState.ERROR
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
    
    def process_messages(self, max_messages: Optional[int] = None) -> int:
        """
        Process pending messages from all subscribed topics.
        
        Args:
            max_messages: Maximum messages to process per call (uses default if None)
            
        Returns:
            Number of messages processed
        """
        if max_messages is None:
            max_messages = self.max_messages_per_cycle
        if self.status != PluginState.RUNNING or not self.plugin:
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
                    
                    # Call plugin's callback
                    if hasattr(self.plugin, callback_name):
                        callback = getattr(self.plugin, callback_name)
                        result = callback(msg, self._local_env)
                        
                        # Republish if configured (would go to orchestrator)
                        if result is not None and publish_to:
                            # In multiprocessing version, we skip republishing
                            # (would need to send back to orchestrator)
                            pass
                    
                    messages_processed += 1
                    
                except Exception as e:
                    logger.error(f"Error processing message in {callback_name}: {e}")
                    if self.plugin:
                        self.plugin.on_error(e, f"callback_{callback_name}")
        
        return messages_processed
    
    def get_status(self) -> WorkerStatus:
        """
        Get plugin status information.
        
        Returns:
            WorkerStatus dataclass with plugin status
        """
        return WorkerStatus(
            plugin_id=self.plugin_id,
            status=self.status.value,
            is_running=self.plugin._is_running if self.plugin else False,
            subscriptions=len(self._subscriptions),
            env_subscriptions=len(self._env_subscriptions),
            errors=len(self.error_log)
        )
    
    def get_errors(self) -> List[str]:
        """Get recent error messages."""
        return self.error_log.copy()
    
    def clear_errors(self) -> None:
        """Clear error log."""
        self.error_log.clear()
    
    @staticmethod
    def run(plugin_id: str, descriptor_dict: Dict[str, Any], config: Dict[str, Any], 
            cmd_queue: mp.Queue, resp_queue: mp.Queue) -> None:
        """
        Main entry point for plugin subprocess.
        
        This runs in the plugin process and handles command dispatch.
        
        Args:
            plugin_id: Unique plugin identifier
            descriptor_dict: Plugin descriptor dictionary
            config: Configuration options
            cmd_queue: Queue for receiving commands
            resp_queue: Queue for sending responses
        """
        # Set up logging in subprocess
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Wrap queues in a duplex channel (worker side: receives from cmd_queue, sends to resp_queue)
        channel = Channel(resp_queue, cmd_queue)
        
        worker = PluginWorker(plugin_id)
        
        # Load plugin
        if not worker.load_plugin(descriptor_dict, config):
            return
        
        # Main command loop
        running = True
        while running:
            try:
                # Try to get a command (non-blocking)
                cmd = channel.try_receive(timeout=worker.cmd_receive_timeout)
                
                if cmd is None:
                    # No command available
                    if worker.status == PluginState.RUNNING:
                        worker.process_messages()
                    else:
                        time.sleep(worker.process_sleep_interval)
                    continue
                
                # Dispatch command
                if cmd.command == "get_subscriptions":
                    # Return list of subscription topic names
                    sub_names = [sub["name"] for sub in worker._subscriptions]
                    resp = Responses.success(cmd.cmd_id, sub_names)
                    channel.send(resp)
                
                elif cmd.command == "get_env_subscriptions":
                    # Return list of env subscription keys
                    env_names = [sub["name"] for sub in worker._env_subscriptions]
                    resp = Responses.success(cmd.cmd_id, env_names)
                    channel.send(resp)
                
                elif cmd.command == "register_subscription":
                    topic = worker._safe_arg(cmd.args, 0, "")
                    queue = worker._safe_arg(cmd.args, 1)
                    if queue:
                        # Find this subscription and set its queue
                        for sub in worker._subscriptions:
                            if sub["name"] == topic:
                                sub["queue"] = queue
                                logger.debug(f"Registered queue for subscription '{topic}'")
                                break
                    resp = Responses.success(cmd.cmd_id, True)
                    channel.send(resp)
                
                elif cmd.command == "start":
                    success = worker.start()
                    resp = Responses.success(cmd.cmd_id, success)
                    channel.send(resp)
                
                elif cmd.command == "stop":
                    success = worker.stop()
                    running = False
                    resp = Responses.success(cmd.cmd_id, success)
                    channel.send(resp)
                
                elif cmd.command == "update_config":
                    config = worker._safe_arg(cmd.args, 0, {})
                    success = worker.update_config(config)
                    resp = Responses.success(cmd.cmd_id, success)
                    channel.send(resp)
                
                elif cmd.command == "receive_env_update":
                    update = worker._safe_arg(cmd.args, 0)
                    if update:
                        worker.receive_env_update(update)
                    resp = Responses.success(cmd.cmd_id, True)
                    channel.send(resp)
                
                elif cmd.command == "execute_service":
                    method_name = worker._safe_arg(cmd.args, 0, "")
                    service_args = cmd.args[1:] if len(cmd.args) > 1 else ()
                    try:
                        result = worker.execute_service(method_name, *service_args, **cmd.kwargs)
                        resp = Responses.success(cmd.cmd_id, result)
                    except Exception as e:
                        resp = Responses.error(cmd.cmd_id, str(e))
                    channel.send(resp)
                
                elif cmd.command == "get_status":
                    status = worker.get_status()
                    resp = Responses.success(cmd.cmd_id, status)
                    channel.send(resp)
                
                elif cmd.command == "get_errors":
                    errors = worker.get_errors()
                    resp = Responses.success(cmd.cmd_id, errors)
                    channel.send(resp)
                
                elif cmd.command == "clear_errors":
                    worker.clear_errors()
                    resp = Responses.success(cmd.cmd_id, True)
                    channel.send(resp)
                
                else:
                    logger.warning(f"Unknown command: {cmd.command}")
                    resp = Responses.error(cmd.cmd_id, f"Unknown command: {cmd.command}")
                    channel.send(resp)
                    
            except KeyboardInterrupt:
                running = False
            except Exception as e:
                logger.error(f"Error in plugin process: {e}\n{traceback.format_exc()}")
                running = False
