"""
Plugin Manager (torch.multiprocessing backend): Central orchestration for the multiprocessing-based plugin system.

Responsibilities:
1. Plugin lifecycle management (load, start, stop, unload)
2. Pub/sub topic routing
3. Environment variable management and broadcasting
4. Service request/response dispatch
5. Plugin status monitoring

Uses torch.multiprocessing instead of Ray for simpler, lighter-weight process management.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple
from threading import Thread, Event
import time
import uuid

try:
    import torch.multiprocessing as mp
except ImportError:
    import multiprocessing as mp

# Set fork mode for queue sharing between processes
try:
    mp.set_start_method('fork', force=True)
except RuntimeError:
    pass  # Already set

from modularnet.plugin.plugin_descriptor import PluginDescriptor
from modularnet.plugin.plugin_handler import PluginHandler, CommandMessage, ResponseMessage
from modularnet.plugin.environment_manager import EnvironmentManager, EnvUpdate

logger = logging.getLogger(__name__)


class PluginOrchestrator:
    """
    Manages all plugin orchestration using multiprocessing.
    
    Provides:
    - Plugin loading, starting, stopping
    - Pub/sub message routing
    - Environment variable management
    - Service invocation
    - Status monitoring
    """
    
    def __init__(self):
        """Initialize plugin orchestrator."""
        self._plugins: Dict[str, Dict[str, Any]] = {}  # plugin_id -> {process, queues, etc}
        self._topics: Dict[str, List[mp.Queue]] = {}
        self._env_subscriptions: Dict[str, List[str]] = {}  # key -> [plugin_ids]
        self.env_manager = EnvironmentManager()
    
    def register_plugin(self, plugin_id: str, process: mp.Process, cmd_queue: mp.Queue, resp_queue: mp.Queue) -> bool:
        """
        Register a plugin process.
        
        Args:
            plugin_id: Unique plugin identifier
            process: Multiprocessing Process object
            cmd_queue: Queue for sending commands to plugin
            resp_queue: Queue for receiving responses from plugin
            
        Returns:
            True if successful
        """
        if plugin_id in self._plugins:
            logger.warning(f"Plugin {plugin_id} already registered, overwriting")
        
        self._plugins[plugin_id] = {
            "process": process,
            "cmd_queue": cmd_queue,
            "resp_queue": resp_queue,
            "status": "REGISTERED"
        }
        logger.info(f"Plugin {plugin_id} registered")
        return True
    
    def unregister_plugin(self, plugin_id: str) -> bool:
        """
        Unregister a plugin process.
        
        Args:
            plugin_id: Plugin identifier
            
        Returns:
            True if successful
        """
        if plugin_id not in self._plugins:
            logger.warning(f"Plugin {plugin_id} not found")
            return False
        
        plugin_info = self._plugins[plugin_id]
        process = plugin_info["process"]
        
        # Terminate process if still running
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
            if process.is_alive():
                process.kill()
        
        del self._plugins[plugin_id]
        
        # Clean up env subscriptions
        for key in list(self._env_subscriptions.keys()):
            if plugin_id in self._env_subscriptions[key]:
                self._env_subscriptions[key].remove(plugin_id)
        
        logger.info(f"Plugin {plugin_id} unregistered")
        return True
    
    # --- Pub/Sub Topic Management ---
    
    def subscribe(self, topic: str, max_backlog: int = 100) -> mp.Queue:
        """
        Create a subscription to a pub/sub topic.
        
        Args:
            topic: Topic name
            max_backlog: Maximum queued messages
            
        Returns:
            Queue for receiving messages
        """
        if topic not in self._topics:
            self._topics[topic] = []
        
        sub_queue = mp.Queue(maxsize=max_backlog)
        self._topics[topic].append(sub_queue)
        
        logger.debug(f"New subscription to topic '{topic}'")
        return sub_queue
    
    def publish(self, topic: str, msg: Any) -> int:
        """
        Publish message to a topic.
        
        Args:
            topic: Topic name
            msg: Message object
            
        Returns:
            Number of subscribers that received the message
        """
        delivered = 0
        failed = 0
        
        # Get queues from both old _topics dict and new registry
        queues_to_deliver = []
        if topic in self._topics:
            queues_to_deliver.extend(self._topics[topic])
        
        for q in queues_to_deliver:
            try:
                q.put(msg, block=False)
                delivered += 1
            except Exception:
                # Queue full - drop message (backpressure)
                failed += 1
        
        if failed > 0:
            logger.debug(f"Published to '{topic}': {delivered} delivered, {failed} dropped")
        
        return delivered
    
    # --- Environment Variable Management ---
    
    def subscribe_env(self, key: str, plugin_id: str) -> bool:
        """
        Subscribe plugin to environment variable changes.
        
        Args:
            key: Environment variable key
            plugin_id: Plugin identifier
            
        Returns:
            True if successful
        """
        if key not in self._env_subscriptions:
            self._env_subscriptions[key] = []
        
        if plugin_id not in self._env_subscriptions[key]:
            self._env_subscriptions[key].append(plugin_id)
        
        logger.debug(f"Plugin {plugin_id} subscribed to env var '{key}'")
        return True
    
    def set_env_variable(self, key: str, value: Any) -> None:
        """
        Set environment variable and notify subscribers.
        
        Args:
            key: Variable name
            value: New value
        """
        old_value = self.env_manager.get(key)
        self.env_manager.set(key, value)
        
        # Notify subscribed plugins via command queue
        if key in self._env_subscriptions:
            update = EnvUpdate(key=key, value=value, old_value=old_value)
            for plugin_id in self._env_subscriptions[key]:
                if plugin_id in self._plugins:
                    cmd = CommandMessage(
                        cmd_id=str(uuid.uuid4()),
                        command="receive_env_update",
                        args=(update,),
                        kwargs={}
                    )
                    try:
                        self._plugins[plugin_id]["cmd_queue"].put(cmd, block=False)
                    except Exception as e:
                        logger.error(f"Failed to send env update to {plugin_id}: {e}")
    
    def get_env_variable(self, key: str, default: Any = None) -> Any:
        """Get environment variable value."""
        return self.env_manager.get(key, default)
    
    def get_all_env_variables(self) -> Dict[str, Any]:
        """Get all environment variables."""
        return self.env_manager.get_all()
    
    # --- Service Invocation ---
    
    def invoke_service(
        self, 
        plugin_id: str, 
        method_name: str, 
        args: tuple = (), 
        kwargs: dict = None
    ) -> Any:
        """
        Invoke a service method on a plugin.
        
        Args:
            plugin_id: Plugin identifier
            method_name: Method name
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            Method return value
        """
        if plugin_id not in self._plugins:
            raise ValueError(f"Plugin {plugin_id} not found")
        
        kwargs = kwargs or {}
        plugin_info = self._plugins[plugin_id]
        cmd_queue = plugin_info["cmd_queue"]
        resp_queue = plugin_info["resp_queue"]
        
        # Send command
        cmd_id = str(uuid.uuid4())
        cmd = CommandMessage(
            cmd_id=cmd_id,
            command="execute_service",
            args=(method_name, *args),
            kwargs=kwargs
        )
        
        try:
            cmd_queue.put(cmd, block=False)
        except Exception as e:
            raise RuntimeError(f"Failed to send service command to {plugin_id}: {e}")
        
        # Wait for response with timeout
        timeout = 30.0
        start = time.time()
        while time.time() - start < timeout:
            try:
                resp = resp_queue.get(block=False)
                if resp.cmd_id == cmd_id:
                    if resp.error:
                        raise Exception(resp.error)
                    return resp.result
                else:
                    # Put it back if it's for a different command
                    resp_queue.put(resp)
            except:
                pass
            
            time.sleep(0.01)
        
        raise TimeoutError(f"Service call to {plugin_id}.{method_name} timed out after {timeout}s")
    
    # --- Status and Monitoring ---
    
    def get_plugin_status(self, plugin_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific plugin."""
        if plugin_id not in self._plugins:
            return None
        
        plugin_info = self._plugins[plugin_id]
        process = plugin_info["process"]
        
        return {
            "plugin_id": plugin_id,
            "alive": process.is_alive(),
            "pid": process.pid if process else None,
        }
    
    def get_all_plugin_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all plugins."""
        statuses = {}
        for plugin_id in self._plugins.keys():
            status = self.get_plugin_status(plugin_id)
            if status:
                statuses[plugin_id] = status
        
        return statuses
    
    def list_plugins(self) -> List[str]:
        """List all registered plugin IDs."""
        return list(self._plugins.keys())


class PluginManager:
    """
    High-level plugin manager interface for multiprocessing backend.
    
    Provides user-friendly methods for managing plugins:
    - Load/start/stop/unload plugins
    - Publish/subscribe to topics
    - Get/set environment variables
    - Invoke services
    - Monitor plugin status
    """
    
    def __init__(self):
        """Initialize plugin manager."""
        self.orchestrator = PluginOrchestrator()
        self._plugin_handlers: Dict[str, Dict[str, Any]] = {}
        self._processing_thread: Optional[Thread] = None
        self._stop_event = Event()
        self._subscription_queues: Dict[str, Dict[str, Any]] = {}  # plugin_id -> {topic -> queue}
        # Store subscription queues per plugin - use Manager to create shareable queues
        self._plugin_subscription_queues: Dict[str, Dict[str, Any]] = {}  # plugin_id -> {topic -> queue}
        # Create a manager for cross-process sharing of queues (Manager.Queue can be pickled)
        self._manager = mp.Manager()
        self._manager_queues: Dict[str, Dict[str, Any]] = {}  # plugin_id -> {topic -> queue}
    
    def load_plugin(
        self,
        descriptor: PluginDescriptor,
        config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Load a plugin from descriptor.
        
        Args:
            descriptor: PluginDescriptor with package/class/version info
            config: Configuration options (overrides descriptor defaults)
            
        Returns:
            True if successful
        """
        plugin_id = descriptor.name
        config = config or {}
        
        try:
            # Create queues for communication with plugin process
            cmd_queue = mp.Queue()
            resp_queue = mp.Queue()
            
            # Create and start plugin process
            handler_args = (plugin_id, descriptor.to_dict(), config, cmd_queue, resp_queue)
            process = mp.Process(
                target=PluginHandler.run,
                args=handler_args,
                name=f"plugin-{plugin_id}"
            )
            process.start()
            
            # Register with orchestrator
            self.orchestrator.register_plugin(plugin_id, process, cmd_queue, resp_queue)
            
            # Store handler info
            self._plugin_handlers[plugin_id] = {
                "process": process,
                "cmd_queue": cmd_queue,
                "resp_queue": resp_queue,
                "descriptor": descriptor
            }
            
            # Initialize subscription queue tracking for this plugin
            self._plugin_subscription_queues[plugin_id] = {}
            
            logger.info(f"Plugin {descriptor} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading plugin {descriptor}: {e}")
            if plugin_id in self.orchestrator._plugins:
                self.orchestrator.unregister_plugin(plugin_id)
            return False
    
    def unload_plugin(self, plugin_id: str) -> bool:
        """
        Unload a plugin.
        
        Args:
            plugin_id: Plugin identifier
            
        Returns:
            True if successful
        """
        try:
            # Stop if running
            self.stop_plugin(plugin_id)
            
            # Unregister from orchestrator
            self.orchestrator.unregister_plugin(plugin_id)
            
            if plugin_id in self._plugin_handlers:
                del self._plugin_handlers[plugin_id]
            
            logger.info(f"Plugin {plugin_id} unloaded")
            return True
            
        except Exception as e:
            logger.error(f"Error unloading plugin {plugin_id}: {e}")
            return False
    
    def start_plugin(self, plugin_id: str) -> bool:
        """
        Start a loaded plugin.
        
        Args:
            plugin_id: Plugin identifier
            
        Returns:
            True if successful
        """
        if plugin_id not in self._plugin_handlers:
            logger.error(f"Plugin {plugin_id} not loaded")
            return False
        
        try:
            plugin_info = self._plugin_handlers[plugin_id]
            cmd_queue = plugin_info["cmd_queue"]
            resp_queue = plugin_info["resp_queue"]
            
            # Wait briefly for plugin to load
            time.sleep(0.1)
            
            # Get subscription info from plugin
            sub_cmd_id = str(uuid.uuid4())
            cmd = CommandMessage(
                cmd_id=sub_cmd_id,
                command="get_subscriptions",
                args=(),
                kwargs={}
            )
            cmd_queue.put(cmd, block=False)
            
            subscriptions = []
            env_subscriptions = []
            
            # Wait for subscription response with cmd_id matching
            timeout = time.time() + 0.5
            while time.time() < timeout:
                try:
                    response = resp_queue.get(block=False)
                    if response.cmd_id == sub_cmd_id:
                        subscriptions = response.result if response.result else []
                        break
                    else:
                        # Put it back if it's for a different command
                        resp_queue.put(response)
                except:
                    time.sleep(0.01)
            
            # Get env subscriptions
            env_cmd_id = str(uuid.uuid4())
            cmd = CommandMessage(
                cmd_id=env_cmd_id,
                command="get_env_subscriptions",
                args=(),
                kwargs={}
            )
            cmd_queue.put(cmd, block=False)
            
            # Wait for env subscription response with cmd_id matching
            timeout = time.time() + 0.5
            while time.time() < timeout:
                try:
                    response = resp_queue.get(block=False)
                    if response.cmd_id == env_cmd_id:
                        env_subscriptions = response.result if response.result else []
                        break
                    else:
                        # Put it back if it's for a different command
                        resp_queue.put(response)
                except:
                    time.sleep(0.01)
            
            # Create queues for each subscription and send to plugin
            for sub_name in subscriptions:
                # Use Manager.Queue so it can be pickled and sent through cmd_queue
                sub_queue = self._manager.Queue(maxsize=100)
                
                # Store locally
                self._plugin_subscription_queues[plugin_id][sub_name] = sub_queue
                
                # Add to orchestrator's topics
                if sub_name not in self.orchestrator._topics:
                    self.orchestrator._topics[sub_name] = []
                self.orchestrator._topics[sub_name].append(sub_queue)
                
                # Send subscription queue to plugin via command
                link_cmd_id = str(uuid.uuid4())
                cmd = CommandMessage(
                    cmd_id=link_cmd_id,
                    command="register_subscription",
                    args=(sub_name, sub_queue),
                    kwargs={}
                )
                cmd_queue.put(cmd, block=False)
                
                # Wait for registration response
                timeout = time.time() + 0.3
                while time.time() < timeout:
                    try:
                        response = resp_queue.get(block=False)
                        if response.cmd_id == link_cmd_id:
                            break
                        else:
                            resp_queue.put(response)
                    except:
                        time.sleep(0.01)
                
                logger.debug(f"Plugin {plugin_id} subscription created for topic '{sub_name}'")
            
            # Register env subscriptions
            for env_key in env_subscriptions:
                self.orchestrator.subscribe_env(env_key, plugin_id)
                logger.debug(f"Plugin {plugin_id} subscribed to env key '{env_key}'")
            
            # Send start command
            start_cmd_id = str(uuid.uuid4())
            cmd = CommandMessage(
                cmd_id=start_cmd_id,
                command="start",
                args=(),
                kwargs={}
            )
            cmd_queue.put(cmd, block=False)
            
            # Wait for start response
            timeout = time.time() + 0.5
            while time.time() < timeout:
                try:
                    response = resp_queue.get(block=False)
                    if response.cmd_id == start_cmd_id:
                        logger.info(f"Plugin {plugin_id} started")
                        return response.result if response.result else True
                    else:
                        # Put it back if it's for a different command
                        resp_queue.put(response)
                except:
                    time.sleep(0.01)
            
            logger.warning(f"Plugin {plugin_id} start timed out, assuming success")
            return True
        except Exception as e:
            logger.error(f"Error starting plugin {plugin_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def stop_plugin(self, plugin_id: str) -> bool:
        """
        Stop a running plugin.
        
        Args:
            plugin_id: Plugin identifier
            
        Returns:
            True if successful
        """
        if plugin_id not in self._plugin_handlers:
            return False
        
        try:
            plugin_info = self._plugin_handlers[plugin_id]
            cmd_queue = plugin_info["cmd_queue"]
            
            # Send stop command
            cmd = CommandMessage(
                cmd_id=str(uuid.uuid4()),
                command="stop",
                args=(),
                kwargs={}
            )
            cmd_queue.put(cmd, block=False)
            
            logger.info(f"Plugin {plugin_id} stopped")
            return True
        except Exception as e:
            logger.error(f"Error stopping plugin {plugin_id}: {e}")
            return False
    
    def update_plugin_config(self, plugin_id: str, config: Dict[str, Any]) -> bool:
        """
        Update plugin configuration.
        
        Args:
            plugin_id: Plugin identifier
            config: New configuration
            
        Returns:
            True if successful
        """
        if plugin_id not in self._plugin_handlers:
            return False
        
        try:
            plugin_info = self._plugin_handlers[plugin_id]
            cmd_queue = plugin_info["cmd_queue"]
            
            # Send update_config command
            cmd = CommandMessage(
                cmd_id=str(uuid.uuid4()),
                command="update_config",
                args=(config,),
                kwargs={}
            )
            cmd_queue.put(cmd, block=False)
            
            logger.info(f"Plugin {plugin_id} configuration updated")
            return True
        except Exception as e:
            logger.error(f"Error updating config for {plugin_id}: {e}")
            return False
    
    def start_all_plugins(self) -> None:
        """Start all loaded plugins."""
        for plugin_id in self._plugin_handlers.keys():
            self.start_plugin(plugin_id)

    def stop_all_plugins(self) -> None:
        """Stop all running plugins."""
        for plugin_id in self._plugin_handlers.keys():
            self.stop_plugin(plugin_id)
    
    # --- Pub/Sub ---
    
    def publish(self, topic: str, data: Any) -> int:
        """
        Publish data to a topic.
        
        Args:
            topic: Topic name
            data: Data to publish
            
        Returns:
            Number of subscribers that received the message
        """
        return self.orchestrator.publish(topic, data)
    
    # --- Environment Variables ---
    
    def set_env_variable(self, key: str, value: Any) -> None:
        """
        Set an environment variable (broadcast to all subscribers).
        
        Args:
            key: Variable name
            value: Value
        """
        self.orchestrator.set_env_variable(key, value)
    
    def get_env_variable(self, key: str, default: Any = None) -> Any:
        """Get an environment variable."""
        return self.orchestrator.get_env_variable(key, default)
    
    def get_all_env_variables(self) -> Dict[str, Any]:
        """Get all environment variables."""
        return self.orchestrator.get_all_env_variables()
    
    # --- Services ---
    
    def invoke_service(
        self, 
        plugin_id: str, 
        method_name: str, 
        *args, 
        **kwargs
    ) -> Any:
        """
        Invoke a service method on a plugin.
        
        Args:
            plugin_id: Plugin identifier
            method_name: Method name
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Method result
        """
        return self.orchestrator.invoke_service(
            plugin_id, 
            method_name, 
            args, 
            kwargs
        )
    
    # --- Monitoring ---
    
    def get_plugin_status(self, plugin_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a plugin."""
        return self.orchestrator.get_plugin_status(plugin_id)
    
    def get_all_plugin_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all plugins."""
        return self.orchestrator.get_all_plugin_status()
    
    def list_plugins(self) -> List[str]:
        """List all loaded plugin IDs."""
        return self.orchestrator.list_plugins()
    
    # --- Processing Loop ---
    
    def start_processing_loop(self, interval: float = 0.1) -> None:
        """
        Start background thread that processes messages.
        
        Args:
            interval: Processing interval in seconds
        """
        if self._processing_thread:
            logger.warning("Processing loop already running")
            return
        
        self._stop_event.clear()
        self._processing_thread = Thread(
            target=self._process_loop,
            args=(interval,),
            daemon=True
        )
        self._processing_thread.start()
        logger.info("Plugin processing loop started")
    
    def stop_processing_loop(self) -> None:
        """Stop the background processing thread."""
        if not self._processing_thread:
            return
        
        self._stop_event.set()
        self._processing_thread.join(timeout=5.0)
        self._processing_thread = None
        logger.info("Plugin processing loop stopped")
    
    def _process_loop(self, interval: float) -> None:
        """Background loop that processes messages for all plugins."""
        while not self._stop_event.is_set():
            try:
                # Check that all plugin processes are alive
                for plugin_id in list(self._plugin_handlers.keys()):
                    handler_info = self._plugin_handlers[plugin_id]
                    process = handler_info["process"]
                    if not process.is_alive():
                        logger.warning(f"Plugin {plugin_id} process died unexpectedly")
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(interval)
    
    def shutdown(self) -> None:
        """Shutdown the plugin manager."""
        self.stop_processing_loop()
        self.stop_all_plugins()
        
        # Terminate all processes
        for plugin_id in list(self._plugin_handlers.keys()):
            self.unload_plugin(plugin_id)
