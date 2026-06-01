"""
Plugin Worker: Subprocess handler for plugin execution.

PluginWorker.run() is the entry point for each plugin subprocess.
It creates plugin instance, handles incoming requests, and routes to handlers.

Command dispatch:
  - All incoming requests routed via @command_handler decorator
  - Services invoked via reflection (self.plugin.method_name(*args, **kwargs))
  - No manual if/elif chains
"""


import logging
import importlib
import threading
from typing import Any, Callable, Dict, List, Optional

from modularnet.plugin.plugin_message import Commands, PendingRequest, Request, Subscriber, Subscription
from modularnet.plugin.plugin_base import BasePlugin, PluginDescriptor

from modularnet.plugin.plugin_rpc import  Channel,PluginRPC

from torch import multiprocessing as mp

logger = logging.getLogger(__name__)

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
        channel: Channel,
    ) -> None:
        """
        Entry point for plugin subprocess.
        
        Args:
            plugin_id: Plugin instance identifier
            descriptor_dict: Serialized plugin descriptor
            config: Plugin configuration
            cmd_queue: Queue for incoming requests
            resp_queue: Queue for outgoing responses
            manager_req_queue: Queue for sending requests to manager
            manager_resp_queue: Queue for receiving responses from manager
        """
        worker = PluginWorker(
            plugin_id,
            descriptor_dict,
            config,
            channel
        )
        worker.run_loop()
    
    def __init__(
        self,
        plugin_id: str,
        descriptor_dict: Dict[str, Any],
        config: Dict[str, Any],
        channel: Channel,
        #cmd_queue: Any,
        #resp_queue: Any,
        #manager_req_queue: Any = None,
        #manager_resp_queue: Any = None
    ):
        """Initialize worker."""
        self.plugin_id = plugin_id
        self.descriptor = PluginDescriptor.from_dict(descriptor_dict)
        self.config = config
        self.channel = channel  # Note: queues swapped for worker
       
        
        # Topic subscriptions: topic_name -> queue
        
        self.topic_published = set()
        self.topic_subscriptions: Dict[str, Subscription] = {}
        self.topic_subscribers: Dict[str, Dict[str,Subscriber]] = {}
        # Topic subscriptions: topic_name -> queue
        self.env_subscriptions: Dict[str, Callable] = {}
        
        # Error log
        self.error_log: list = []
        # Instantiate the plugin
        self.plugin: Optional[BasePlugin] = None
        
        
        # Create manager proxy for direct manager access
        self.rpc: PluginRPC = None

        self._init_plugin()
        self._init_messanger()
        
    
    def _init_plugin(self) -> None:
        """Instantiate plugin from descriptor."""
        try:
            # Import plugin class
            
            module = importlib.import_module(self.descriptor.package)
            plugin_class: BasePlugin = getattr(module, self.descriptor.class_name)
            

            # Instantiate plugin with proxy
            self.plugin = plugin_class(self, self.plugin_id, self.config )
            logger.info(f"Plugin {self.plugin_id} instantiated")
        except Exception as e:
            logger.error(f"Failed to init plugin {self.plugin_id}: {e}")
            raise

    def _init_messanger(self) -> None:
        """Initialize command handlers from plugin methods.

            + on request
            > method
        

            + PLUGIN_START = "plugin_start"
            + PLUGIN_STOP = "plugin_stop"
            + PLUGIN_CALL = "plugin_call"
            + PLUGIN_STATUS = "plugin_status"
            - PLUGIN_STATUS_ALL = "plugin_status_all"
            + PLUGIN_ERRORS = "plugin_errors"

            > ENV_SET = "env_set"
            > ENV_GET = "env_get"
            > ENV_GET_ALL = "env_get_all"
            > ENV_SUBSCRIBE = "env_subscribe"
            > ENV_UNSUBSCRIBE = "env_unsubscribe"
            + ENV_UPDATE = "env_update"

            * TOPIC_SUBSCRIBE = "topic_subscribe"
            * TOPIC_UNSUBSCRIBE = "topic_unsubscribe"
            * TOPIC_PUBLISH = "topic_publish"
            * TOPIC_LIST = "topic_list"
        """
        try:
            self.rpc = PluginRPC(self.plugin_id, self.channel)
            self.rpc.on_request(Commands.PLUGIN_START, self._respond_start)
            self.rpc.on_request(Commands.PLUGIN_STOP, self._respond_stop)
            self.rpc.on_request(Commands.PLUGIN_CALL, self._respond_call)
            self.rpc.on_request(Commands.PLUGIN_STATUS, self._respond_status)
            self.rpc.on_request(Commands.PLUGIN_ERRORS, self._respond_errors)
            
            self.rpc.on_request(Commands.ENV_UPDATE, self._respond_env_update)
            
            self.rpc.on_request(Commands.TOPIC_SUBSCRIBE, self._respond_topic_subscribe)
            self.rpc.on_request(Commands.TOPIC_UNSUBSCRIBE, self._respond_topic_unsubscribe)
            self.rpc.on_request(Commands.TOPIC_LIST, self._request_topic_list)

        except Exception as e:
            logger.error(f"Failed to init messanger: {e}")
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
                for topic, subscription in list(self.topic_subscriptions.items()):
                    try:
                        # Non-blocking get with tiny timeout
                        data = subscription.queue.get_nowait()
                        subscription.callback(data)
                    except Exception:
                        threading.Event().wait(0)
                        continue
                
                # Small sleep to avoid busy loop
                
            except Exception as e:
                logger.error(f"Topic listener error: {e}")




    # ========================================================================
    # Commands to send
    # ========================================================================
    
    """Commands that plugin can send to manager or other plugins.
        > ENV_SET = "env_set"
        > ENV_GET = "env_get"
        > ENV_GET_ALL = "env_get_all"
        > ENV_SUBSCRIBE = "env_subscribe"
        > ENV_UNSUBSCRIBE = "env_unsubscribe"
    """
    def env_set(self, var_name: str, value: Any, wait: bool=True) -> PendingRequest|bool:
        """Set an environment variable."""
        pending: PendingRequest = self.rpc.env_set( var_name, value)
        if not wait: return pending
        
        # Wait for response if no callback provided
        response = pending.wait()
        if response.error:
            logger.error(f"Error setting env variable: {response.error}")
            self.error_log.append(str(response.error))
            return False
        return response.result
    
    def env_get(self, var_name: str, wait: bool=True) -> PendingRequest|Any:
        """Get an environment variable."""
        pending: PendingRequest = self.rpc.env_get(var_name)
        if not wait: return pending
        
        # Wait for response if no callback provided
        response = pending.wait()
        if response.error:
            logger.error(f"Error getting env variable: {response.error}")
            self.error_log.append(str(response.error))
            return None
        return response.result
    
    def env_get_all(self, wait: bool=True) -> PendingRequest|dict:
        """Get all environment variables."""
        pending: PendingRequest = self.rpc.env_get_all()
        if not wait: return pending
        
        # Wait for response if no callback provided
        response = pending.wait()
        if response.error:
            logger.error(f"Error getting all env variables: {response.error}")
            self.error_log.append(str(response.error))
            return None
        return response.result
    
    def env_subscribe(self, var_name: str, callback: Callable) -> bool:
        """Get all environment variables."""

        pending: PendingRequest = self.rpc.env_subscribe(var_name)
        
        # Wait for response if no callback provided
        response = pending.wait()
        if response.error:
            logger.error(f"Error getting all env variables: {response.error}")
            self.error_log.append(str(response.error))
            return False
        
        self.env_subscriptions[var_name] = callback
        return True
    
    def env_unsubscribe(self, var_name: str, wait: bool=True) -> bool:
        """Get all environment variables."""
        pending: PendingRequest = self.rpc.env_unsubscribe(var_name)
        
        # Wait for response if no callback provided
        response = pending.wait()
        if response.error:
            logger.error(f"Error getting all env variables: {response.error}")
            self.error_log.append(str(response.error))
            return False
        
        if var_name in self.env_subscriptions:
            del self.env_subscriptions[var_name]
        return True
    

    """
    Commands that plugin can send to manager or other plugins.
        * TOPIC_SUBSCRIBE = "topic_subscribe"
        * TOPIC_UNSUBSCRIBE = "topic_unsubscribe"
        * TOPIC_PUBLISH = "topic_publish"
        * TOPIC_LIST = "topic_list"
    """

    def topic_subscribe(self, topic: str, callback: Callable) -> bool:
        """Get all environment variables."""
        subscription = Subscription(topic, mp.Queue(), callback)
        subscriber = Subscriber(self.plugin_id, topic, subscription.queue)


        pending: PendingRequest = self.rpc.topic_subscribe(subscriber)
        # Wait for response if no callback provided
        response = pending.wait()
        if response.error:
            logger.error(f"Error subscribing to topic '{topic}': {response.error}")
            self.error_log.append(str(response.error))
            return False
        if response.result:
            self.topic_subscriptions[topic] = subscription
        return response.result
    
    def topic_unsubscribe(self, topic: str) -> bool:
        """Get all environment variables."""
        if not topic in self.topic_subscriptions:
            logger.warning(f"Plugin {self.plugin_id} not subscribed to topic '{topic}'")
            return True
        
        subscription = self.topic_subscriptions[topic] 
        
        subscriber = Subscriber(self.plugin_id, topic, subscription.queue)


        pending: PendingRequest = self.rpc.topic_unsubscribe(subscriber)
        # Wait for response if no callback provided
        response = pending.wait()
        if response.error:
            logger.error(f"Error unsubscribing to topic '{topic}': {response.error}")
            self.error_log.append(str(response.error))
            return False
        
        if response.result:
            del self.topic_subscriptions[topic]

        return response.result


    def topic_list(self, wait: bool=True) -> PendingRequest|list:
        """Get all environment variables."""
        pending: PendingRequest = self.rpc.topic_list()
        if not wait: return pending
        
        # Wait for response if no callback provided
        response = pending.wait()
        if response.error:
            logger.error(f"Error getting all env variables: {response.error}")
            self.error_log.append(str(response.error))
            return None
        return response.result
    

    def topic_publish(self, topic: str, data: Any) -> None:
        """Get all environment variables."""
        if topic not in self.topic_subscribers:
            self.topic_subscribers[topic] = {}
        topic_subscribers: dict[str, Subscriber] = self.topic_subscribers.get(topic)

        if topic not in self.topic_published:
            reponse = self.rpc.topic_publish(topic, None).wait()
            if reponse.error:
                logger.error(f"Error registering topic '{topic}': {reponse.error}")
                self.error_log.append(str(reponse.error))
                return False
            
            manager_subscribers = reponse.result
            self.topic_published.add(topic)
            for subscriber in reponse.result:
                topic_subscribers[subscriber.plugin_id] = subscriber

        
        
        for subscriber in topic_subscribers.values():
            try:
                subscriber.queue.put_nowait(data)
            except Exception as e:
                logger.error(f"Failed to publish to subscriber {subscriber.plugin_id} on topic '{topic}': {e}")
        
        
        return True

    # ========================================================================
    # On Command Recievd
    # ========================================================================
    
    def _respond_start(self, rpc: PluginRPC, req: Request) -> bool:
        """Start plugin."""
        try:
            self.plugin.on_start()
            return True
        except Exception as e:
            logger.error(f"Failed to start plugin: {e}")
            raise
    
    def _respond_stop(self, rpc: PluginRPC, req: Request) -> bool:
        """Stop plugin."""
        try:
            self.plugin.on_stop()
            return True
        except Exception as e:
            logger.error(f"Failed to stop plugin: {e}")
            raise
    
    
    def _respond_env_update(self, rpc: PluginRPC, req: Request) -> None:
        """Receive environment variable change notification."""
        var_name = req.args[0]
        callback = self.env_subscriptions[var_name]
        callback(var_name)
        return True
    
    def _respond_status(self, rpc: PluginRPC, req: Request) -> Dict[str, Any]:
        """Get plugin status."""
        topics_read = self.topic_subscriptions.keys()
        topics_write = self.topic_published
        return {
            "plugin_id": self.plugin_id,
            "status": "running",
            "is_running": True,
            "subs": len(topics_read),
            "pubs": len(topics_write),
            "errors": len(self.error_log)
        }
    
    def _respond_errors(self, rpc: PluginRPC, req: Request) -> list:
        """Get error log."""
        return self.error_log.copy()
    

    def _respond_call(self, rpc: PluginRPC, req: Request) -> Any:
        """Invoke service method on plugin via reflection."""
        command = req.command
        args = req.args
        kwargs = req.kwargs
        
        method = getattr(self.plugin, command)
        return method(*args, **kwargs)
            
    
    def _respond_topic_subscribe(self, rpc: PluginRPC, req: Request) -> bool:
        """Handle topic subscribe request."""
        plugin_id = req.sender
        subscriber: Subscriber = req.args[0]
        topic = subscriber.topic
        
        if topic not in self.topic_subscribers:
            self.topic_subscribers[topic] = {}
        self.topic_subscribers[topic][plugin_id] = subscriber
        return True
    
    def _respond_topic_unsubscribe(self, rpc: PluginRPC, req: Request) -> bool:
        """Handle topic unsubscribe request."""
        plugin_id = req.sender
        subscriber: Subscriber = req.args[0]
        topic = subscriber.topic

        if topic not in self.topic_subscribers: return True
        if plugin_id not in self.topic_subscribers[topic]: return True
        del self.topic_subscribers[topic][plugin_id]
        
        return True
    
    def _request_topic_list(self, rpc: PluginRPC, req: Request) -> list:
        """Handle topic list request."""
        return self.topic_published.copy()
    
