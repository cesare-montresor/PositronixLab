

import itertools
from multiprocessing import Lock
from typing import Any, Callable, Dict, Optional
import logging
from torch import multiprocessing as mp
from queue import Empty
import threading
from modularnet.plugin.plugin_message import CommandType, Message, PendingRequest, Request, RequestCallback, RequestEnvGet, RequestEnvGetAll, RequestEnvSet, RequestEnvSubscribe, RequestEnvUnsubscribe, RequestEnvUpdate, RequestPluginCall, RequestPluginErrors, RequestPluginStart, RequestPluginStatus, RequestPluginStatusAll, RequestPluginStop, RequestTopicAnnounce, RequestTopicList, RequestTopicSubscribe, RequestTopicUnsubscribe, Response, Subscriber


logger = logging.getLogger(__name__)



# ============================================================================
# CHANNEL: Bidirectional duplex IPC with proper async response routing
# ============================================================================


class Channel:
    """
    Duplex communication channel with request/response routing.
    """
    
    def Reverse(self) -> Channel:
        """Create a reversed channel for the other side (swapped queues)."""
        rev  = Channel(self.queue_receive, self.queue_send)
        rev.reversed = not self.reversed
        return rev

    def __init__(self, queue_send: mp.Queue=None, queue_receive: mp.Queue=None):
        self.queue_send = queue_send or mp.Queue()
        self.queue_receive = queue_receive or mp.Queue()
        self.reversed = False

    def send(self, message: Message) -> None:
        """Send a message (Request or Response)."""
        self.queue_send.put(message)
    
     # --- Worker side API ---
    def receive(self, timeout: Optional[float] = None) -> Optional[Message]:
        """
        Worker side: block waiting for incoming request.
        
        Args:
            timeout: Max wait time (None = wait forever)
        
        Returns:
            Request object, or None if timeout
        """

        message = None
        try:
            message = self.queue_receive.get(timeout=timeout)
        except Empty: 
            pass
        
        return message
    

class MessageUID:
    """Thread-safe unique ID generator."""
    def __init__(self, minimal=True):
        self._counter = itertools.count()
        self._lock = Lock()
        self.minimal = minimal
        self._last = 0
    
    def __next__(self) -> str:
        with self._lock:
            num = next(self._counter)
            self._last = num
        uid = f"{threading.get_ident()}_{num}"
        if self.minimal: return uid
        return f"{threading.current_thread().name}_{uid}"
            
    def next(self) -> str: self.__next__()
    def current(self) -> str: return self._last


class PluginCoreRPC:
    """
    Wraps bidirectional channel for communication between manager and plugin for a clean API interface:
    - Environment variable access
    - Pub/sub publishing
    - Service invocation
    - Manager state queries
    
    Usage in plugin:
        def on_start(self):
            model = self.manager.get_env("model")
            self.manager.set_env("training_status", "started")
            self.manager.publish("training_topic", {"epoch": 1})
    """

    
    def __init__(self, plugin_id: str, channel: Channel=None, timeout: float = 5.0):
        """
        Initialize manager proxy.
        
        Args:
            plugin_id: Plugin instance identifier
            channel: Bidirectional channel for communication with manager
            timeout: Request timeout in seconds
        """
        self.plugin_id = plugin_id
        self.channel = channel if channel is not None else Channel()
        self.timeout = timeout
        
        self.command_handlers: Dict[str, RequestCallback] = {}

        self.uid = MessageUID()


        # Request/response routing: cmd_id → Queue(maxsize=1)
        self._pending: Dict[str, PendingRequest] = {}
        self._pending_lock = threading.Lock()
        
        # Reader thread routes Responses to reply slots
        self._reader = threading.Thread(target=self._channel_listener, daemon=True)
        self._reader.start()
    
      
    def forward(self, req: Request, wait=True) -> Any:
        forward_req = req.copy()
        
        # Forward the call to the target plugin and wait for response
        remote_result = self.request(forward_req)
        if not wait: return remote_result
        
        remote_result = remote_result.wait()
        if remote_result.error:
            raise Exception(f"Error on forwarding {req}: {remote_result.error}")
        return remote_result.result

    def request(self, req: Request) -> PendingRequest:
        """
        send a request
        """
        if not req.message_id:
            req.message_id = self.uid()
        if not req.sender:
            req.sender = self.plugin_id

        pending_request = PendingRequest(req.message_id,)
        with self._pending_lock:
            self._pending[req.message_id] = pending_request
        self.channel.send(req)
        return pending_request
    

    def _channel_listener(self) -> None:
        """
        Background thread: drain incoming Responses and route to reply slots.
        
        Puts each Response into its pending request's queue.
        Warns on orphaned responses (request already timed out).
        """
        while True:
            try:


                message = self.channel.receive() #(timeout=1.0)
                if message is None: continue  # Timeout, loop again

                if isinstance(message, Request):
                    self._handle_request(message)
                elif isinstance(message, Response):   
                    self._handle_response(message)
                else:
                    logger.error(f"Received unknown message type: {message}")
            except Exception as e:
                logger.error(f"Response router error: {e}")



    def _handle_request(self, req: Request) -> None:
        """Dispatch request to appropriate handler."""
        try:
            command = req.command
            if command not in CommandType:
                raise ValueError(f"Unknown command: {command}")
            if command not in self.command_handlers:
                raise ValueError(f"Command not registered: {command}")

            threading.Thread(target=self._execute_command, args=(req), daemon=True).start() # use pool ?        
        except Exception as e:
            logger.error(f"Request handler error: {e}", exc_info=True)
            self.channel.send(Response(message_id=req.message_id, error=str(e)))
    
    def _handle_response(self, resp: Response) -> bool:
        """Route incoming response to pending request."""
        with self._pending_lock:
            pending_request = self._pending.pop(resp.message_id, None)
        
        if pending_request is None:
            logger.error(f"Received response for unknown message_id={resp.message_id}")
            return False
        
        pending_request.set_result(resp)
        return True
    
    def _execute_command(self, req: Request) -> Any:
        """
        Execute a command and wait for response.
        
        Args:
            command: Command name
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        if req.command not in self.command_handlers:
            raise ValueError(f"Command not registered: {req.command}")
        
        handler = self.command_handlers[req.command]
        result = None
        error = None
        try:
            result = handler(self, req)
        except Exception as e:
            error = str(e)
            logger.error(f"Error occurred while executing command '{req.command}': {e}")

        resp = Response(message_id=req.message_id, result=result, error=error)
        self.channel.send(resp)
        
    
    def on_request(self, command_name: CommandType, func: RequestCallback) -> None:
        """
        Register a command handler for an incoming message
        
        Args:
            command_name: Name of the command
            func: Function to handle the command
        """
        self.command_handlers[command_name] = func


class PluginLifeCycleRPC(PluginCoreRPC):
    """Convience class with manager-specific commands pre-registered.
    PLUGIN_START = "plugin_start"
    PLUGIN_STOP = "plugin_stop"
    PLUGIN_STATUS = "plugin_status"
    PLUGIN_STATUS_ALL = "plugin_status_all"
    PLUGIN_ERRORS = "plugin_errors"
    PLUGIN_CALL = "plugin_call"
    """
    def plugin_start(self):
        """
        Start the plugin process.
        """
        try:
            return self.request(RequestPluginStart())
        except Exception as e:
            logger.error(f"Failed to start plugin: {e}")
    

    def plugin_stop(self) :
        """
        Stop the plugin process.
        """
        try:
            return self.request(RequestPluginStop())
        except Exception as e:
            logger.error(f"Failed to stop plugin: {e}")
        
    def plugin_call(self, plugin_id: str, method_name: str, *args, **kwargs):
        """
        Call a method on the plugin.
        
        Args:
            plugin_id: ID of the plugin to call
            method_name: Name of the method to call
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Method return value
        """
        try:
            return self.request(RequestPluginCall(plugin_id=plugin_id, method_name=method_name, args=args, kwargs=kwargs))
        except Exception as e:
            logger.error(f"Failed to call plugin method '{method_name}': {e}")
            return None
        
    
    # ========================================================================
    # STATUS: Query manager state
    # ========================================================================
    
    def plugin_status(self, plugin_id: str=None) -> PendingRequest | None:
        """
        Get status of another plugin.
        
        Args:
            plugin_id: Plugin identifier
        
        Returns:
            Status dict or None if not found
        """
        try:
            return self.request(RequestPluginStatus(plugin_id=plugin_id))
        except Exception as e:
            logger.debug(f"Failed to get status for {plugin_id}: {e}")
            return None
    
    def plugin_status_all(self):
        """
        List all loaded plugins and their status.
        
        Returns:
            Dictionary of plugin_id -> status info
        """
        try:
            return self.request(RequestPluginStatusAll())
        except Exception as e:
            logger.error(f"Failed to list plugins: {e}")

    
    def plugin_errors(self, plugin_id:str = None):
        """
        Get any errors from the manager or other plugins.
        
        Returns:
            Dictionary of plugin_id -> error info
        """
        try:
            return self.request(RequestPluginErrors(plugin_id=plugin_id))
        except Exception as e:
            logger.error(f"Failed to get plugin errors: {e}")
            return {}

class PluginEnvRPC(PluginCoreRPC):
    """Convience class with plugin-specific commands pre-registered."""

    # ========================================================================
    # ENVIRONMENT VARIABLES
    # ========================================================================
    
    def env_get(self, var_name: str):
        """
        Get environment variable from manager.
        
        Args:
            key: Variable key
            default: Default value if not found
        
        Returns:
            Variable value
        """
        try:
            return self.request(RequestEnvGet(var_name=var_name))
        except Exception as e:
            logger.error(f"Failed to get env variable '{var_name}': {e}")

    
    def env_set(self, var_name: str, value: Any):
        """
        Set environment variable in manager (notifies subscribed plugins).
        
        Args:
            key: Variable key
            value: Variable value
        """
        try:
            return self.request(RequestEnvSet(var_name=var_name, value=value))
        except Exception as e:
            logger.error(f"Failed to set env variable '{var_name}': {e}")
    
    def env_get_all(self):
        """
        Get all environment variables.
        
        Returns:
            Dictionary of all env variables
        """
        try:
            return self.request(RequestEnvGetAll())
        except Exception as e:
            logger.error(f"Failed to get all env variables: {e}")
            return {}
        
    def env_subscribe(self, var_name: str):
        """
            Subscribe to an environment variable.

        """
        try:
            return self.request(RequestEnvSubscribe(var_name=var_name))
        except Exception as e:
            logger.error(f"Failed to subscribe to env variable '{var_name}': {e}")
            return {}
        
    def env_unsubscribe(self, var_name: str):
        """
            Unsubscribe from an environment variable.

        """
        try:
            return self.request(RequestEnvUnsubscribe(var_name=var_name))
        except Exception as e:
            logger.error(f"Failed to unsubscribe from env variable '{var_name}': {e}")

    def env_update(self, var_name: str) :
        """
            Update an environment variable.

        """
        try:
            return self.request(RequestEnvUpdate(var_name=var_name))
        except Exception as e:
            logger.error(f"Failed to notify update env variable '{var_name}': {e}")

class PluginTopicRPC(PluginCoreRPC):
    # ========================================================================
    # PUB/SUB: Publishing
    # ========================================================================
    
    def topic_announce(self, topic: str):
        """
        Announce a topic.
        
        Args:
            topic: Topic name
        """
        try:
            return self.request(RequestTopicAnnounce(topic=topic))
        except Exception as e:
            logger.error(f"Failed to announce topic '{topic}': {e}")

    def topic_subscribe(self, subscriber: Subscriber):
        """
        Subscribe to a topic.
        
        Args:
            topic: Topic name
            callback: Function to call when a message is received
        """
        try:
            return self.request(RequestTopicSubscribe(subscriber=subscriber))
        except Exception as e:
            logger.error(f"Failed to subscribe to '{subscriber}': {e}")

    def topic_unsubscribe(self, subscriber: Subscriber):
        """
        Unsubscribe from a topic.
        
        Args:
            topic: Topic name
            callback: Function to remove from subscribers
        """
        try:
            return self.request(RequestTopicUnsubscribe(subscriber=subscriber))
        except Exception as e:
            logger.error(f"Failed to unsubscribe from '{subscriber}': {e}")

    def topic_list(self, plugin_id: str=None):
        """
        List all published topics.
        Returns:
            List of topic names
        """
        try:
            return self.request(RequestTopicList(plugin_id=plugin_id))
        except Exception as e:
            logger.error(f"Failed to list topics: {e}")
        
class PluginRPC(PluginCoreRPC, PluginLifeCycleRPC, PluginEnvRPC, PluginTopicRPC):
    """Convience class with all command categories pre-registered."""
    pass

