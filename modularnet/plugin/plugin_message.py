"""
Plugin System Message Protocol (Refactored)

Unified, lean messaging system for inter-process communication:

Core Concepts:
  1. Request/Response: Unified for ALL operations (lifecycle, services, status)
  2. DataExchange: Unified subscription model for pub/sub + env vars
  3. Channel: Proper async request/response routing with background thread
  
Message Flows:
  - Control: Manager sends Request("start"), Worker replies with Response
  - Services: Manager sends Request("call_service", ...), Worker replies
  - Pub/Sub: Manager.publish(topic, data) → direct to subscriber queues
  - Env Vars: Manager.set_env(key, value) → broadcasts EnvUpdate to subscribers

No Commands/Responses wrappers, no double indirection. Just Request and Response.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from enum import Enum
import uuid
import logging
from concurrent.futures import Future


from typing import TYPE_CHECKING
from torch import multiprocessing as mp

if TYPE_CHECKING: from modularnet.plugin.plugin_worker import PluginRPC

logger = logging.getLogger(__name__)


RequestCallback = Callable[["PluginRPC","Request"], Any]
ResponseCallback = Callable[["PluginRPC","Response"], Any]

# ============================================================================
# CORE MESSAGE TYPES
# ============================================================================

class Commands(Enum):
    """Standard commands for plugin control."""
    PLUGIN_START = "plugin_start"
    PLUGIN_STOP = "plugin_stop"
    PLUGIN_CALL = "plugin_call"
    PLUGIN_STATUS = "plugin_status"
    PLUGIN_STATUS_ALL = "plugin_status_all"
    PLUGIN_ERRORS = "plugin_errors"

    ENV_SET = "env_set"
    ENV_GET = "env_get"
    ENV_GET_ALL = "env_get_all"
    ENV_SUBSCRIBE = "env_subscribe"
    ENV_UNSUBSCRIBE = "env_unsubscribe"
    ENV_UPDATE = "env_update"

    TOPIC_SUBSCRIBE = "topic_subscribe"
    TOPIC_UNSUBSCRIBE = "topic_unsubscribe"
    TOPIC_PUBLISH = "topic_publish"
    TOPIC_LIST = "topic_list"

    def __str__(self): return str(self.value)


@dataclass
class Message:
    """Base message class for all communication."""
    sender: str
    message_id: str 

@dataclass
class Request(Message):
    """
    Unified request for all operations.
    Worker replies with Response(message_id, result=...) or Response(message_id, error=...)
    """
    command: Commands
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)

@dataclass
class Response(Message):
    """
    Unified response for all request results.
     - result: The return value of the request (if successful)
     - error: An error message (if an error occurred)
    """
    result: Any = None
    error: Optional[str] = None
    
class PendingRequest():
    def __init__(self, message_id: str, callback: Callable=None):
        super().__init__()
        self.future = Future()
        self.message_id = message_id
        if callback:
            self.future.add_done_callback(callback)

    def on_ready(self, callback: Callable):
        self.future.add_done_callback(callback)
    
    def set_result(self, result: Response|Any):
        if not isinstance(result, Response):
            result = Response(message_id=self.message_id, result=result)
        self.future.set_result(result)

    def set_error(self, error: str|Exception|Any):
        self.future.set_result(Response(message_id=self.message_id, error=str(error)))

    def wait(self, timeout=None) -> None | Response:
        try:
            result: Response = self.future.result(timeout=timeout)
            return result
        except TimeoutError:
            return None
        
    def __hash__(self):
        return hash(self.message_id)

@dataclass
class Subscription:
    """
    Represents a subscription to a topic or environment variable.
    """
    topic: str
    queue: mp.Queue # For topic subscriptions, queue to receive data
    callback: Callable  # Method on plugin to call with new data or env update

@dataclass
class Subscriber:
    """Represents a subscriber to a topic or environment variable."""
    plugin_id: str
    topic: str
    queue: mp.Queue

# ============================================================================
# STATUS DATA STRUCTURES
# ============================================================================

@dataclass
class WorkerStatus:
    """Status snapshot from plugin worker."""
    plugin_id: str
    status: str  # "initialized", "loaded", "running", "stopped", "error"
    is_running: bool
    subscriptions: int      # Number of data exchange subscriptions
    errors: int


@dataclass
class ProcessStatus:
    """Status of a plugin from manager perspective."""
    plugin_id: str
    process_alive: bool
    process_pid: Optional[int]
    worker_status: Optional[WorkerStatus] = None

