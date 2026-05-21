"""
Plugin System Message Protocol (Refactored)

Unified, lean messaging system for inter-process communication:

Core Concepts:
  1. Request/Response: Unified for ALL operations (lifecycle, services, status)
  2. DataExchange: Unified subscription model for pub/sub + env vars
  3. Channel: Proper async request/response routing with background thread
  
Message Flows:
  - Control: Manager sends Request("start"), Worker replies with Response
  - Services: Manager sends Request("invoke_service", ...), Worker replies
  - Pub/Sub: Manager.publish(topic, data) → direct to subscriber queues
  - Env Vars: Manager.set_env(key, value) → broadcasts EnvUpdate to subscribers

No Commands/Responses wrappers, no double indirection. Just Request and Response.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, Callable
from enum import Enum
import uuid
import threading
import logging
from queue import Queue, Empty

logger = logging.getLogger(__name__)


# ============================================================================
# CORE MESSAGE TYPES
# ============================================================================

@dataclass
class Request:
    """
    Unified request for all operations.
    
    All manager→worker communication uses this:
    - Lifecycle: Request("start"), Request("stop"), Request("update_config", config)
    - Services: Request("invoke_service", "method_name", *args, **kwargs)
    - Subscriptions: Request("get_data_exchanges")
    - Status: Request("get_status"), Request("get_errors")
    
    Worker replies with Response(cmd_id, result=...) or Response(cmd_id, error=...)
    """
    command: str
    cmd_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    
    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}


@dataclass
class Response:
    """
    Unified response for all request results.
    
    Attributes:
        cmd_id: Matches Request.cmd_id for routing
        result: Return value (None if error)
        error: Error message (None on success)
    """
    cmd_id: str
    result: Any = None
    error: Optional[str] = None
    
    @property
    def ok(self) -> bool:
        """Whether succeeded."""
        return self.error is None
    
    @property
    def failed(self) -> bool:
        """Whether failed."""
        return self.error is not None


# ============================================================================
# DATA EXCHANGE SUBSCRIPTIONS (Unified pub/sub + env)
# ============================================================================

class SubscriptionType(Enum):
    """Types of subscriptions."""
    TOPIC = "topic"        # Pub/sub topic subscription
    ENV_VAR = "env_var"    # Environment variable subscription


@dataclass
class DataExchange:
    """
    Unified subscription descriptor for pub/sub topics and env vars.
    
    Single model for both kinds of subscriptions:
    - Topic: when data arrives on "topic_name", call callback(data, env)
    - EnvVar: when "var_name" changes, call callback(EnvUpdate)
    
    Attributes:
        sub_type: Type of subscription (TOPIC or ENV_VAR)
        name: Topic name or env var key
        callback: Method name on plugin to invoke
        republish_to: Optional topic to republish callback results (topic-only)
    """
    sub_type: SubscriptionType
    name: str
    callback: str
    republish_to: Optional[str] = None
    
    @classmethod
    def topic(cls, name: str, callback: str, republish_to: Optional[str] = None) -> "DataExchange":
        """Create a topic subscription."""
        return cls(SubscriptionType.TOPIC, name, callback, republish_to)
    
    @classmethod
    def env_var(cls, name: str, callback: str) -> "DataExchange":
        """Create an env var subscription."""
        return cls(SubscriptionType.ENV_VAR, name, callback)


@dataclass
class EnvUpdate:
    """Represents an environment variable change."""
    key: str
    value: Any
    old_value: Any = None


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


# ============================================================================
# CHANNEL: Bidirectional duplex IPC with proper async response routing
# ============================================================================

class Channel:
    """
    Duplex communication channel with request/response routing.
    
    Provides both synchronous request/response (blocking) and fire-and-forget sends.
    Uses a background reader thread to route incoming Responses to their pending requests.
    
    Manager side:
        channel = Channel(cmd_queue, resp_queue)
        result = channel.request("start")              # blocks until response
        channel.send("publish", "topic", data)         # fire-and-forget
    
    Worker side:
        channel = Channel(resp_queue, cmd_queue)       # queues swapped!
        req = channel.receive()                        # blocks until request
        channel.reply(req.cmd_id, result=data)         # send response
    """
    
    def __init__(self, send_queue: Any, receive_queue: Any):
        """
        Initialize duplex channel.
        
        Args:
            send_queue: Queue for outgoing messages
            receive_queue: Queue for incoming messages
        """
        self._send_queue = send_queue
        self._receive_queue = receive_queue
        
        # Request/response routing: cmd_id → Queue(maxsize=1)
        self._pending: Dict[str, Queue] = {}
        self._pending_lock = threading.Lock()
        
        # Reader thread routes Responses to reply slots
        self._reader = threading.Thread(target=self._response_router, daemon=True)
        self._reader.start()
    
    # --- Synchronous request/response ---
    
    def request(self, command: str, *args, timeout: float = 5.0, **kwargs) -> Any:
        """
        Synchronous request: send command and wait for response.
        
        Args:
            command: Command name
            *args: Positional arguments
            timeout: Max wait time in seconds
            **kwargs: Keyword arguments
        
        Returns:
            Response result value
        
        Raises:
            TimeoutError: If no response within timeout
            RuntimeError: If worker returned error
        """
        req = Request(command, args=args, kwargs=kwargs)
        slot: Queue = Queue(maxsize=1)
        
        with self._pending_lock:
            self._pending[req.cmd_id] = slot
        
        try:
            self._send_queue.put(req)
            try:
                resp: Response = slot.get(timeout=timeout)
            except Empty:
                raise TimeoutError(f"No response for '{command}' within {timeout}s")
        finally:
            with self._pending_lock:
                self._pending.pop(req.cmd_id, None)
        
        if not resp.ok:
            raise RuntimeError(resp.error)
        return resp.result
    
    # --- Fire-and-forget send ---
    
    def send(self, command: str, *args, **kwargs) -> None:
        """
        Fire-and-forget send: send command, no response expected.
        
        Args:
            command: Command name
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        self._send_queue.put(Request(command, args=args, kwargs=kwargs))
    
    # --- Worker side API ---
    
    def receive(self, timeout: Optional[float] = None) -> Optional[Request]:
        """
        Worker side: block waiting for incoming request.
        
        Args:
            timeout: Max wait time (None = wait forever)
        
        Returns:
            Request object, or None if timeout
        """
        try:
            return self._receive_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def reply(self, cmd_id: str, result: Any = None, error: Optional[str] = None) -> None:
        """
        Worker side: send response back to manager.
        
        Args:
            cmd_id: Command ID to match with request
            result: Return value
            error: Error message (if error)
        """
        self._send_queue.put(Response(cmd_id=cmd_id, result=result, error=error))
    
    # --- Internal response router ---
    
    def _response_router(self) -> None:
        """
        Background thread: drain incoming Responses and route to reply slots.
        
        Runs on manager side. Puts each Response into its pending request's queue.
        Warns on orphaned responses (request already timed out).
        """
        while True:
            try:
                msg = self._receive_queue.get()
                if not isinstance(msg, Response):
                    # This is a Request (worker side receiving) — put back
                    self._receive_queue.put(msg)
                    continue
                
                with self._pending_lock:
                    slot = self._pending.get(msg.cmd_id)
                
                if slot is not None:
                    slot.put(msg)
                else:
                    logger.debug(f"Orphaned response for cmd_id={msg.cmd_id}, discarding")
            except Exception as e:
                logger.error(f"Response router error: {e}")
