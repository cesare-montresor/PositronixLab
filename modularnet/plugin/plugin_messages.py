"""
Plugin System Message Protocol

Defines all message types for inter-process communication between PluginManager
(main process) and PluginWorker (subprocesses).

Message Flow:
    PluginManager → CommandMessage → PluginWorker (via cmd_queue)
    PluginWorker → ResponseMessage → PluginManager (via resp_queue)
    PluginManager → EnvUpdate → PluginWorker (as command arg)
    PluginWorker ← Pub/Sub messages ← PluginManager (via topic queues)
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum
import uuid

from modularnet.plugin.environment_manager import EnvUpdate


# ============================================================================
# COMMAND MESSAGES: Main → Worker
# ============================================================================

class CommandType(Enum):
    """Types of commands that can be sent to a PluginWorker."""
    # Lifecycle
    START = "start"
    STOP = "stop"
    UPDATE_CONFIG = "update_config"
    
    # Pub/Sub
    GET_SUBSCRIPTIONS = "get_subscriptions"
    REGISTER_SUBSCRIPTION = "register_subscription"
    
    # Environment
    GET_ENV_SUBSCRIPTIONS = "get_env_subscriptions"
    RECEIVE_ENV_UPDATE = "receive_env_update"
    
    # Services
    EXECUTE_SERVICE = "execute_service"
    
    # Status
    GET_STATUS = "get_status"
    GET_ERRORS = "get_errors"
    CLEAR_ERRORS = "clear_errors"


@dataclass
class CommandMessage:
    """
    Message sent from PluginManager to PluginWorker.
    
    Attributes:
        command: Command type (see CommandType enum)
        cmd_id: Unique command identifier (for matching with response)
        args: Positional arguments for the command
        kwargs: Keyword arguments for the command
    """
    command: str
    cmd_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure kwargs is always a dict."""
        if self.kwargs is None:
            self.kwargs = {}


# ============================================================================
# RESPONSE MESSAGES: Worker → Main
# ============================================================================

@dataclass
class ResponseMessage:
    """
    Message sent from PluginWorker back to PluginManager.
    
    Attributes:
        command: Command type (see CommandType enum)
        cmd_id: Unique command identifier (matches CommandMessage.cmd_id)
        result: Return value of the command (None if error)
        error: Error message if command failed (None on success)
    """
    cmd_id: str
    result: Any = None
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        """Whether the command succeeded."""
        return self.error is None
    
    @property
    def failed(self) -> bool:
        """Whether the command failed."""
        return self.error is not None


# ============================================================================
# ENVIRONMENT UPDATE MESSAGES
# ============================================================================

# EnvUpdate is imported from environment_manager (see above)


# ============================================================================
# STATUS DATA STRUCTURES
# ============================================================================

@dataclass
class WorkerStatus:
    """
    Status information from a plugin worker (subprocess).
    
    Attributes:
        plugin_id: Unique plugin identifier
        status: Lifecycle status (INITIALIZED, LOADED, RUNNING, STOPPED, ERROR)
        is_running: Whether the plugin is actively running
        subscriptions: Number of pub/sub subscriptions
        env_subscriptions: Number of environment variable subscriptions
        errors: Number of logged errors
    """
    plugin_id: str
    status: str
    is_running: bool
    subscriptions: int
    env_subscriptions: int
    errors: int


@dataclass
class ProcessStatus:
    """
    Status information from the plugin manager about a process.
    
    Includes process-level info (alive, pid) and optionally the worker status.
    
    Attributes:
        plugin_id: Unique plugin identifier
        process_alive: Whether the subprocess is still running
        process_pid: Process ID of the subprocess
        worker_status: Worker status (if available), None if process dead or error
    """
    plugin_id: str
    process_alive: bool
    process_pid: Optional[int]
    worker_status: Optional[WorkerStatus] = None


# ============================================================================
# PUB/SUB MESSAGE TYPES
# ============================================================================

@dataclass
class PubSubMessage:
    """
    Generic pub/sub message that can be published to topics.
    
    These are user-defined payloads published via manager.publish().
    They can be any serializable Python object.
    
    Note: For now, pub/sub messages are just passed through directly.
          In future, could add metadata like:
          - source_plugin_id
          - timestamp
          - priority
    """
    # Currently just a marker - actual content is the message itself
    pass


# ============================================================================
# COMMAND CONSTRUCTORS (Helpers for common commands)
# ============================================================================

class Commands:
    """Helper class for constructing common command messages."""
    
    @staticmethod
    def start(cmd_id: str) -> CommandMessage:
        """Construct a START command."""
        return CommandMessage(CommandType.START.value, cmd_id)
    
    @staticmethod
    def stop(cmd_id: str) -> CommandMessage:
        """Construct a STOP command."""
        return CommandMessage(CommandType.STOP.value, cmd_id)
    
    @staticmethod
    def update_config(cmd_id: str, config: dict) -> CommandMessage:
        """Construct an UPDATE_CONFIG command."""
        return CommandMessage(CommandType.UPDATE_CONFIG.value, cmd_id, (config,))
    
    @staticmethod
    def get_subscriptions(cmd_id: str) -> CommandMessage:
        """Construct a GET_SUBSCRIPTIONS command."""
        return CommandMessage(CommandType.GET_SUBSCRIPTIONS.value, cmd_id)
    
    @staticmethod
    def register_subscription(cmd_id: str, topic: str, queue: Any) -> CommandMessage:
        """Construct a REGISTER_SUBSCRIPTION command."""
        return CommandMessage(CommandType.REGISTER_SUBSCRIPTION.value, cmd_id, (topic, queue))
    
    @staticmethod
    def get_env_subscriptions(cmd_id: str) -> CommandMessage:
        """Construct a GET_ENV_SUBSCRIPTIONS command."""
        return CommandMessage(CommandType.GET_ENV_SUBSCRIPTIONS.value, cmd_id)
    
    @staticmethod
    def receive_env_update(cmd_id: str, update: EnvUpdate) -> CommandMessage:
        """Construct a RECEIVE_ENV_UPDATE command."""
        return CommandMessage(CommandType.RECEIVE_ENV_UPDATE.value, cmd_id, (update,))
    
    @staticmethod
    def execute_service(cmd_id: str, method_name: str, *args, **kwargs) -> CommandMessage:
        """Construct an EXECUTE_SERVICE command."""
        return CommandMessage(CommandType.EXECUTE_SERVICE.value, cmd_id, (method_name, *args), kwargs)
    
    @staticmethod
    def get_status(cmd_id: str) -> CommandMessage:
        """Construct a GET_STATUS command."""
        return CommandMessage(CommandType.GET_STATUS.value, cmd_id)
    
    @staticmethod
    def get_errors(cmd_id: str) -> CommandMessage:
        """Construct a GET_ERRORS command."""
        return CommandMessage(CommandType.GET_ERRORS.value, cmd_id)
    
    @staticmethod
    def clear_errors(cmd_id: str) -> CommandMessage:
        """Construct a CLEAR_ERRORS command."""
        return CommandMessage(CommandType.CLEAR_ERRORS.value, cmd_id)


# ============================================================================
# RESPONSE CONSTRUCTORS (Helpers for common responses)
# ============================================================================

class Responses:
    """Helper class for constructing response messages."""
    
    @staticmethod
    def success(cmd_id: str, result: Any = None) -> ResponseMessage:
        """Construct a successful response."""
        return ResponseMessage(cmd_id, result=result)
    
    @staticmethod
    def error(cmd_id: str, error_msg: str) -> ResponseMessage:
        """Construct an error response."""
        return ResponseMessage(cmd_id, error=error_msg)


# ============================================================================
# MESSAGE PROTOCOL DOCUMENTATION
# ============================================================================

__doc__ += """

COMMAND PROTOCOL
================

1. START
   Manager → Worker: "Start the plugin"
   Worker → Manager: bool (success/failure)

2. STOP
   Manager → Worker: "Stop the plugin"
   Worker → Manager: bool (success/failure)

3. UPDATE_CONFIG
   Manager → Worker: ("Update config with...", config_dict)
   Worker → Manager: bool (success/failure)

4. GET_SUBSCRIPTIONS
   Manager → Worker: "What topics do you subscribe to?"
   Worker → Manager: List[str] (topic names)

5. REGISTER_SUBSCRIPTION
   Manager → Worker: ("Register for topic", topic, queue)
   Worker → Manager: bool (success/failure)

6. GET_ENV_SUBSCRIPTIONS
   Manager → Worker: "What env vars do you watch?"
   Worker → Manager: List[str] (env var keys)

7. RECEIVE_ENV_UPDATE
   Manager → Worker: ("Env var changed", EnvUpdate)
   Worker → Manager: None (fire-and-forget)

8. EXECUTE_SERVICE
   Manager → Worker: ("Call method", method_name, *args, **kwargs)
   Worker → Manager: Any (method return value)

9. GET_STATUS
   Manager → Worker: "What's your status?"
   Worker → Manager: Dict (status info)

10. GET_ERRORS
    Manager → Worker: "What errors have you logged?"
    Worker → Manager: List[str] (error messages)

11. CLEAR_ERRORS
    Manager → Worker: "Clear error log"
    Worker → Manager: None


PUB/SUB MESSAGE PROTOCOL
========================

Topic Messages are sent directly (not wrapped):
   Manager.publish(topic, data) → Queue → Worker callback

Example:
   manager.publish("sensor_data", {"temp": 25.5})
   → worker receives {"temp": 25.5} directly
   → calls plugin.on_sensor_data({"temp": 25.5}, env_cache)


ENVIRONMENT UPDATE PROTOCOL
===========================

Env vars are broadcast via command:
   Manager.set_env_variable(key, value)
   → Manager creates EnvUpdate(key, value, old_value)
   → For each subscribed worker:
       Send CommandMessage(..., "receive_env_update", (update,))
   → Worker calls plugin.on_key_change(update)
"""


# ============================================================================
# DUPLEX CHANNEL (Wraps send/receive queue pairs)
# ============================================================================

class Channel:
    """
    Bidirectional communication channel for duplex IPC.
    
    Wraps a send queue and receive queue into a single interface, making
    bidirectional communication clearer and more symmetric.
    
    The same Channel can be used from both sides:
    - Sender uses send() and receive() for request/response
    - Receiver uses receive() and send() for response
    
    Example (Manager side):
        channel = Channel(cmd_queue, response_queue)
        channel.send(CommandMessage(...))
        response = channel.receive(timeout=10)
    
    Example (Worker side):
        channel = Channel(cmd_queue, response_queue)
        msg = channel.receive()  # blocks until command
        channel.send(ResponseMessage(...))
    """
    
    def __init__(self, send_queue: Any, receive_queue: Any):
        """
        Initialize a duplex channel.
        
        Args:
            send_queue: Queue for outgoing messages
            receive_queue: Queue for incoming messages
        """
        self._send_queue = send_queue
        self._receive_queue = receive_queue
    
    def send(self, message: Any) -> None:
        """
        Send a message through the channel.
        
        Args:
            message: The message to send (must be picklable)
        
        Raises:
            Exception: If queue operations fail
        """
        self._send_queue.put(message)
    
    def receive(self, timeout: Optional[float] = None) -> Any:
        """
        Receive a message from the channel.
        
        Args:
            timeout: Maximum time to wait in seconds (None = wait forever)
        
        Returns:
            The received message
        
        Raises:
            TimeoutError: If no message arrives within timeout
            Exception: If queue operations fail
        """
        try:
            return self._receive_queue.get(timeout=timeout)
        except Exception as e:
            # Check if it's an Empty exception (from queue module)
            if "Empty" in str(type(e).__name__):
                raise TimeoutError(f"No message received within {timeout}s")
            raise
    
    def try_receive(self, timeout: float = 0.1) -> Optional[Any]:
        """
        Try to receive a message without blocking indefinitely.
        
        Args:
            timeout: Time to wait in seconds before returning None (default 0.1s)
        
        Returns:
            The received message, or None if timeout
        """
        try:
            return self._receive_queue.get(timeout=timeout)
        except Exception:
            return None
