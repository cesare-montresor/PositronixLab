"""
ModularNet Plugin System: Lightweight multiprocessing-based plugin architecture.

Provides modular plugin execution with pub/sub messaging, environment variables,
and service invocation.

Main Components:
- PluginDescriptor: Serializable plugin metadata
- BasePlugin: Abstract base class for plugins
- PluginManager: Plugin orchestration and management
- PluginWorker: Plugin subprocess worker
- PluginMessages: IPC message protocol (CommandMessage, ResponseMessage)
"""

from .plugin_descriptor import PluginDescriptor
from .plugin_base import BasePlugin, PluginState, Subscription
from .environment_manager import EnvironmentManager, EnvUpdate
from .plugin_manager import PluginManager
from .plugin_messages import (
    CommandMessage, ResponseMessage, CommandType, Commands, Responses, Channel,
    WorkerStatus, ProcessStatus
)

__all__ = [
    "PluginDescriptor",
    "BasePlugin",
    "PluginState",
    "Subscription",
    "EnvironmentManager",
    "EnvUpdate",
    "PluginManager",
    "CommandMessage",
    "ResponseMessage",
    "CommandType",
    "Commands",
    "Responses",
    "Channel",
    "WorkerStatus",
    "ProcessStatus",
]
