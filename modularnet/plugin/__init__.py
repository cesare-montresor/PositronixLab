"""
ModularNet Plugin System: Lightweight multiprocessing-based plugin architecture.

Provides modular plugin execution with pub/sub messaging, environment variables,
and service invocation.

Main Components:
- PluginDescriptor: Serializable plugin metadata
- BasePlugin: Abstract base class for plugins
- PluginManager: Plugin orchestration and management
- PluginWorker: Plugin subprocess worker
- PluginMessages: Unified IPC message protocol (Request, Response)
- DataExchange: Unified subscription model (topics + env vars)
"""

from .plugin_descriptor import PluginDescriptor
from .plugin_base import BasePlugin, PluginState
from .environment_manager import EnvironmentManager, EnvUpdate
from .plugin_manager import PluginManager
from .plugin_messages import (
    Request, Response, Channel,
    DataExchange, SubscriptionType,
    WorkerStatus, ProcessStatus
)

__all__ = [
    "PluginDescriptor",
    "BasePlugin",
    "PluginState",
    "EnvironmentManager",
    "EnvUpdate",
    "PluginManager",
    "Request",
    "Response",
    "Channel",
    "DataExchange",
    "SubscriptionType",
    "WorkerStatus",
    "ProcessStatus",
]
