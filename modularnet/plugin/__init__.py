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
- ManagerProxy: Client-side proxy for direct manager access from plugins
"""

from modularnet.plugin.plugin_base import BasePlugin, PluginState, PluginDescriptor
from modularnet.plugin.environment_manager import EnvironmentManager
from modularnet.plugin.plugin_manager import PluginManager
from modularnet.plugin.plugin_message import (
    Request, Response,
    Subscription, 
    PluginStatus, ProcessStatus
)
from modularnet.plugin.plugin_rpc import PluginMessanger

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
    "Subscription",
    "SubscriptionType",
    "PluginStatus",
    "ProcessStatus",
    "PluginMessanger",
]
