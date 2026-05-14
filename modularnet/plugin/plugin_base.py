
from __future__ import annotations
from enum import Enum
from multiprocessing import shared_memory

from modularnet.logger import Logger
import time
from typing import Dict, List

from torch import multiprocessing as mp

from modularnet.task.task import Task


class PluginStatus(Enum):
    INIT = 0
    SETUP = 1
    RUNNING = 2
    COMPLETE = 4
    ERROR = -1



class PluginBase:

    def __init__(self, options: dict={}, name: str = None, fatal_errors: bool = False, logging=None):
        self.name = name if name is not None else self.__class__.__name__
        self.fatal_errors = fatal_errors
        self.options = options
        self.version = "0.1"
        self.task_main = None
        self.logger = logging if logging is not None else Logger.capture(name=self.name)
        
        self.start_event = mp.Event()
        self.stop_event = mp.Event()


        self.manager = None

        self.status = mp.Value('i', PluginStatus.INIT)  # Shared value for status across processes

    def set_manager(self, manager: PluginManagerBase):
        self.manager = manager

    def manifest(self):
        return {
            "name": self.name,
            "version": self.version
        } 
        
        
    def start(self, wait: bool = False):
        self.start_event.clear()
        self.stop_event.clear()
        kwargs = {
            "status": self.status,
            "options": self.options,
            "fatal_errors": self.fatal_errors    
        }
        self.task_main = Task.create(target=self._run_main, mode="process", kwargs=kwargs)
        self.task_main.start()
        self.start_event.set()
           


    def stop(self, wait: bool = True):
        self.stop_event.set()
        if wait: 
            self.task_main.join()
        

    
    def _run_main(self, options: dict={}, **kwargs):
        if 'status' not in kwargs:
            raise ValueError("Plugin run_main requires 'status' in kwargs")
        self.status = kwargs['status'] # mp.Value
        if not isinstance(self.status, mp.Value):
            raise ValueError("Plugin run_main requires 'status' to be a multiprocessing Value")

        self.status.value = PluginStatus.SETUP
        self.onStart()
        self.options = options
        if 'fatal_errors' in kwargs:
            self.fatal_errors = kwargs['fatal_errors']

        while not self.stop_event.is_set():
            self.status.value = PluginStatus.RUNNING
            try:
                self.onUpdate()
            except Exception as e:
                print(f"{self.__class__.__name__} encountered an error: {e}")
                import traceback
                traceback.print_exc()
                if self.fatal_errors:
                    self.status.value = PluginStatus.ERROR
                    self.stop_event.set()
                    self.onStop()
                    raise e
            time.sleep(0.001)  # Prevent busy waiting
        self.status.value = PluginStatus.FINISH
        self.onStop()
    # to override
    def onStart(self): pass
    def onUpdate(self): pass
    def onStop(self): pass






class PluginManagerBase:
    """Base manager for registering and controlling plugins."""
    def __init__(self):
        self.plugins: Dict[str, PluginBase] = {}
    
    def register_plugin(self, plugin: PluginBase, name: str = None):
        """Register a plugin with the manager."""
        if name is None:
            name = plugin.name
        self.plugins[name] = plugin
        plugin.set_manager(self)
    
    def unregister_plugin(self, plugin: PluginBase, name: str = None):
        """Unregister a plugin with the manager."""
        if name is None:
            name = plugin.name
        if name in self.plugins:
            del self.plugins[name]
        plugin.set_manager(None)
    
    def start_plugin(self, name: str):
        """Start a registered plugin."""
        if name in self.plugins:
            self.plugins[name].start()
    
    def stop_plugin(self, name: str):
        """Stop a registered plugin."""
        if name in self.plugins:
            self.plugins[name].stop()
    
    def start_all_plugins(self):
        """Start all registered plugins."""
        for plugin in self.plugins.values():
            plugin.start()
    
    def stop_all_plugins(self):
        """Stop all registered plugins."""
        for plugin in self.plugins.values():
            plugin.stop()
    
    def get_plugin(self, name: str) -> PluginBase:
        """Get a plugin by name."""
        return self.plugins.get(name)
    
    def list_plugins(self) -> List[str]:
        """List all registered plugin names."""
        return list(self.plugins.keys())
    