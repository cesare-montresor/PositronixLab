
from __future__ import annotations

from modularnet.logger import Logger
import time
from typing import Dict, List

from torch import multiprocessing as mp

from modularnet.task.task import Task


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
            "options": self.options,
            "fatal_errors": self.fatal_errors    
        }
        self.task_main = Task.create(target=self.run_main, mode="process", kwargs=kwargs)
        self.task_main.start()
        self.start_event.set()
           


    def stop(self, wait: bool = True):
        self.stop_event.set()
        if wait: 
            self.task_main.join()
        

    
    def run_main(self, options: dict={}, **kwargs):
        self.onStart()
        self.options = options
        if 'fatal_errors' in kwargs:
            self.fatal_errors = kwargs['fatal_errors']

        while not self.stop_event.is_set():
            try:
                self.onUpdate()
            except Exception as e:
                print(f"{self.__class__.__name__} encountered an error: {e}")
                import traceback
                traceback.print_exc()
                if self.fatal_errors:
                    self.stop_event.set()
                    self.onStop()
                    raise e
            time.sleep(0.001)  # Prevent busy waiting
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
    


