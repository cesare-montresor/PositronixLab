



from modularnet.plugin.plugin import Plugin
from modularnet.modularobserver.modularobserver import ModularObserver


class ModularObserverPlugin(Plugin):
    def __init__(self, manager):
        super().__init__(manager)
        
        self.modular_observer = ModularObserver(self.manager)
        self.model = None
        self.taps = [] 


    def onStart(self):
        super().onStart()

        self.model = self.env_get("model")

        self.modular_observer.watch(self.model)
        