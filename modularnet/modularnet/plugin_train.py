import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, List

import torch
from torch import cuda
import torchinfo

from modularnet.dataprovider.higgs_dataprovider import BaseDataProvider
from modularnet.deeplearning import DeeplearningCommon
from modularnet.modularobserver.layerobserver import LayerObserver
from modularnet.modularobserver.modularobserver import ModularObserver


from modularnet.plugin.plugin_transport import _IpcEventPool, _IpcEventSlot, IpcEventCuda, IpcEventPoolManaged
from modularnet.plugin.plugin_worker import PluginWorker
from modularnet.spacemath.spacemath import SpaceMath
from modularnet.trial.trial_config import TrialConfig
from modularnet.dataprovider.dataprovider import DataProvider
from modularnet.metamodel.metamodel import MetaModel
from modularnet.plugin import BasePlugin, PluginDescriptor
from torchsummary import summary

logger = logging.getLogger(__name__)

class ObservableTrainPlugin(BasePlugin):
    
    @classmethod
    def default_options(cls) -> Dict[str, Any]:
        return {
            "device": None,
            "debug": False,
            "compute_scores": False,
            "num_workers_data": 2,
            "ipc_event_pool_max_size": 8,
        }
 
    @classmethod
    def Descriptor(cls, options: Dict[str, Any] = None) -> PluginDescriptor:
        return super().Descriptor(
            version="1.0.0",
            author="ModularNet Team",
            description="Plugin that sets up the model and data provider for deep learning trials.",
            options=options
        )
 
    def __init__(
        self,
        worker: PluginWorker,
        plugin_id: str,
        options: Dict[str, Any],        
    ):
        super().__init__(worker, plugin_id, options )
        
        self.device = options.get("device")
        self.debug = options.get("debug")
        self.compute_scores: bool = options.get("compute_scores")  # If True, send scores; if False, send activations
        self.num_workers_data: int = options.get("num_workers_data")
        self.ipc_event_pool_max_size: int = options.get("ipc_event_pool_max_size", 8)
 
        self.trial_config: TrialConfig = None  # Will be set in on_start via env_get("trial_config")
        self.model: MetaModel = None
        self.dataprovider: BaseDataProvider = None
        self.modular_observer: ModularObserver = None  # Placeholder for modular observer if needed
        self.trainer: DeeplearningCommon = None  # Will be initialized in main()
 
        # Created in on_start(), once the model's resolved device is known.
        # cuda.Stream() with no device arg binds to whatever the *current*
        # device happens to be at construction time, which may not match the
        # model's eventual device in a spawned worker that hasn't set its
        # device context yet -- so we defer creation rather than doing it here.
        self.forward_stream: Optional[cuda.Stream] = None
 
        # One interprocess-event pool per layer_id, created lazily the first
        # time that layer reports scores. See _IpcEventPool for why this
        # replaces a single reused event / unbounded per-call allocation.
        self._ipc_event_pools: Dict[str, _IpcEventPool] = {}
 
 
    def on_start(self) -> None:
        logger.info(f"[{self.plugin_id}] Starting deeplearning setup...")
 
        self.trial_config = self.env_get("trial_config")
        
        if not self.trial_config:
            raise ValueError("trial_config not provided in config")
        
        # Pass model device to dataset
        self.dataprovider = DataProvider.from_config(
            self.trial_config,
            num_workers=self.num_workers_data
        )
 
        # Create model first to determine device
        self.model = MetaModel(self.plugin_id, self.trial_config, device=self.device)
 
        # Now that the model's resolved device is known, bind the forward
        # stream to it explicitly instead of relying on ambient current-device
        # state at __init__ time.
        model_device = getattr(self.model, "device", None) or self.device or torch.device("cuda")
        self.forward_stream = cuda.Stream(device=model_device)
 
        #self.env_set("dataprovider", self.dataprovider)
 
        self.modular_observer = ModularObserver(self.trial_config, self.model)
        self.modular_observer.static_map()
        x,_ = next(iter(self.dataprovider.train))
        self.modular_observer.runtime_map(x)  # Pass only the input tensor to runtime_map
        self.modular_observer.on_forward_layer(self.on_layer_forward)
 
        self.observed_layers = {observer.layer_id: observer.to_dict() for observer in self.modular_observer.observers}
 
        self.env_set("observed_layers", self.observed_layers)  # Make observed layers available to other plugins
        self.env_set("model_summary", self.modular_observer.latest_graph_dict )  # Make model summary available to other plugins
        
 
 
        self.trainer = DeeplearningCommon(
            config=self.trial_config,
            model=self.model,
            dp=self.dataprovider
        )
 
        self.trainer.on_epoch_start(self.on_epoch_start)
        self.trainer.on_epoch_end(self.on_epoch_end)
 
    def on_graph_update(self, graph_dict: Dict[str, Any]):
        self.env_set("observed_layers", self.observed_layers)  # Make observed layers available to other plugins
        self.env_set("model_summary", self.modular_observer.latest_graph_dict )  # Make model summary available to other plugins
 
    def on_epoch_start(self, epoch: int, epochs: int):
        self.env_set("epoch", (epoch, epochs))
    
    def on_epoch_end(self, epoch: int, epochs: int, metrics: Dict[str, float]):
        self.topic_publish("metrics", {"layer_id": "model", "metric": metrics, "name": "training"})
        
    
    def main(self):
        autoencoder = self.trial_config.task  == 'autoencoder'
        self.trainer.train(lr=self.trial_config.learning_rate, epoch_total=self.trial_config.epochs, autoencoder=autoencoder)
        self.stop()
 
 
    def on_layer_forward(self, layer: LayerObserver, x_in: torch.Tensor, x_out: torch.Tensor):
        if self.compute_scores:
            self.on_layer_forward_scores(layer, x_in, x_out)
        else:
            self.on_layer_forward_activations(layer, x_in, x_out)
 
    def on_layer_forward_activations(self, layer: LayerObserver, x_in: torch.Tensor, x_out: torch.Tensor):
        #TODO: Should I send scores? should i use cuda timelines ? 
 
        data = {
            'layer_id': layer.layer_id,
            # .clone() is required, not optional: under torch.compile(mode=
            # "reduce-overhead") x_out can be a CUDA graph pool buffer that
            # gets overwritten in-place on the next forward pass, before the
            # subscriber on the other side of topic_publish has read it.
            # .detach() alone only cuts the autograd link -- it does not copy
            # storage.
            "activations": x_out.detach().clone()
        }
        self.topic_publish("activations", data)
        
    
 
    def on_layer_forward_scores(self, layer: LayerObserver, x_in: torch.Tensor, x_out: torch.Tensor):
        #TODO: Should I send scores? should i use cuda timelines ? 
 
        self.forward_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.forward_stream):
            scores = SpaceMath.activations_to_scores(
                x_out.detach(), layer.layer_type, layer.activation,
                self.trial_config.modular_space_active_q, cnn_locality_weight=0.5
            )
            event = self._next_ipc_event(layer.layer_id)
            event.record()
        data = {
            'layer_id': layer.layer_id,
            'scores': scores,
            'ready': event
        }
 
        self.topic_publish("scores", data)
 
    def _next_ipc_event(self, layer_id: str) -> cuda.Event:
        """
        Returns an interprocess CUDA event for this layer from a
        demand-grown, capped pool (created lazily on first use).
 
        This replaces minting a fresh torch.cuda.Event(interprocess=True) on
        every single forward call, which is what previously caused unbounded
        IPC handle growth (thousands of live handles per epoch across all
        layers). It also replaces a single reused-per-layer event, which is
        unsafe: record() can re-arm an event before a slower downstream
        consumer has finished reading the data tied to its previous record().
        See _IpcEventPool for the growth strategy, the query()-based reuse
        heuristic, and its caveats.
        """
        pool = self._ipc_event_pools.get(layer_id)
        if pool is None:
            pool = _IpcEventPool(max_size=self.ipc_event_pool_max_size)
            self._ipc_event_pools[layer_id] = pool
        return pool.next(layer_id)



class ObservableTrainPlugin(BasePlugin):
    
    @classmethod
    def default_options(cls) -> Dict[str, Any]:
        return {
            "device": None,
            "debug": False,
            "compute_scores": False,
            "num_workers_data": 2,
            "ipc_event_pool_max_size": 8,
        }
 
    @classmethod
    def Descriptor(cls, options: Dict[str, Any] = None) -> PluginDescriptor:
        return super().Descriptor(
            version="1.0.0",
            author="ModularNet Team",
            description="Plugin that sets up the model and data provider for deep learning trials.",
            options=options
        )
 
    def __init__(
        self,
        worker: PluginWorker,
        plugin_id: str,
        options: Dict[str, Any],        
    ):
        super().__init__(worker, plugin_id, options )
        
        self.device = options.get("device")
        self.debug = options.get("debug")
        self.compute_scores: bool = options.get("compute_scores")  # If True, send scores; if False, send activations
        self.num_workers_data: int = options.get("num_workers_data")
        self.ipc_event_pool_max_size: int = options.get("ipc_event_pool_max_size", 8)
 
        self.trial_config: TrialConfig = None  # Will be set in on_start via env_get("trial_config")
        self.model: MetaModel = None
        self.dataprovider: BaseDataProvider = None
        self.modular_observer: ModularObserver = None  # Placeholder for modular observer if needed
        self.trainer: DeeplearningCommon = None  # Will be initialized in main()
 
        # Created in on_start(), once the model's resolved device is known.
        # cuda.Stream() with no device arg binds to whatever the *current*
        # device happens to be at construction time, which may not match the
        # model's eventual device in a spawned worker that hasn't set its
        # device context yet -- so we defer creation rather than doing it here.
        self.forward_stream: Optional[cuda.Stream] = None
 
        # One interprocess-event pool per layer_id, created lazily the first
        # time that layer reports scores. See _IpcEventPool for why this
        # replaces a single reused event / unbounded per-call allocation.
        self._ipc_event_pools: Dict[str, IpcEventPoolManaged] = {}
 
 
    def on_start(self) -> None:
        logger.info(f"[{self.plugin_id}] Starting deeplearning setup...")
 
        self.trial_config = self.env_get("trial_config")
        
        if not self.trial_config:
            raise ValueError("trial_config not provided in config")
        
        # Pass model device to dataset
        self.dataprovider = DataProvider.from_config(
            self.trial_config,
            num_workers=self.num_workers_data
        )
 
        # Create model first to determine device
        self.model = MetaModel(self.plugin_id, self.trial_config, device=self.device)
 
        # Now that the model's resolved device is known, bind the forward
        # stream to it explicitly instead of relying on ambient current-device
        # state at __init__ time.
        model_device = getattr(self.model, "device", None) or self.device or torch.device("cuda")
        self.forward_stream = cuda.Stream(device=model_device)
 
        #self.env_set("dataprovider", self.dataprovider)
 
        self.modular_observer = ModularObserver(self.trial_config, self.model)
        self.modular_observer.static_map()
        x,_ = next(iter(self.dataprovider.train))
        self.modular_observer.runtime_map(x)  # Pass only the input tensor to runtime_map
        self.modular_observer.on_forward_layer(self.on_layer_forward)
 
        self.observed_layers = {observer.layer_id: observer.to_dict() for observer in self.modular_observer.observers}
 
        self.env_set("observed_layers", self.observed_layers)  # Make observed layers available to other plugins
        self.env_set("model_summary", self.modular_observer.latest_graph_dict )  # Make model summary available to other plugins
        
 
 
        self.trainer = DeeplearningCommon(
            config=self.trial_config,
            model=self.model,
            dp=self.dataprovider
        )
 
        self.trainer.on_epoch_start(self.on_epoch_start)
        self.trainer.on_epoch_end(self.on_epoch_end)
 
    def on_graph_update(self, graph_dict: Dict[str, Any]):
        self.env_set("observed_layers", self.observed_layers)  # Make observed layers available to other plugins
        self.env_set("model_summary", self.modular_observer.latest_graph_dict )  # Make model summary available to other plugins
 
    def on_epoch_start(self, epoch: int, epochs: int):
        self.env_set("epoch", (epoch, epochs))
    
    def on_epoch_end(self, epoch: int, epochs: int, metrics: Dict[str, float]):
        self.topic_publish("metrics", {"layer_id": "model", "metric": metrics, "name": "training"})
        
    
    def main(self):
        autoencoder = self.trial_config.task  == 'autoencoder'
        self.trainer.train(lr=self.trial_config.learning_rate, epoch_total=self.trial_config.epochs, autoencoder=autoencoder)
        self.stop()
 
 
    def on_layer_forward(self, layer: LayerObserver, x_in: torch.Tensor, x_out: torch.Tensor):
        if self.compute_scores:
            self.on_layer_forward_scores(layer, x_in, x_out)
        else:
            self.on_layer_forward_activations(layer, x_in, x_out)
 
    def on_layer_forward_activations(self, layer: LayerObserver, x_in: torch.Tensor, x_out: torch.Tensor):
        #TODO: Should I send scores? should i use cuda timelines ? 
 
        data = {
            'layer_id': layer.layer_id,
            # .clone() is required, not optional: under torch.compile(mode=
            # "reduce-overhead") x_out can be a CUDA graph pool buffer that
            # gets overwritten in-place on the next forward pass, before the
            # subscriber on the other side of topic_publish has read it.
            # .detach() alone only cuts the autograd link -- it does not copy
            # storage.
            "activations": x_out.detach().clone()
        }
        self.topic_publish("activations", data)
        
    
 
    def on_layer_forward_scores(self, layer: LayerObserver, x_in: torch.Tensor, x_out: torch.Tensor):
        #TODO: Should I send scores? should i use cuda timelines ? 
 
        self.forward_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.forward_stream):
            scores = SpaceMath.activations_to_scores(
                x_out.detach(), layer.layer_type, layer.activation,
                self.trial_config.modular_space_active_q, cnn_locality_weight=0.5
            )
            ipc_event_cuda = self._acquire_ipc_slot(layer.layer_id)
            ipc_event_cuda.record()

        data = {
            'layer_id': layer.layer_id,
            'scores': scores,
            'ready': ipc_event_cuda,
        }
        self.topic_publish("scores", data)
    
    def _next_ipc_event(self, layer_id: str) -> cuda.Event:
        """
        Returns an interprocess CUDA event for this layer from a
        demand-grown, capped pool (created lazily on first use).
 
        This replaces minting a fresh torch.cuda.Event(interprocess=True) on
        every single forward call, which is what previously caused unbounded
        IPC handle growth (thousands of live handles per epoch across all
        layers). It also replaces a single reused-per-layer event, which is
        unsafe: record() can re-arm an event before a slower downstream
        consumer has finished reading the data tied to its previous record().
        See _IpcEventPool for the growth strategy, the query()-based reuse
        heuristic, and its caveats.
        """
        pool = self._ipc_event_pools.get(layer_id)
        if pool is None:
            pool = _IpcEventPool(max_size=self.ipc_event_pool_max_size)
            self._ipc_event_pools[layer_id] = pool
        return pool.next(layer_id)



    def _acquire_ipc_slot(self, layer_id: str) -> IpcEventCuda:
        """
        Returns a reusable paired producer/consumer event slot for this
        layer from a fixed-size pool (created lazily on first use).
 
        Replaces minting a fresh torch.cuda.Event(interprocess=True) on every
        forward call (unbounded IPC handle growth) and replaces reusing a
        single event per layer (unsafe: record() can re-arm an event before
        a slower consumer has finished with the data tied to its previous
        record()). See _IpcEventPool for the acquire/reuse/failsafe logic.
        """
        pool = self._ipc_event_pools.get(layer_id)
        if pool is None:
            pool = IpcEventPoolManaged(size=self.ipc_event_pool_size, stale_after_s=self.ipc_event_stale_after_s)
            self._ipc_event_pools[layer_id] = pool
        return pool.acquire(layer_id)
