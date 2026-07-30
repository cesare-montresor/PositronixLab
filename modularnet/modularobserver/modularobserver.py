from typing import Dict, List

from torch import nn
import torch

from modularnet.modularobserver.layerobserver import LayerCallback, WeightCallback, LayerObserver
from modularnet.modularobserver.modelmapper import Graph, ModelMapper
from modularnet.spacemath.common import is_layer, is_linear, is_conv
from modularnet.trial.trial_config import TrialConfig


# =====================================================================
# 4. ModularObserver (Observation & Metric Manager)
# =====================================================================

class ModularObserver:
    """Manages LayerObservers and metric collection, powered by ModelMapper."""
    def __init__(
        self, 
        config: TrialConfig, 
        model: nn.Module, 
        device: torch.device = None,
    ):
        self.config = config
        self.device = device or getattr(model, 'device', torch.device("cpu"))
        self.model = model
        
        self.observers: List[LayerObserver] = []
        self.by_name: Dict[str, LayerObserver] = {}
        self.by_layer: Dict[nn.Module, LayerObserver] = {}
        
        self.latest_graph_dict = None

        self.mapper = ModelMapper(
            model=self.model,
            is_suitable_layer=self._is_suitable_layer,
            on_graph_changed=self._on_graph_discovered
        )
        
        self._graph_update_callback = None  # Optional callback for external graph updates

    def graph(self) -> Graph:
        return self.mapper.graph

    def _is_suitable_layer(self, module: nn.Module, name: str) -> bool:
        """Determines which modules are tracked as unique layers in the graph."""
        if len(list(module.parameters())) == 0:
            return False # only learnable layers
        # return is_layer(module)
        # return is_linear(module) or is_conv(module)        
        return is_linear(module) # todo: support all architectures

    def _on_graph_discovered(self, mapper: ModelMapper):
        """Called automatically when the mapper finds new structure."""
        self._sync_observers_with_mapper()
        if self._graph_update_callback is None: return
        self._graph_update_callback(self.mapper.graph)

    def _sync_observers_with_mapper(self):
        """Syncs the underlying topological nodes with metric-collecting LayerObservers."""
        for name, node in self.mapper.graph.nodes.items():
            if name not in self.by_name:
                module_ref = node.module_ref
                lo = LayerObserver(self, name, node)
                self.observers.append(lo)
                self.by_name[name] = lo
                self.by_layer[module_ref] = lo

    
    def map(self, dummy_data: torch.Tensor=None, trace_samples: int = 1) -> Dict[str, List]:
        """
        Runs a traced forward pass to accurately construct edges and
        discover functional activations, then returns the ready Cytoscape dictionary.

        Args:
            dummy_data: input tensor used to drive the traced forward pass.
            trace_samples: number of samples to use for tracing.
        """
        self.mapper.map(self.model, dummy_data, trace_samples=trace_samples)

    # =====================================================================
    # Callbacks Forwarding
    # =====================================================================
    
    def on_graph_update(self, callback):
        """Sets a callback to be called whenever the graph is updated."""
        self._graph_update_callback = callback

    def on_forward_layer(self, callback: LayerCallback = None):
        for observer in self.observers:
            observer.on_forward_layer(callback)
            
    def on_backward_layer(self, callback: LayerCallback = None):
        for observer in self.observers:
            observer.on_backward_layer(callback)

    def on_grad_weights(self, callback: WeightCallback = None):
        for observer in self.observers:
            observer.on_grad_weights(callback) 

    def clear(self):
        """Removes all hooks and clears the mapped observers."""
        self.on_forward_layer(None)
        self.on_backward_layer(None)
        self.on_grad_weights(None)
        self.observers = []
        self.by_name = {}
        self.by_layer = {}