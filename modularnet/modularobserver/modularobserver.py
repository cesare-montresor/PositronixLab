from typing import Dict, List

from torch import nn
import torch

from modularnet.modularobserver.layerobserver import LayerCallback, WeightCallback, LayerObserver
from modularnet.modularobserver.modelmapper import ModelMapper
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

    def _is_suitable_layer(self, module: nn.Module, name: str) -> bool:
        """Determines which modules are tracked as unique layers in the graph."""
        # return is_layer(module)
        # return is_linear(module) or is_conv(module)        
        return is_linear(module) # todo: support all architectures

    def _on_graph_discovered(self, mapper: ModelMapper):
        """Called automatically when the mapper finds new structure."""
        self._sync_observers_with_mapper()
        self.latest_graph_dict = mapper.to_dict()
        if self._graph_update_callback is None: return
        self._graph_update_callback(self.latest_graph_dict)

    def _sync_observers_with_mapper(self):
        """Syncs the underlying topological nodes with metric-collecting LayerObservers."""
        for name, node_data in self.mapper.nodes.items():
            if name not in self.by_name:
                module_ref = node_data['module_ref']
                depth = node_data.get('depth', 0)
                num_units = node_data['out_units']
                
                layer_id = str(len(self.observers))
                lo = LayerObserver(self, module_ref, layer_id, depth, num_units)
                lo.name = name
                lo.param_ids = node_data['param_ids']
                
                self.observers.append(lo)
                self.by_name[name] = lo
                self.by_layer[module_ref] = lo
            else:
                # Keep depth perfectly synced in case dynamic tracing adjusted it
                self.by_name[name].depth = node_data.get('depth', self.by_name[name].depth)
            
            # Sync functional activations
            lo = self.by_name[name]
            for act in node_data['functional_acts']:
                if hasattr(lo, 'set_functional_activation'):
                    lo.set_functional_activation(act)

    def static_map(self, ) -> Dict[str, List]:
        self.mapper.static_mapping()
        return self.latest_graph_dict

    def runtime_map(self, dummy_data: torch.Tensor, retrace: bool = False) -> Dict[str, List]:
        """
        Runs a traced forward pass to accurately construct edges and
        discover functional activations, then returns the ready Cytoscape dictionary.

        Args:
            dummy_data: input tensor used to drive the traced forward pass.
            retrace: if False (default), the mapper only traces dynamically on
                the first call ("once") and subsequent calls are cheap no-ops
                that reuse the previously discovered graph. If True, the
                mapper re-traces on every call ("yes") -- use this only if
                the model's control flow can change the graph shape between
                calls (e.g. data-dependent branching), since re-tracing repeats
                the hook + TorchFunctionMode overhead every time.
        """
        self.mapper.dynamic_mode = "yes" if retrace else "once"
        with self.mapper.trace_context():
            _ = self.model(dummy_data)
        self.mapper.dynamic_mode = "no"

        return self.latest_graph_dict

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