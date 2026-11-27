from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from positronixlab.modularobserver.modelmapper import ModelGraph, ModelMapper, ModelNode
from positronixlab.pipeline import PipelineConfig
from positronixlab.spacemath.common import is_layer, is_linear

PreForwardCallback = Callable[..., Any]
ForwardCallback = Callable[..., Any]
WeightCallback = Callable[[nn.Module, torch.nn.Parameter, torch.Tensor], torch.Tensor | None]



# =====================================================================
# 4. ModularObserver (Observation & Metric Manager)
# =====================================================================

class ModularObserver:
    """Manages LayerObservers and metric collection, powered by ModelMapper."""
    def __init__(
        self, 
        config: PipelineConfig, 
        model: nn.Module, 
        device: torch.device | None = None,
        observable_types: tuple[type[nn.Module], ...] | None = None,
        target_layers: list[str] | None = None,
        exclude_patterns: tuple[str, ...] = (),
    ):
        self.config = config
        model_device = getattr(model, "device", None)
        if device is not None:
            self.device = device
        elif isinstance(model_device, torch.device):
            self.device = model_device
        else:
            first_param = next(model.parameters(), None)
            self.device = first_param.device if first_param is not None else torch.device("cpu")
        self.model = model
        self.observable_types = observable_types
        self.target_layers = set(target_layers) if target_layers is not None else None
        self.exclude_patterns = tuple(exclude_patterns)
        
        self.nodes_list: list[ModelNode] = []
        self.nodes_by_name: dict[str, ModelNode] = {}
        self.nodes_by_layer: dict[nn.Module, ModelNode] = {}
        
        self.latest_graph_dict = None

        self.mapper = ModelMapper(
            model=self.model,
            is_suitable_layer=self._is_suitable_layer,
            on_graph_changed=self._on_graph_discovered
        )
        self.hooks: dict[str, RemovableHandle | list[RemovableHandle]] = {}
        self._graph_update_callback = None  # Optional callback for external graph updates

    def graph(self) -> ModelGraph:
        return self.mapper.graph
    
    def _is_suitable_layer(self, module: nn.Module, name: str) -> bool:
        """Determines which modules are tracked as unique layers in the graph."""
        if self.target_layers is not None and name in self.target_layers:
            return True
        if self.observable_types is not None and isinstance(module, self.observable_types):
            return True
        return is_layer(module)
        
    def _is_observable_layer(self, module: nn.Module, name: str) -> bool:
        """Determines which modules are tracked as unique layers in the graph."""
        if len(list(module.parameters())) == 0:
            return False # only learnable layers
        if self.exclude_patterns and any(pat in name for pat in self.exclude_patterns):
            return False
        if self.target_layers is not None:
            return name in self.target_layers
        if self.observable_types is not None:
            return isinstance(module, self.observable_types)
        return is_linear(module)


    def _on_graph_discovered(self, mapper: ModelMapper):
        """Called automatically when the mapper finds new structure."""
        self._sync_observers_with_mapper()
        if self._graph_update_callback is None:
            return
        self._graph_update_callback(self.mapper.graph)

    def _sync_observers_with_mapper(self):
        """Syncs the underlying topological nodes with metric-collecting LayerObservers."""
        for name, node in self.mapper.graph.nodes.items():
            if not self._is_observable_layer(node.module_ref, name):
                continue
            if node.name not in self.nodes_by_name:
                self.nodes_list.append(node)
                self.nodes_by_name[node.name] = node
                self.nodes_by_layer[node.module_ref] = node

    def layer_names(self) -> list[str]:
        """Returns a list of all observable layer names."""
        return list(self.nodes_by_name.keys())

    def by_name(self, name: str) -> ModelNode | None:
        """Returns the LayerObserver for a given layer name."""
        return self.nodes_by_name.get(name, None)

    def by_layer(self, layer: nn.Module) -> ModelNode | None:
        """Returns the LayerObserver for a given layer module."""
        return self.nodes_by_layer.get(layer, None)
    
    def map(self, dummy_data: torch.Tensor | None = None, trace_samples: int = 1) -> None:
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



    def on_grad_weights(self, callback: WeightCallback | None = None):
        for node in self.nodes_list:
            hook_name = f"{node.name}_weights"
            module_ref = node.module_ref
            if hook_name in self.hooks:
                existing = self.hooks[hook_name]
                if isinstance(existing, list):
                    for hook in existing:
                        hook.remove()
                else:
                    existing.remove()
                del self.hooks[hook_name]
            if callback is not None:
                cb = callback
                hooks = [
                    param.register_hook(lambda grad, m=module_ref, p=param, fn=cb: fn(m, p, grad))
                    for param in module_ref.parameters()
                ]
                self.hooks[hook_name] = hooks

    def on_pre_forward_layer(self, callback: PreForwardCallback | None = None):
        for node in self.nodes_list:
            hook_name = f"{node.name}_pre_forward"
            module_ref = node.module_ref
            if hook_name in self.hooks:
                existing = self.hooks[hook_name]
                if isinstance(existing, RemovableHandle):
                    existing.remove()
                del self.hooks[hook_name]
            if callback is not None:
                hook = module_ref.register_forward_pre_hook(callback)
                self.hooks[hook_name] = hook

    def on_forward_layer(self, callback: ForwardCallback | None = None):
        for node in self.nodes_list:
            module_ref = node.module_ref
            hook_name = f"{node.name}_forward"
            if hook_name in self.hooks:
                existing = self.hooks[hook_name]
                if isinstance(existing, RemovableHandle):
                    existing.remove()
                del self.hooks[hook_name]
            if callback is not None:
                hook = module_ref.register_forward_hook(callback)
                self.hooks[hook_name] = hook

    def on_backward_layer(self, callback: ForwardCallback | None = None):
        for node in self.nodes_list:
            module_ref = node.module_ref
            hook_name = f"{node.name}_backward"
            if hook_name in self.hooks:
                existing = self.hooks[hook_name]
                if isinstance(existing, RemovableHandle):
                    existing.remove()
                del self.hooks[hook_name]
            if callback is not None:
                hook = module_ref.register_full_backward_hook(callback)
                self.hooks[hook_name] = hook

    
    def clear(self):
        """Removes all hooks and clears the mapped observers."""
        self.on_pre_forward_layer(None)
        self.on_forward_layer(None)
        self.on_backward_layer(None)
        self.on_grad_weights(None)
        self.nodes_list = []
        self.nodes_by_name = {}
        self.nodes_by_layer = {}