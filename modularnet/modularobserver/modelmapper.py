import torch
from torch import nn
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode
from typing import Callable, Dict, List, Set, Optional, Any, Tuple

# Assume these are imported from your existing project structure
from modularnet.spacemath.common import is_activation, is_layer
# =====================================================================
# 1. Metric Extractor Utility
# =====================================================================

def extract_layer_io_metrics(module: nn.Module) -> Dict[str, Any]:
    """Extracts input/output dimensions and the domain-specific unit type."""
    metrics = {'in_units': None, 'out_units': None, 'unit_type': 'unknown', 'param_ids': set()}
    
    # Anchor parameter IDs for gradient mapping later
    metrics['param_ids'] = {id(p) for p in module.parameters(recurse=False) if p.requires_grad}

    if isinstance(module, nn.Linear):
        metrics.update({'in_units': module.in_features, 'out_units': module.out_features, 'unit_type': 'neurons'})
    elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d)):
        metrics.update({'in_units': module.in_channels, 'out_units': module.out_channels, 'unit_type': 'feature_maps'})
    elif isinstance(module, nn.Embedding):
        metrics.update({'in_units': module.num_embeddings, 'out_units': module.embedding_dim, 'unit_type': 'embeddings'})
    elif isinstance(module, (nn.RNNBase, nn.RNN, nn.LSTM, nn.GRU)):
        metrics.update({'in_units': module.input_size, 'out_units': module.hidden_size, 'unit_type': 'hidden_states'})
    elif isinstance(module, nn.MultiheadAttention):
        metrics.update({'in_units': module.embed_dim, 'out_units': module.embed_dim, 'unit_type': 'attention_dim'})
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
        if hasattr(module, 'num_features'):
            metrics.update({'in_units': module.num_features, 'out_units': module.num_features, 'unit_type': 'normalized_channels'})
        elif hasattr(module, 'normalized_shape'):
            shape = module.normalized_shape[0] if isinstance(module.normalized_shape, (tuple, list)) else module.normalized_shape
            metrics.update({'in_units': shape, 'out_units': shape, 'unit_type': 'normalized_features'})
            
    return metrics


# =====================================================================
# 2. Functional Activation Interceptor
# =====================================================================

class _MapperInterceptor(TorchFunctionMode):
    """Intercepts functional activations (F.relu, etc.) to link them to the graph."""
    def __init__(self, mapper: 'ModelMapper'):
        super().__init__()
        self.mapper = mapper
        self.target_ops = {F.relu, F.gelu, F.silu, F.softmax, F.log_softmax, torch.relu, torch.sigmoid, torch.tanh}

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        output = func(*args, **kwargs)
        if func in self.target_ops:
            self.mapper._record_functional(func.__name__)
        return output


# =====================================================================
# 3. ModelMapper (Topology Engine)
# =====================================================================

class ModelMapper:
    """Handles PyTorch topology, metric mapping, and dynamic tracing."""
    def __init__(
        self, 
        model: nn.Module, 
        dynamic_mode: str = "no", # "no", "once", "yes"
        is_suitable_layer: Optional[Callable[[nn.Module, str], bool]] = None,
        on_graph_changed: Optional[Callable[['ModelMapper'], None]] = None
    ):
        self.model = model
        self.dynamic_mode = dynamic_mode
        self.is_suitable_layer = is_suitable_layer or (lambda m, n: True)
        self.on_graph_changed = on_graph_changed
        
        # Graph State
        self.nodes: Dict[str, Dict[str, Any]] = {}  # name -> node_metadata
        self.edges: Set[Tuple[str, str]] = set()    # (source_name, target_name)
        
        # Tracing State
        self.sequence: List[str] = []
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._interceptor = _MapperInterceptor(self)
        self._has_run_dynamic = False
        self._graph_changed_in_pass = False

    
    def to_dict(self) -> Dict[str, List]:
        """Serializes the graph for Dash Cytoscape, including depths & capacities."""
        elements = []
        for name, data in self.nodes.items():
            acts = ", ".join(data['functional_acts'])
            label_parts = [f"{name}\n({data['class_name']})"]
            
            if data['in_units'] is not None and data['out_units'] is not None:
                label_parts.append(f"in: {data['in_units']} | out: {data['out_units']} {data['unit_type']}")
            if acts: 
                label_parts.append(f"[{acts}]")
            
            elements.append({
                'data': {
                    'id': name, 
                    'label': "\n".join(label_parts), 
                    'class_name': data['class_name'],
                    'depth': data.get('depth', 0)
                },
                'classes': f"node-{data['class_name'].lower()} node-module"
            })
            
        for src, tgt in self.edges:
            elements.append({'data': {'source': src, 'target': tgt}})
            
        return {'elements': elements}


    def static_mapping(self):
        """Builds initial nodes from the model definition with base integer depths."""
        self._graph_changed_in_pass = False
        static_depth_counter = 0

        for name, module in self.model.named_modules():
            if not list(module.children()) and self.is_suitable_layer(module, name):
                self._add_node(name, module, default_depth=static_depth_counter)
                static_depth_counter += 1

        if self._graph_changed_in_pass and self.on_graph_changed:
            self.on_graph_changed(self)

    def _add_node(self, name: str, module: nn.Module, default_depth: Optional[int] = None):
        if name not in self.nodes:
            io_metrics = extract_layer_io_metrics(module)
            self.nodes[name] = {
                'id': name,
                'class_name': type(module).__name__,
                'module_ref': module,
                'functional_acts': [],
                'in_units': io_metrics['in_units'],
                'out_units': io_metrics['out_units'],
                'unit_type': io_metrics['unit_type'],
                'param_ids': io_metrics['param_ids'],
                'depth': default_depth  # None if discovered dynamically
            }
            self._graph_changed_in_pass = True

    def _add_edge(self, source: str, target: str):
        if source == target:
            # Guard against self-loops caused by a module firing its forward
            # pre-hook more than once in the same pass (weight tying, RNN
            # unrolling, a module invoked twice in the same graph, etc).
            # A self-loop here would also corrupt _assign_dynamic_depths,
            # which derives a node's depth from its neighbors.
            return
        if (source, target) not in self.edges:
            self.edges.add((source, target))
            self._graph_changed_in_pass = True

    def _record_functional(self, act_name: str):
        if not self.sequence: return
        last_node = self.sequence[-1]
        
        if last_node in self.nodes and act_name not in self.nodes[last_node]['functional_acts']:
            self.nodes[last_node]['functional_acts'].append(act_name)
            self._graph_changed_in_pass = True

    def _assign_dynamic_depths(self):
        """
        Calculates depths for newly discovered nodes:
        Forward: parent + 1. Backward: child - 1. Final: Normalizes starting at 0.
        """
        if not self.sequence: return

        # 1. Forward Pass: Assign (prev + 1)
        for i in range(1, len(self.sequence)):
            curr_id, prev_id = self.sequence[i], self.sequence[i-1]
            if curr_id in self.nodes and self.nodes[curr_id].get('depth') is None:
                if prev_id in self.nodes and self.nodes[prev_id].get('depth') is not None:
                    self.nodes[curr_id]['depth'] = self.nodes[prev_id]['depth'] + 1

        # 2. Backward Pass: Assign (next - 1)
        for i in range(len(self.sequence) - 2, -1, -1):
            curr_id, next_id = self.sequence[i], self.sequence[i+1]
            if curr_id in self.nodes and self.nodes[curr_id].get('depth') is None:
                if next_id in self.nodes and self.nodes[next_id].get('depth') is not None:
                    self.nodes[curr_id]['depth'] = self.nodes[next_id]['depth'] - 1

        # 3. Fallback for completely isolated/unanchored nodes
        current_depth = 0
        for node_id in self.sequence:
            if node_id in self.nodes and self.nodes[node_id].get('depth') is None:
                self.nodes[node_id]['depth'] = current_depth
                current_depth += 1

        # 4. Offset Normalization: shift everything so min depth is exactly 0
        all_depths = [n['depth'] for n in self.nodes.values() if n.get('depth') is not None]
        if all_depths:
            offset = -min(all_depths)
            for node in self.nodes.values():
                if node.get('depth') is not None:
                    node['depth'] = int(node['depth'] + offset)

    def trace_context(self):
        """Context manager to wrap the forward pass for dynamic tracing."""
        class _DummyContext:
            def __enter__(self): pass
            def __exit__(self, *args): pass

        if self.dynamic_mode == "no" or (self.dynamic_mode == "once" and self._has_run_dynamic):
            return _DummyContext()

        return self._ActiveTraceContext(self)

    class _ActiveTraceContext:
        def __init__(self, mapper: 'ModelMapper'):
            self.mapper = mapper

        def __enter__(self):
            self.mapper.sequence.clear()
            self.mapper._graph_changed_in_pass = False
            
            for name, module in self.mapper.model.named_modules():
                if not list(module.children()) and self.mapper.is_suitable_layer(module, name):
                    def make_hook(mod_name, mod_ref):
                        def hook(m, inp):
                            self.mapper._add_node(mod_name, mod_ref, default_depth=None)
                            # Only wire an edge from the previous distinct node.
                            # If this module fired last in the sequence too
                            # (e.g. called twice in a row within one forward
                            # pass), skip the edge instead of creating a
                            # self-loop; the repeated call is still recorded
                            # in `sequence` for depth-assignment purposes.
                            if self.mapper.sequence and self.mapper.sequence[-1] != mod_name:
                                self.mapper._add_edge(self.mapper.sequence[-1], mod_name)
                            self.mapper.sequence.append(mod_name)
                        return hook
                    self.mapper._hooks.append(module.register_forward_pre_hook(make_hook(name, module)))
            
            self.mapper._interceptor.__enter__()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.mapper._interceptor.__exit__(exc_type, exc_val, exc_tb)
            for h in self.mapper._hooks: h.remove()
            self.mapper._hooks.clear()
            
            self.mapper._assign_dynamic_depths()
            self.mapper._has_run_dynamic = True

            if self.mapper._graph_changed_in_pass and self.mapper.on_graph_changed:
                self.mapper.on_graph_changed(self.mapper)