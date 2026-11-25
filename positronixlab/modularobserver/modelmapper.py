import torch
import warnings
from torch import nn
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode
from dataclasses import dataclass
from typing import Callable, Dict, List, Set, Optional, Any, Union

# Assume these are imported from your existing project structure
from positronixlab.spacemath.spacemath import LayerType


# =====================================================================
# 0. Trace budget sentinel
# =====================================================================

TRACE_ALWAYS = -1  # dynamic tracing never deactivates on its own


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
# 2. Graph model: Node / Edge / Graph
# =====================================================================

@dataclass
class ModelNode:
    name: str
    class_name: str
    module_ref: nn.Module
    layer_type: Any
    param_ids: Set[int]
    in_units: Optional[int] = None
    out_units: Optional[int] = None
    unit_type: str = 'unknown'
    activation_fn: Optional[Callable] = None
    depth: Optional[int] = None

    # Fields describing what the module currently *is* -- always trust the
    # freshly-extracted value over the stored one. module_ref is compared by
    # identity separately below since nn.Module doesn't define __eq__.
    _STRUCTURAL_FIELDS = ('class_name', 'layer_type', 'in_units', 'out_units', 'unit_type', 'param_ids')

    def update_from(self, new: 'ModelNode') -> bool:
        """
        Merges freshly-extracted metadata from `new` into self, in place.
        Returns True if anything actually changed.

        Structural fields are overwritten outright. `activation_fn` and
        `depth` are discovery fields populated incrementally by dynamic
        tracing, so they're only overwritten when `new` actually provides a
        value -- a re-scan that hasn't (yet) rediscovered them shouldn't
        wipe out what tracing already found.
        """
        changed = False

        if self.module_ref is not new.module_ref:
            self.module_ref = new.module_ref
            changed = True

        for name in self._STRUCTURAL_FIELDS:
            old_val, new_val = getattr(self, name), getattr(new, name)
            if old_val != new_val and new_val is not None:
                setattr(self, name, new_val)
                changed = True

        if new.activation_fn is not None and new.activation_fn != self.activation_fn:
            self.activation_fn = new.activation_fn
            changed = True

        if new.depth is not None and new.depth != self.depth:
            self.depth = new.depth
            changed = True

        return changed

    def to_dict(self) -> Dict[str, Any]:
        """Serializes full node metadata (module_ref/activation_fn stringified)."""
        return {
            'id': self.name,
            'class_name': self.class_name,
            'module_ref': str(self.module_ref),
            'layer_type': self.layer_type,
            'activation_fn': str(self.activation_fn.__name__) if self.activation_fn is not None else '',
            'in_units': self.in_units,
            'out_units': self.out_units,
            'unit_type': self.unit_type,
            'param_ids': list(self.param_ids), 
            'depth': self.depth,
        }

@dataclass(frozen=True)
class ModelEdge:
    source: str
    target: str

    @property
    def name(self) -> str:
        return f"{self.source}-{self.target}"

    def to_dict(self) -> Dict[str, Any]:
        return {'source': self.source, 'target': self.target}

    def __key(self):
        return (self.source, self.target)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, ModelEdge):
            return self.__key() == other.__key()
        return NotImplemented

class ModelGraph:
    """Holds nodes/edges and owns their serialization."""

    def __init__(self):
        self.nodes: Dict[str, ModelNode] = {}
        self.edges: Set[ModelEdge] = set()

    def add_node(self, node: ModelNode) -> bool:
        """Adds a new node, or refreshes an existing one in place via
        Node.update_from(). Returns True if the node is new or changed
        (i.e. the graph needs a redraw)."""
        existing = self.nodes.get(node.name)
        if existing is None:
            self.nodes[node.name] = node
            return True
        return existing.update_from(node)

    def add_edge(self, source: str, target: str) -> bool:
        """Returns True if the edge was newly added. Rejects self-loops."""
        if source == target:
            # Guard against self-loops caused by a module firing its forward
            # pre-hook more than once in the same pass (weight tying, RNN
            # unrolling, a module invoked twice in the same graph, etc).
            # A self-loop here would also corrupt depth assignment, which
            # derives a node's depth from its neighbors.
            return False
        edge = ModelEdge(source, target)
        if edge in self.edges:
            return False
        self.edges.add(edge)
        return True


class GraphFormatter:


    @classmethod
    def to_dict(cls, graph: ModelGraph) -> Dict[str, List|Dict[str, Any]]:
        """Returns a dict with 'nodes' and 'edges' lists for the graph."""
        return {
            "nodes": cls.nodes_dict(graph),
            "edges": cls.edges_list(graph)
        }
    
    
    @classmethod
    def to_cytoscape(cls, graph: ModelGraph) -> Dict[str, List]:
        """Serializes the full graph for Dash Cytoscape."""
        elements = [cls.node_to_cytoscape(node) for node in graph.nodes.values()]
        elements += [cls.edge_to_cytoscape(edge) for edge in graph.edges]
        return {'elements': elements}

    @classmethod
    def node_to_cytoscape(cls, node: ModelNode) -> Dict[str, Any]:
        """Serializes a single node for Dash Cytoscape."""
        data = node.to_dict()
        label_keys = ['id', 'class_name', 'unit_type', 'in_units', 'out_units', 'activation_fn']
        label_parts = [f"{key}: {data[key]}" for key in label_keys if data.get(key) is not None]
        data['label'] = "\n".join(label_parts)                
        classes = f"node-{data.get('class_name', 'class-none').lower()} node-module"
        
        return { 'data': data, 'classes': classes }

    @classmethod
    def edge_to_cytoscape(cls, edge: ModelEdge) -> Dict[str, Any]:
        """Serializes a single edge for Dash Cytoscape."""
        return {'data': edge.to_dict()}

    @classmethod
    def nodes_dict(cls, graph: ModelGraph) -> Dict[str, Any]:
        """Returns all node metadata, keyed by node id."""
        return {name: node.to_dict() for name, node in graph.nodes.items()}

    @classmethod
    def nodes_list(cls, graph: ModelGraph) -> List[Dict[str, Any]]:
        """Returns all node metadata, keyed by node id."""
        return [node.to_dict() for node in graph.nodes.values()]

    @classmethod
    def edges_list(cls, graph: ModelGraph) -> List[Dict[str, Any]]:
        """Returns all edge metadata, keyed by edge."""
        return [edge.to_dict() for edge in graph.edges]

    @classmethod
    def edges_dict(cls, graph: ModelGraph) -> Dict[str, Any]:
        """Returns all edge metadata, keyed by edge."""
        return {edge.name: edge.to_dict() for edge in graph.edges}


# =====================================================================
# 3. Functional Activation Interceptor
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
            self.mapper._record_functional(func)
        return output


# =====================================================================
# 4. ModelMapper (Topology Engine)
# =====================================================================

class ModelMapper:
    """Handles PyTorch topology, metric mapping, and dynamic tracing."""

    @staticmethod
    def _validate_trace_budget(value: Union[int, float]) -> Union[int, float]:
        if isinstance(value, float):
            if not (0.0 <= value <= 1.0):
                warnings.warn(
                    f"ModelMapper: trace_samples={value!r} is a float outside [0, 1]; "
                    "floats are treated as a per-call trace probability, so this will "
                    "behave like a clamped rate. Use an int for count-based budgets.",
                    RuntimeWarning,
                    stacklevel=3,
                )
        return value

    def nodes(self) -> Dict[str, ModelNode]:
        """Returns all nodes' metadata as a dictionary keyed by node id."""
        return GraphFormatter.nodes_dict(self.graph)

    def edges(self) -> List[ModelEdge]:
        """Returns all edges as a list of Edge objects."""
        return GraphFormatter.edges_list(self.graph)

    def __init__(
        self,
        model: nn.Module,
        trace_samples: Union[int, float] = 0,
        is_suitable_layer: Optional[Callable[[nn.Module, str], bool]] = None,
        on_graph_changed: Optional[Callable[['ModelMapper'], None]] = None,
    ):
        self.model = model
        # How eagerly upcoming calls should trace:
        #   0            -> tracing off, map()/trace_context() are no-ops
        #   N (int)      -> trace on the next N calls, then auto-deactivate (0)
        #   TRACE_ALWAYS (-1) -> trace on every call, never deactivates
        #   f (float in [0, 1]) -> per-call probability of tracing; a
        #       standing rate, not consumed/decremented like the int forms
        self.trace_budget = self._validate_trace_budget(trace_samples)
        self.is_suitable_layer = is_suitable_layer or (lambda m, n: True)
        self.on_graph_changed = on_graph_changed

        # Graph State
        self.graph = ModelGraph()

        # Tracing State
        self.sequence: List[str] = []
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._interceptor = _MapperInterceptor(self)
        self._graph_changed_in_pass = False
        self._has_run_static = False


    def map(
        self,
        model: nn.Module,
        data: Optional[torch.Tensor] = None,
        trace_samples: Optional[int] = None,
    ) -> Dict[str, List]:
        """
        Single entry point for keeping the graph in sync with `model`.

        Always ensures the one-time static pass has run (the bare leaf-layer
        list). If `data` is given and there's trace budget left, also runs a
        traced forward pass on `data` to discover edges and functional
        activations, consuming one unit of budget.

        Args:
            model: the model to map/trace. Becomes self.model. If a traced
                pass runs, training-mode is snapshotted, forced to eval()
                for the duration -- so the pass doesn't perturb BatchNorm
                running stats or fire dropout -- and restored afterward
                regardless of outcome.
            data: sample input for a traced forward pass. If None, or if the
                trace budget is currently 0, no forward pass happens at all
                -- this is a cheap static-only call.
            trace_samples: if given, (re)arms the trace budget before this
                call:
                    0            -> tracing off, static-only from now on
                    N (int)      -> trace on this call and the next N-1
                    TRACE_ALWAYS -> always trace, never deactivates
                    f (float, 0 < f <= 1) -> probabilistic: each call
                        independently traces with probability f. This is a
                        standing rate, not consumed/decremented -- unlike
                        the int forms it doesn't wind down on its own.
                If omitted, whatever budget is currently set (from
                construction or a previous call) is used as-is.
        """
        self.model = model
        self._static_mapping()  # one-time; no-op if it already ran

        if trace_samples is not None:
            self.trace_budget = self._validate_trace_budget(trace_samples)

        if data is None:
            return self.graph
        # Only skip the forward pass entirely when we deterministically know
        # it won't trace. A float budget can't be resolved without drawing,
        # and that draw happens inside trace_context() -- so for float
        # budgets we always run the pass and let trace_context() decide.
        if isinstance(self.trace_budget, int) and self.trace_budget == 0:
            return self.graph

        was_training = model.training
        model_device = getattr(model, "device", None)
        if model_device is None:
            first_param = next(model.parameters(), None)
            model_device = first_param.device if first_param is not None else torch.device("cpu")
        try:
            model.eval()
            with torch.no_grad(), self.trace_context():
                _ = model(data.to(device=model_device))
        finally:
            model.train(was_training)

        return self.graph

    def _static_mapping(self):
        """One-time init gate, meant to run exactly once (from __init__).
        This is the bare minimum representation of the model -- the flat
        list of leaf layers -- used as-is when the trace budget is 0.
        Further structural changes should go through map(), not this."""
        if self._has_run_static:
            return
        self._has_run_static = True
        self._update_static_nodes()

    def _update_static_nodes(self):
        """Shared walker behind both _static_mapping() and static_map():
        adds/upserts a node for every suitable leaf module, in named_modules()
        order, then refreshes the cached dict and fires on_graph_changed if
        anything actually changed."""
        self._graph_changed_in_pass = False
        depth_counter = 0

        for name, module in self.model.named_modules():
            if not list(module.children()) and self.is_suitable_layer(module, name):
                self._add_node(name, module, default_depth=depth_counter)
                depth_counter += 1

        if self._graph_changed_in_pass and self.on_graph_changed:
            self.on_graph_changed(self)

    def _add_node(self, name: str, module: nn.Module, default_depth: Optional[int] = None):
        io_metrics = extract_layer_io_metrics(module)
        node = ModelNode(
            name=name,
            class_name=type(module).__name__,
            module_ref=module,
            layer_type=LayerType.get_layer_type(module),
            param_ids=io_metrics['param_ids'],
            in_units=io_metrics['in_units'],
            out_units=io_metrics['out_units'],
            unit_type=io_metrics['unit_type'],
            depth=default_depth,  # None if discovered dynamically, or unknown on a re-scan
        )
        if self.graph.add_node(node):
            self._graph_changed_in_pass = True

    def _add_edge(self, source: str, target: str):
        if self.graph.add_edge(source, target):
            self._graph_changed_in_pass = True

    def _record_functional(self, act_fn: Callable):
        if not self.sequence:
            return
        last_node = self.graph.nodes.get(self.sequence[-1])

        if last_node is not None and last_node.activation_fn is None:
            last_node.activation_fn = act_fn
            self._graph_changed_in_pass = True

    def _assign_dynamic_depths(self):
        """
        Calculates depths for newly discovered nodes:
        Forward: parent + 1. Backward: child - 1. Final: Normalizes starting at 0.
        """
        if not self.sequence:
            return

        nodes = self.graph.nodes

        # 1. Forward Pass: Assign (prev + 1)
        for i in range(1, len(self.sequence)):
            curr_id, prev_id = self.sequence[i], self.sequence[i - 1]
            if curr_id in nodes and nodes[curr_id].depth is None:
                if prev_id in nodes and nodes[prev_id].depth is not None:
                    nodes[curr_id].depth = nodes[prev_id].depth + 1

        # 2. Backward Pass: Assign (next - 1)
        for i in range(len(self.sequence) - 2, -1, -1):
            curr_id, next_id = self.sequence[i], self.sequence[i + 1]
            if curr_id in nodes and nodes[curr_id].depth is None:
                if next_id in nodes and nodes[next_id].depth is not None:
                    nodes[curr_id].depth = nodes[next_id].depth - 1

        # 3. Fallback for completely isolated/unanchored nodes
        current_depth = 0
        for node_id in self.sequence:
            if node_id in nodes and nodes[node_id].depth is None:
                nodes[node_id].depth = current_depth
                current_depth += 1

        # 4. Offset Normalization: shift everything so min depth is exactly 0
        all_depths = [n.depth for n in nodes.values() if n.depth is not None]
        if all_depths:
            offset = -min(all_depths)
            for node in nodes.values():
                if node.depth is not None:
                    node.depth = int(node.depth + offset)

    def trace_context(self):
        """Context manager to wrap the forward pass for dynamic tracing."""
        class _DummyContext:
            def __enter__(self): pass
            def __exit__(self, *args): pass

        if self.trace_budget == 0:
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
            for h in self.mapper._hooks:
                h.remove()
            self.mapper._hooks.clear()

            if exc_type is None and not self.mapper.sequence:
                warnings.warn(
                    f"ModelMapper: traced forward pass on {type(self.mapper.model).__name__} "
                    "recorded no layer calls, so no edges or functional activations could be "
                    "discovered. This usually means no leaf module passed is_suitable_layer(), "
                    "or the model's forward() doesn't route through its registered submodules "
                    "(e.g. it operates on raw tensors/functions instead of nn.Module children). "
                    "Dynamic tracing will have no effect on the graph beyond the static mapping.",
                    RuntimeWarning,
                    stacklevel=2,
                )

            self.mapper._assign_dynamic_depths()
            if self.mapper.trace_budget != TRACE_ALWAYS:
                self.mapper.trace_budget = max(0, self.mapper.trace_budget - 1)

            if self.mapper._graph_changed_in_pass and self.mapper.on_graph_changed:
                self.mapper.on_graph_changed(self.mapper)