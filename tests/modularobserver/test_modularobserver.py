from __future__ import annotations

import functools
import sys
import types
from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from positronixlab.modularobserver.layerobserver import LayerObserver
from positronixlab.modularobserver.modelmapper import (
    TRACE_ALWAYS,
    GraphFormatter,
    ModelEdge,
    ModelGraph,
    ModelMapper,
    ModelNode,
    extract_layer_io_metrics,
)
from positronixlab.spacemath.common import is_layer, is_linear


def _ensure_parallelazy_stub() -> None:
    """Allow importing `ModularObserver` in sandboxed/offline envs without `parallelazy`."""
    try:
        import parallelazy  # noqa: F401
        from parallelazy.plugin_data import ManagerOptions  # noqa: F401
        return
    except ImportError:
        pass

    @dataclass
    class _StubManagerOptions:
        work_dir: str | None = None
        log_dir: str | None = None
        display_logs: bool = True
        write_logs: bool = True

    pkg = types.ModuleType("parallelazy")
    pkg.PluginManager = object  # type: ignore[attr-defined]
    pkg.is_debugpy_attached = lambda: False  # type: ignore[attr-defined]

    plugin_data = types.ModuleType("parallelazy.plugin_data")
    plugin_data.ManagerOptions = _StubManagerOptions  # type: ignore[attr-defined]
    pkg.plugin_data = plugin_data  # type: ignore[attr-defined]

    sys.modules["parallelazy"] = pkg
    sys.modules["parallelazy.plugin_data"] = plugin_data


_ensure_parallelazy_stub()
from positronixlab.modularobserver.modularobserver import ModularObserver  # noqa: E402

# ---------------------------------------------------------------------------
# Test Models
# ---------------------------------------------------------------------------

class DeviceMLP(nn.Module):
    """MLP exposing `.device` (matches MetaModel/BaseModel contract)."""

    def __init__(self, in_dim: int = 8, hidden: int = 12, out_dim: int = 4):
        super().__init__()
        self.device = torch.device("cpu")
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)


class HybridNet(nn.Module):
    """Mixed Conv + Linear + Identity model to test filtering & metrics."""

    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
        self.conv = nn.Conv2d(3, 6, kernel_size=3, padding=1)
        self.ident = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(6, 10)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv(x))
        x = self.ident(x)
        x = self.pool(x).flatten(1)
        x = F.gelu(self.fc1(x))
        return self.fc2(x)


class RepeatedLayerNet(nn.Module):
    """Calls the same linear module twice consecutively to test self-loop rejection."""

    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
        self.shared = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.shared(x))
        x = F.relu(self.shared(x))
        return self.head(x)


# ---------------------------------------------------------------------------
# 1. extract_layer_io_metrics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "module, expected_in, expected_out, expected_type",
    [
        (nn.Linear(8, 16), 8, 16, "neurons"),
        (nn.Conv1d(3, 6, 3), 3, 6, "feature_maps"),
        (nn.Conv2d(3, 8, 3), 3, 8, "feature_maps"),
        (nn.Conv3d(2, 4, 3), 2, 4, "feature_maps"),
        (nn.ConvTranspose2d(8, 4, 2), 8, 4, "feature_maps"),
        (nn.Embedding(100, 32), 100, 32, "embeddings"),
        (nn.LSTM(10, 20), 10, 20, "hidden_states"),
        (nn.GRU(10, 16), 10, 16, "hidden_states"),
        (nn.MultiheadAttention(16, 4), 16, 16, "attention_dim"),
        (nn.BatchNorm2d(12), 12, 12, "normalized_channels"),
        (nn.LayerNorm(16), 16, 16, "normalized_features"),
        (nn.ReLU(), None, None, "unknown"),
    ],
)
def test_extract_layer_io_metrics(module, expected_in, expected_out, expected_type):
    metrics = extract_layer_io_metrics(module)
    assert metrics["in_units"] == expected_in
    assert metrics["out_units"] == expected_out
    assert metrics["unit_type"] == expected_type
    expected_ids = {id(p) for p in module.parameters(recurse=False) if p.requires_grad}
    assert metrics["param_ids"] == expected_ids


def test_extract_layer_io_metrics_groupnorm_bug():
    metrics = extract_layer_io_metrics(nn.GroupNorm(2, 8))
    assert metrics["in_units"] == 8
    assert metrics["out_units"] == 8
    assert metrics["unit_type"] == "normalized_channels"


# ---------------------------------------------------------------------------
# 2. ModelNode / ModelEdge / ModelGraph / GraphFormatter
# ---------------------------------------------------------------------------

def test_model_graph_node_upsert_and_edge_dedup():
    graph = ModelGraph()
    fc = nn.Linear(4, 8)
    n1 = ModelNode(
        name="fc1",
        class_name="Linear",
        module_ref=fc,
        layer_type="linear",
        param_ids={id(p) for p in fc.parameters()},
        in_units=4,
        out_units=8,
        unit_type="neurons",
        activation_fn=F.relu,
        depth=0,
    )
    assert graph.add_node(n1) is True
    # Identical upsert -> no change
    assert graph.add_node(n1) is False

    # Re-scan with activation_fn=None and depth=None must preserve discovery fields
    n1_rescan = ModelNode(
        name="fc1",
        class_name="Linear",
        module_ref=fc,
        layer_type="linear",
        param_ids={id(p) for p in fc.parameters()},
        in_units=4,
        out_units=8,
        unit_type="neurons",
        activation_fn=None,
        depth=None,
    )
    assert graph.add_node(n1_rescan) is False
    assert graph.nodes["fc1"].activation_fn is F.relu
    assert graph.nodes["fc1"].depth == 0

    # Edges: self-loop rejected, duplicate rejected, valid accepted
    assert graph.add_edge("fc1", "fc1") is False
    assert graph.add_edge("fc1", "fc2") is True
    assert graph.add_edge("fc1", "fc2") is False
    assert len(graph.edges) == 1
    edge = next(iter(graph.edges))
    assert edge.name == "fc1-fc2"
    assert edge == ModelEdge("fc1", "fc2")


def test_graph_formatter_exports():
    model = DeviceMLP()
    mapper = ModelMapper(model, is_suitable_layer=lambda m, n: is_layer(m))
    graph = mapper.map(model, data=torch.randn(2, 8), trace_samples=1)

    as_dict = GraphFormatter.to_dict(graph)
    assert set(as_dict.keys()) == {"nodes", "edges"}
    assert set(as_dict["nodes"].keys()) == {"fc1", "fc2"}
    assert as_dict["edges"] == [{"source": "fc1", "target": "fc2"}]

    assert len(GraphFormatter.nodes_list(graph)) == 2
    assert "fc1-fc2" in GraphFormatter.edges_dict(graph)

    cyto = GraphFormatter.to_cytoscape(graph)
    assert "elements" in cyto
    elements = cyto["elements"]
    # 2 nodes + 1 edge
    assert len(elements) == 3
    node_elems = [e for e in elements if "classes" in e]
    edge_elems = [e for e in elements if "classes" not in e]
    assert len(node_elems) == 2
    assert len(edge_elems) == 1
    assert edge_elems[0]["data"] == {"source": "fc1", "target": "fc2"}
    for ne in node_elems:
        assert "node-linear" in ne["classes"]
        assert "node-module" in ne["classes"]
        assert "label" in ne["data"]
        assert "id: " in ne["data"]["label"]


def test_model_node_to_dict_callable_without_dunder_name_bug():
    fc = nn.Linear(4, 4)
    node = ModelNode(
        name="fc",
        class_name="Linear",
        module_ref=fc,
        layer_type="linear",
        param_ids=set(),
        activation_fn=functools.partial(F.softmax, dim=-1),
    )
    d = node.to_dict()
    assert isinstance(d["activation_fn"], str)


# ---------------------------------------------------------------------------
# 3. ModelMapper
# ---------------------------------------------------------------------------

def test_model_mapper_static_and_dynamic_tracing():
    changes = []
    model = DeviceMLP()
    mapper = ModelMapper(
        model,
        trace_samples=0,
        is_suitable_layer=lambda m, n: is_layer(m),
        on_graph_changed=lambda m: changes.append(len(m.graph.nodes)),
    )

    # Static-only pass (no data, budget=0)
    graph = mapper.map(model)
    assert set(graph.nodes.keys()) == {"fc1", "fc2"}
    assert len(graph.edges) == 0
    assert graph.nodes["fc1"].activation_fn is None
    assert len(changes) == 1

    # Dynamic traced pass discovers edges + activations and restores training mode
    model.train()
    assert model.training is True
    graph = mapper.map(model, data=torch.randn(2, 8), trace_samples=1)
    assert model.training is True
    assert mapper.trace_budget == 0
    assert ModelEdge("fc1", "fc2") in graph.edges
    assert graph.nodes["fc1"].activation_fn is F.relu
    assert graph.nodes["fc2"].activation_fn is F.softmax
    assert len(changes) == 2

    # Convenience views
    assert set(mapper.nodes().keys()) == {"fc1", "fc2"}
    assert mapper.edges() == [{"source": "fc1", "target": "fc2"}]


def test_model_mapper_trace_always_and_trace_context():
    model = DeviceMLP()
    mapper = ModelMapper(
        model,
        trace_samples=TRACE_ALWAYS,
        is_suitable_layer=lambda m, n: is_layer(m),
    )
    for _ in range(3):
        with mapper.trace_context():
            _ = model(torch.randn(2, 8))
        assert mapper.trace_budget == TRACE_ALWAYS

    assert ModelEdge("fc1", "fc2") in mapper.graph.edges
    # Hooks must be cleaned up after context exit
    assert len(mapper._hooks) == 0


def test_model_mapper_rejects_self_loops_on_repeated_module():
    model = RepeatedLayerNet()
    mapper = ModelMapper(model, is_suitable_layer=lambda m, n: is_layer(m))
    graph = mapper.map(model, data=torch.randn(2, 4), trace_samples=1)
    assert ModelEdge("shared", "shared") not in graph.edges
    assert ModelEdge("shared", "head") in graph.edges


def test_model_mapper_warns_on_empty_trace_sequence_and_float_out_of_range():
    class RawTensorModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.device = torch.device("cpu")
            self.unused = nn.Linear(4, 4)

        def forward(self, x):
            return x * 2.0

    with pytest.warns(RuntimeWarning, match="float outside"):
        ModelMapper(RawTensorModel(), trace_samples=1.5)

    m = RawTensorModel()
    mapper = ModelMapper(m, is_suitable_layer=lambda mod, name: is_layer(mod))
    with pytest.warns(RuntimeWarning, match="recorded no layer calls"):
        mapper.map(m, data=torch.randn(2, 4), trace_samples=1)


def test_model_mapper_map_standard_nn_module_without_device_bug():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    mapper = ModelMapper(model, is_suitable_layer=lambda m, n: is_layer(m))
    graph = mapper.map(model, data=torch.randn(2, 4), trace_samples=1)
    assert len(graph.edges) == 1


def test_model_mapper_float_trace_budget_standing_rate_bug():
    model = DeviceMLP()
    mapper = ModelMapper(model, trace_samples=0.5, is_suitable_layer=lambda m, n: is_layer(m))
    mapper.map(model, data=torch.randn(2, 8))
    # Docstring specifies float budget is a standing rate, not decremented
    assert mapper.trace_budget == 0.5


def test_model_mapper_discovers_leaky_relu_bug():
    class LeakyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.device = torch.device("cpu")
            self.fc1 = nn.Linear(4, 4)
            self.act = nn.LeakyReLU()
            self.fc2 = nn.Linear(4, 2)

        def forward(self, x):
            return self.fc2(self.act(self.fc1(x)))

    model = LeakyNet()
    mapper = ModelMapper(model, is_suitable_layer=lambda m, n: is_layer(m))
    graph = mapper.map(model, data=torch.randn(2, 4), trace_samples=1)
    assert graph.nodes["fc1"].activation_fn is F.leaky_relu


# ---------------------------------------------------------------------------
# 4. ModularObserver
# ---------------------------------------------------------------------------

def test_modular_observer_mapping_and_filtering():
    model = HybridNet()
    updates = []
    observer = ModularObserver(config=None, model=model)
    observer.on_graph_update(lambda g: updates.append(set(g.nodes.keys())))
    observer.map(dummy_data=torch.randn(2, 3, 4, 4), trace_samples=1)

    # Graph includes Conv2d and Linear, and properly excludes nn.Identity
    graph_nodes = set(observer.graph().nodes.keys())
    assert {"conv", "fc1", "fc2"}.issubset(graph_nodes)
    assert "ident" not in graph_nodes

    # Observable layers exclude Conv2d (linear-only filter) and Identity (0 params)
    assert observer.layer_names() == ["fc1", "fc2"]
    assert observer.by_name("fc1").module_ref is model.fc1
    assert observer.by_layer(model.fc2).name == "fc2"
    assert observer.by_name("conv") is None
    assert observer.by_name("ident") is None
    assert len(updates) >= 1


def test_modular_observer_hooks_lifecycle():
    model = DeviceMLP()
    observer = ModularObserver(config=None, model=model)
    observer.map(dummy_data=torch.randn(2, 8), trace_samples=1)

    pre_hits, fwd_hits, bwd_hits = [], [], []

    observer.on_pre_forward_layer(lambda m, inp: pre_hits.append(observer.by_layer(m).name))
    observer.on_forward_layer(lambda m, inp, out: fwd_hits.append((observer.by_layer(m).name, tuple(out.shape))))
    observer.on_backward_layer(lambda m, gin, gout: bwd_hits.append(observer.by_layer(m).name))

    out = model(torch.randn(3, 8))
    out.sum().backward()

    assert pre_hits == ["fc1", "fc2"]
    assert fwd_hits == [("fc1", (3, 12)), ("fc2", (3, 4))]
    assert set(bwd_hits) == {"fc1", "fc2"}

    # Unregister all explicitly and verify hooks no longer fire
    observer.on_pre_forward_layer(None)
    observer.on_forward_layer(None)
    observer.on_backward_layer(None)

    pre_hits.clear()
    fwd_hits.clear()
    bwd_hits.clear()
    model(torch.randn(3, 8)).sum().backward()
    assert pre_hits == []
    assert fwd_hits == []
    assert bwd_hits == []


def test_modular_observer_map_return_value_bug():
    model = DeviceMLP()
    observer = ModularObserver(config=None, model=model)
    result = observer.map(dummy_data=torch.randn(2, 8), trace_samples=1)
    assert result is not None


def test_modular_observer_on_grad_weights_closure_bug():
    model = DeviceMLP()
    observer = ModularObserver(config=None, model=model)
    observer.map(dummy_data=torch.randn(2, 8), trace_samples=1)

    seen_modules = set()
    seen_param_ids = set()

    def _on_grad(m, p, g):
        seen_modules.add(m)
        seen_param_ids.add(id(p))

    observer.on_grad_weights(_on_grad)

    model(torch.randn(2, 8)).sum().backward()
    observer.on_grad_weights(None)

    assert seen_modules == {model.fc1, model.fc2}
    assert len(seen_param_ids) == 4  # fc1.{weight,bias}, fc2.{weight,bias}


def test_modular_observer_clear_leaks_pre_forward_hooks_bug():
    model = DeviceMLP()
    observer = ModularObserver(config=None, model=model)
    observer.map(dummy_data=torch.randn(2, 8), trace_samples=1)

    hits = []
    observer.on_pre_forward_layer(lambda m, inp: hits.append(m))
    observer.clear()

    model(torch.randn(2, 8))
    assert hits == []


def test_modular_observer_reregister_leaks_previous_hook_bug():
    model = DeviceMLP()
    observer = ModularObserver(config=None, model=model)
    observer.map(dummy_data=torch.randn(2, 8), trace_samples=1)

    old_hits, new_hits = [], []
    observer.on_forward_layer(lambda m, i, o: old_hits.append(1))
    observer.on_forward_layer(lambda m, i, o: new_hits.append(1))

    model(torch.randn(2, 8))
    observer.on_forward_layer(None)
    assert old_hits == []
    assert len(new_hits) == 2


# ---------------------------------------------------------------------------
# 5. LayerObserver
# ---------------------------------------------------------------------------

def test_layer_observer_forward_pre_and_backward_hooks():
    model = DeviceMLP()
    mapper = ModelMapper(model, is_suitable_layer=lambda m, n: is_linear(m))
    graph = mapper.map(model)
    lo = LayerObserver(observer=None, layer_id="fc1", node=graph.nodes["fc1"])

    events = []
    lo.on_pre_forward_layer(lambda m, inp: events.append("pre"))
    lo.on_forward_layer(lambda m, inp, out: events.append("fwd"))
    lo.on_backward_layer(lambda m, gin, gout: events.append("bwd"))

    model(torch.randn(2, 8)).sum().backward()
    assert events == ["pre", "fwd", "bwd"]

    lo.on_pre_forward_layer(None)
    lo.on_forward_layer(None)
    lo.on_backward_layer(None)

    events.clear()
    model(torch.randn(2, 8)).sum().backward()
    assert events == []


def test_layer_observer_on_grad_weights_closure_and_cleanup():
    model = DeviceMLP()
    mapper = ModelMapper(model, is_suitable_layer=lambda m, n: is_linear(m))
    graph = mapper.map(model)
    lo = LayerObserver(observer=None, layer_id="fc1", node=graph.nodes["fc1"])

    seen_param_ids = set()
    lo.on_grad_weights(lambda m, p, g: seen_param_ids.add(id(p)))
    model(torch.randn(2, 8)).sum().backward()
    assert seen_param_ids == {id(model.fc1.weight), id(model.fc1.bias)}

    # Re-registering replaces old handles rather than stacking duplicates
    calls = []
    lo.on_grad_weights(lambda m, p, g: calls.append(id(p)))
    model(torch.randn(2, 8)).sum().backward()
    assert len(calls) == 2

    # Passing None cleanly removes all hooks
    calls.clear()
    lo.on_grad_weights(None)
    model(torch.randn(2, 8)).sum().backward()
    assert calls == []


def test_layer_observer_on_grad_weights_none_is_noop():
    model = DeviceMLP()
    mapper = ModelMapper(model, is_suitable_layer=lambda m, n: is_linear(m))
    graph = mapper.map(model)
    lo = LayerObserver(observer=None, layer_id="fc1", node=graph.nodes["fc1"])

    lo.on_grad_weights(None)
    model(torch.randn(2, 8)).sum().backward()
