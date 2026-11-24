"""Runnable CPU example demonstrating ModelMapper, GraphFormatter, and ModularObserver."""

from __future__ import annotations

from dataclasses import dataclass
import sys
import types
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from positronixlab.modularobserver.modelmapper import GraphFormatter, ModelMapper
from positronixlab.spacemath.common import is_layer


def _ensure_parallelazy_stub() -> None:
    """Fallback stub if `parallelazy` is not importable in the current environment."""
    try:
        import parallelazy  # noqa: F401
        from parallelazy.plugin_data import ManagerOptions  # noqa: F401
        return
    except ImportError:
        pass

    @dataclass
    class _StubManagerOptions:
        work_dir: Optional[str] = None
        log_dir: Optional[str] = None
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


class DemoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv(x))
        x = self.pool(x).flatten(1)
        x = F.gelu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)


def main() -> None:
    torch.manual_seed(0)
    model = DemoNet()
    sample_x = torch.randn(4, 3, 8, 8)

    # 1. Standalone ModelMapper + GraphFormatter
    print("=== 1. ModelMapper (Static + Dynamic Trace) ===")
    mapper = ModelMapper(model, is_suitable_layer=lambda m, name: is_layer(m))
    graph = mapper.map(model, data=sample_x, trace_samples=1)

    for name, node in graph.nodes.items():
        act = node.activation_fn.__name__ if node.activation_fn else "none"
        print(
            f"  node={name:<6} class={node.class_name:<8} type={node.layer_type:<7} "
            f"units={node.in_units}->{node.out_units} ({node.unit_type}) act={act}"
        )
    print("  edges:", [(e.source, e.target) for e in graph.edges])

    cyto = GraphFormatter.to_cytoscape(graph)
    print(f"  cytoscape elements: {len(cyto['elements'])} ({len(graph.nodes)} nodes + {len(graph.edges)} edges)")

    # 2. ModularObserver hook collection
    print("\n=== 2. ModularObserver (Hooks & Lifecycle) ===")
    observer = ModularObserver(config=None, model=model)
    observer.on_graph_update(lambda g: print(f"  [graph update] {len(g.nodes)} nodes, {len(g.edges)} edges"))
    observer.map(dummy_data=sample_x, trace_samples=1)
    print("  observable layers:", observer.layer_names())

    activations = {}
    grad_outputs = {}

    observer.on_forward_layer(
        lambda m, inp, out: activations.__setitem__(observer.by_layer(m).name, tuple(out.shape))
    )
    observer.on_backward_layer(
        lambda m, gin, gout: grad_outputs.__setitem__(observer.by_layer(m).name, tuple(gout[0].shape))
    )

    out = model(sample_x)
    out.sum().backward()

    print("  captured forward activations:", activations)
    print("  captured backward grad_outputs:", grad_outputs)

    # Clean up hooks (pre-forward explicitly first, then clear)
    observer.on_pre_forward_layer(None)
    observer.clear()
    print("  cleared; observable layers now:", observer.layer_names())


if __name__ == "__main__":
    main()
