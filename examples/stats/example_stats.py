"""Concise runnable CPU example for positronixlab.stats.

Demonstrates:
  1. Buffer conversion helpers (_as_tensor, _to_np)
  2. DisplacementFilter (standalone spatial gate)
  3. Reducers (Last, Ema, Mean, Window, Displacement)
  4. Stream (ingest / due / drain / merge_streams) + OverallProcessor
  5. UmapProcessor (uses a lightweight CPU stub if cuML is unavailable)

Run:
  .venv/bin/python positronixlab/stats/examples/example_stats.py
"""

import sys
import time
import types

import numpy as np
import torch

from positronixlab.stats.common import _as_tensor, _to_np
from positronixlab.stats.filter import DisplacementFilter
from positronixlab.stats.reducer import (
    DisplacementReducer,
    EmaReducer,
    LastReducer,
    MeanReducer,
    WindowReducer,
)


def _ensure_umap_backend() -> bool:
    """Return True if real cuML/UMAP loaded, else install a tiny CPU stub.

    Workaround for processor.py:14 unconditional `from cuml.manifold.umap import UMAP`.
    """
    try:
        import positronixlab.stats.processor  # noqa: F401

        return True
    except (ImportError, OSError):

        class _StubUMAP:
            def __init__(self, n_components=2, **_kwargs):
                self.n_components = n_components

            def fit(self, x):
                return self

            def transform(self, x):
                return np.asarray(x, dtype=np.float32)[:, : self.n_components].copy()

            def fit_transform(self, x):
                return self.transform(x)

        cuml_mod = types.ModuleType("cuml")
        manifold_mod = types.ModuleType("cuml.manifold")
        umap_mod = types.ModuleType("cuml.manifold.umap")
        manifold_mod.UMAP = _StubUMAP
        umap_mod.UMAP = _StubUMAP
        cuml_mod.manifold = manifold_mod
        sys.modules["cuml"] = cuml_mod
        sys.modules["cuml.manifold"] = manifold_mod
        sys.modules["cuml.manifold.umap"] = umap_mod
        return False


def main() -> None:
    torch.manual_seed(0)
    has_real_umap = _ensure_umap_backend()

    from positronixlab.stats.processor import OverallProcessor, UmapProcessor
    from positronixlab.stats.stream import Stream

    print(f"=== positronixlab.stats CPU example (real cuML/UMAP: {has_real_umap}) ===\n")

    # 1. Conversion helpers
    raw_t = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    t32 = _as_tensor(raw_t)
    np_copy = _to_np(t32)
    print(f"[common] _as_tensor -> dtype={t32.dtype}, requires_grad={t32.requires_grad}; _to_np -> {np_copy}")

    # 2. DisplacementFilter
    gate = DisplacementFilter(threshold=0.5, method="max")
    p0 = torch.zeros(8, 4)
    print(
        "[filter] init:",
        gate.check_and_update("fc1", p0),
        "| small move:",
        gate.check_and_update("fc1", p0 + 0.05),
        "| large move:",
        gate.check_and_update("fc1", p0 + 1.0),
    )

    # 3. Reducers
    last_r, ema_r, mean_r, win_r = LastReducer(), EmaReducer(alpha=0.5), MeanReducer(), WindowReducer(maxlen=3)
    for v in (1.0, 2.0, 3.0, 4.0):
        t = torch.tensor(v)
        last_r.update("x", t)
        ema_r.update("x", t)
        mean_r.update("x", t)
        win_r.update("x", t)
    print(
        f"[reducers] last={last_r.drain()['x'].item():.2f}, "
        f"ema={ema_r.drain()['x'].item():.2f}, "
        f"mean={mean_r.drain()['x'].item():.2f}, "
        f"window={win_r.drain()['x'].tolist()}"
    )

    disp_r = DisplacementReducer(threshold=0.5, method="rms")
    disp_r.update("fc1", p0)
    first = list(disp_r.drain().keys())
    disp_r.update("fc1", p0 + 0.05)
    suppressed = list(disp_r.drain().keys())
    disp_r.update("fc1", p0 + 1.0)
    emitted = list(disp_r.drain().keys())
    print(f"[displacement reducer] init={first}, suppressed={suppressed}, emitted={emitted}")

    # 4. Stream + merge_streams + OverallProcessor
    defaults = {
        "spaceloss": {
            "topic": "metrics",
            "reducer": "mean",
            "processor": "overall",
            "processor_args": {"topic": "stats_spaceloss_overall"},
            "every_n": 2,
        }
    }
    merged = Stream.merge_streams({"spaceloss": {"processor_args": {"key": "avg_loss"}}}, defaults)
    extractors = {"spaceloss": lambda m: [(m["layer"], m["loss"])]}
    stream = Stream("spaceloss", merged["spaceloss"], extractors)

    stream.ingest({"layer": "fc1", "loss": [0.4, 0.2]})
    stream.ingest({"layer": "fc2", "loss": [0.8, 0.6]})
    if stream.due(time.monotonic()):
        blocks, count = stream.drain()
        msgs = stream.processor.process({k: v.cpu() for k, v in blocks.items()})
        overall_msg = msgs[-1]
        print(
            f"[stream+overall] drained {count} msgs -> "
            f"topic={overall_msg['topic']}, key={overall_msg['key']}, value={overall_msg['value'].tolist()}"
        )

    # 5. UmapProcessor + centroids_2d
    umap_proc = UmapProcessor(n_components=2, n_samples=3, min_neighbors=4, max_neighbors=20)
    for step in range(4):
        out = umap_proc.process({"fc1": torch.randn(16, 6)})
        if out:
            print(f"[umap] step {step}: emitted {len(out)} frame(s), last_index={umap_proc.last_index('fc1')}")
    labels = np.array([0] * 8 + [1] * 8)
    centroids = umap_proc.centroids_2d("fc1", labels)
    print(f"[umap] centroids_2d shape={centroids.shape}, dtype={centroids.dtype}")


if __name__ == "__main__":
    main()
