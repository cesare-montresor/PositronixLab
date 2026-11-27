"""
Runnable offline CPU example for `positronixlab.dataprovider`.

Demonstrates:
1. `DataConfig` creation, immutable updates, and deterministic signature
2. Direct instantiation of `IrisDataProvider` (offline sklearn dataset)
3. `IndexedDataset` via `indexed_dataset=True` (yielding `(x, y, idx)`)
4. Writing a custom `DataProvider` subclass over synthetic tensors
5. Routing through `DataProviderFactory.from_config()`
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import torch
from torch.utils.data import TensorDataset, random_split


def _ensure_parallelazy_stub() -> None:
    """Allow running this example offline even if `parallelazy` is unavailable."""
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

from positronixlab.dataprovider import (  # noqa: E402
    DataConfig,
    DataProvider,
    DataProviderFactory,
    IrisDataProvider,
)


class SyntheticDataProvider(DataProvider, plugin_name="synthetic_example"):
    """Minimal offline synthetic provider demonstrating DataProvider subclassing."""

    def __init__(self, n_samples: int = 128, n_features: int = 8, n_classes: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.name = "synthetic_example"
        self.num_features = n_features
        self.num_classes = n_classes
        self.x_shape = (n_features,)
        self.y_shape = ()

        g = torch.Generator().manual_seed(0)
        x = torch.randn(n_samples, n_features, generator=g)
        y = torch.randint(0, n_classes, (n_samples,), generator=g)
        self.dataset = TensorDataset(x, y)

        splits = random_split(self.dataset, self.dataset_split, generator=g)
        self.dataset_train = splits[0]
        self.dataset_val = splits[1]
        self.dataset_test = splits[2] if len(splits) > 2 else None


def main() -> None:
    print("=== 1. DataConfig ===")
    cfg = DataConfig(dataset_name="iris", batch_size=16)
    cfg_updated = cfg.with_updates(batch_size=32)
    print(f"cfg={cfg.to_dict()} sig={cfg.signature()[:8]}")
    print(f"updated={cfg_updated.to_dict()} sig={cfg_updated.signature()[:8]}")

    print("\n=== 2. IrisDataProvider (offline) ===")
    iris = IrisDataProvider(batch_size=16, num_workers=0)
    x, y = next(iter(iris.train))
    print(
        f"name={iris.name} features={iris.num_features} classes={iris.num_classes} "
        f"batch_x={tuple(x.shape)} batch_y={tuple(y.shape)}"
    )

    print("\n=== 3. IndexedDataset mode ===")
    iris_idx = IrisDataProvider(batch_size=16, num_workers=0, indexed_dataset=True)
    x, y, idx = next(iter(iris_idx.train))
    print(f"batch_x={tuple(x.shape)} batch_y={tuple(y.shape)} first_indices={idx[:5].tolist()}")

    print("\n=== 4. Custom SyntheticDataProvider (3-way split) ===")
    synth = SyntheticDataProvider(
        n_samples=128,
        batch_size=16,
        split=[0.5, 0.25, 0.25],
        num_workers=0,
    )
    print(
        f"train_batches={len(synth.train)} val_batches={len(synth.val)} "
        f"test_batches={len(synth.test)}"
    )

    print("\n=== 5. DataProviderFactory.from_config() ===")
    factory_cfg = types.SimpleNamespace(dataset_name="iris", batch_size=16, model_type="fc")
    from_factory = DataProviderFactory.from_config(factory_cfg, num_workers=0)
    fx, fy = next(iter(from_factory.train))
    print(f"factory_provider={type(from_factory).__name__} batch_x={tuple(fx.shape)}")


if __name__ == "__main__":
    main()
