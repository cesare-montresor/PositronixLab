"""
Offline CPU unit tests for `positronixlab.dataprovider`.

Covers:
- `DataConfig` serialization, updates, and deterministic hashing
- `IndexedDataset` wrapper
- `DataReader` batching, worker defaults, CUDA fork/spawn selection, and registry
- `DataProvider` lazy loader creation, `indexed_dataset` propagation, and registry
- `DataProviderFactory` routing and `MetaTransform` flattening
- Offline concrete providers (`Iris`, `Math`, `SCAN`, `OddOneOut`, `CVR`, `Syneval`,
  and `HIGGS` against a temporary mock `.npy` file)
- Strict `xfail` pins for confirmed defects documented in `analysis.md`
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import types
from typing import Optional

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset, random_split
from torchvision import transforms as tv_transforms


def _ensure_parallelazy_stub() -> bool:
    """
    Return True if real `parallelazy` is importable.
    Otherwise install a minimal stub so `positronixlab.dataprovider` can be
    imported in offline/sandboxed environments (see B1 in analysis.md).
    """
    try:
        import parallelazy  # noqa: F401
        from parallelazy.plugin_data import ManagerOptions  # noqa: F401
        return True
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
    return False


_ensure_parallelazy_stub()

from positronixlab.dataprovider import (  # noqa: E402
    DataConfig,
    DataProvider,
    DataProviderFactory,
    DataReader,
    HIGGSDataProvider,
    IndexedDataset,
    IrisDataProvider,
)
from positronixlab.dataprovider.cvr_oddoneout_dataprovider import CVROddOneOutDataProvider  # noqa: E402
from positronixlab.dataprovider.higgs_dataprovider import HIGGSDataset, _load_or_build_npy  # noqa: E402
from positronixlab.dataprovider.math_dataprovider import MathDataProvider  # noqa: E402
from positronixlab.dataprovider.oddoneout_dataprovider import OddOneOutDataProvider  # noqa: E402
from positronixlab.dataprovider.scan_dataprovider import SCANDataProvider  # noqa: E402
from positronixlab.dataprovider.syneval_oddoneout_dataprovider import SynevalOddOneOutDataProvider  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_tensor_dataset() -> TensorDataset:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(64, 6, generator=g)
    y = torch.randint(0, 3, (64,), generator=g)
    return TensorDataset(x, y)


@pytest.fixture
def mock_higgs_dir(tmp_path: Path) -> tuple[Path, int]:
    """Create a small offline HIGGS.npy file (1000 rows, 1 label + 28 features)."""
    rng = np.random.default_rng(42)
    n_samples = 1000
    labels = rng.integers(0, 2, size=(n_samples, 1)).astype(np.float32)
    features = rng.standard_normal((n_samples, 28)).astype(np.float32) * 3.0 + 1.5
    data = np.hstack([labels, features])
    np.save(tmp_path / "HIGGS.npy", data)
    return tmp_path, n_samples


# ---------------------------------------------------------------------------
# 1. DataConfig
# ---------------------------------------------------------------------------

class TestDataConfig:
    def test_defaults(self):
        cfg = DataConfig()
        assert cfg.name == "default_config"
        assert cfg.dataset_name == "fashion_mnist"
        assert cfg.batch_size == 32

    def test_with_updates_and_signature(self):
        c1 = DataConfig(dataset_name="iris", batch_size=16)
        c2 = c1.with_updates(batch_size=32, unknown_field="ignored")
        assert c1.batch_size == 16
        assert c2.batch_size == 32
        assert c2.dataset_name == "iris"
        assert c1.signature() != c2.signature()
        assert c1.signature() == DataConfig(dataset_name="iris", batch_size=16).signature()

    def test_from_dict_coercion_and_unknown_keys(self):
        cfg = DataConfig.from_dict({
            "dataset_name": "mnist",
            "batch_size": "64",
            "extra_key": 123,
        })
        assert cfg.dataset_name == "mnist"
        assert cfg.batch_size == 64
        assert isinstance(cfg.batch_size, int)

    def test_save_and_load_yaml_and_json(self, tmp_path: Path):
        cfg = DataConfig(name="exp1", dataset_name="covertype", batch_size=128)
        for ext in (".yaml", ".json"):
            p = str(tmp_path / f"cfg{ext}")
            cfg.save(p)
            loaded = DataConfig.load(p)
            assert loaded == cfg


# ---------------------------------------------------------------------------
# 2. IndexedDataset & DataReader
# ---------------------------------------------------------------------------

class TestIndexedDatasetAndDataReader:
    def test_indexed_dataset_returns_triple(self, synthetic_tensor_dataset):
        ds = IndexedDataset(synthetic_tensor_dataset)
        assert len(ds) == len(synthetic_tensor_dataset)
        x, y, idx = ds[7]
        expected_x, expected_y = synthetic_tensor_dataset[7]
        assert torch.equal(x, expected_x)
        assert torch.equal(y, expected_y)
        assert idx == 7

    def test_datareader_basic_and_zero_workers(self, synthetic_tensor_dataset):
        reader = DataReader(synthetic_tensor_dataset, batch_size=16, num_workers=0)
        assert reader.persistent_workers is False
        assert reader.multiprocessing_context is None
        batches = list(reader)
        assert len(batches) == 4
        bx, by = batches[0]
        assert bx.shape == (16, 6)
        assert by.shape == (16,)

    def test_datareader_indexed_dataset_flag(self, synthetic_tensor_dataset):
        reader = DataReader(
            synthetic_tensor_dataset,
            batch_size=16,
            num_workers=0,
            indexed_dataset=True,
        )
        bx, by, bidx = next(iter(reader))
        assert bx.shape == (16, 6)
        assert by.shape == (16,)
        assert torch.equal(bidx, torch.arange(16))

    def test_datareader_drop_last_default(self, synthetic_tensor_dataset):
        # 64 samples, batch_size=20 -> drop_last=True yields 3 batches (60 samples)
        reader = DataReader(synthetic_tensor_dataset, batch_size=20, num_workers=0)
        assert len(list(reader)) == 3

        reader_keep = DataReader(
            synthetic_tensor_dataset, batch_size=20, num_workers=0, drop_last=False
        )
        batches = list(reader_keep)
        assert len(batches) == 4
        assert batches[-1][0].shape[0] == 4

    def test_datareader_cuda_context_selection_when_uninitialized(
        self, synthetic_tensor_dataset, monkeypatch
    ):
        monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
        reader = DataReader(synthetic_tensor_dataset, batch_size=16, num_workers=1)
        assert reader.multiprocessing_context is not None
        assert reader.multiprocessing_context.get_start_method() == "fork"

    def test_datareader_subclass_registration_and_duplicate_error(self):
        key = "_TestCustomDataReaderUnique"
        try:
            class _CustomReader(DataReader, plugin_name=key):
                pass

            assert DataReader._registry[key] is _CustomReader

            with pytest.raises(ValueError, match="already registered"):
                class _DuplicateReader(DataReader, plugin_name=key):
                    pass
        finally:
            DataReader._registry.pop(key, None)


# ---------------------------------------------------------------------------
# 3. DataProvider base class
# ---------------------------------------------------------------------------

class TestDataProviderBase:
    def test_custom_provider_lazy_readers_and_caching(self, synthetic_tensor_dataset):
        key = "_TestSyntheticProviderUnique"
        try:
            class _SyntheticProvider(DataProvider, plugin_name=key):
                def __init__(self, ds, **kwargs):
                    super().__init__(**kwargs)
                    self.dataset = ds
                    g = torch.Generator().manual_seed(0)
                    tr, va = random_split(ds, [48, 16], generator=g)
                    self.dataset_train = tr
                    self.dataset_val = va
                    self.dataset_test = None

            p = _SyntheticProvider(
                synthetic_tensor_dataset,
                batch_size=16,
                num_workers=0,
                indexed_dataset=True,
            )
            assert DataProvider._registry[key] is _SyntheticProvider
            assert p._train is None
            assert p._val is None
            assert p.test is None

            train_loader = p.train
            assert train_loader is p.train  # cached
            assert isinstance(train_loader, DataReader)

            bx, by, bidx = next(iter(train_loader))
            assert bx.shape == (16, 6)
            assert by.shape == (16,)
            assert bidx.shape == (16,)
            assert len(list(p.val)) == 1
        finally:
            DataProvider._registry.pop(key, None)

    def test_dataprovider_duplicate_registration_raises(self):
        key = "_TestDupProviderUnique"
        try:
            class _P1(DataProvider, plugin_name=key):
                pass

            with pytest.raises(ValueError, match="already registered"):
                class _P2(DataProvider, plugin_name=key):
                    pass
        finally:
            DataProvider._registry.pop(key, None)


# ---------------------------------------------------------------------------
# 4. DataProviderFactory
# ---------------------------------------------------------------------------

class TestDataProviderFactory:
    def test_unknown_dataset_raises(self):
        cfg = types.SimpleNamespace(
            dataset_name="nonexistent_dataset", batch_size=16, model_type="fc"
        )
        with pytest.raises(ValueError, match="Unknown dataset"):
            DataProviderFactory.from_config(cfg, num_workers=0)

    def test_from_config_iris(self):
        cfg = types.SimpleNamespace(dataset_name="iris", batch_size=16, model_type="fc")
        p = DataProviderFactory.from_config(cfg, num_workers=0)
        assert isinstance(p, IrisDataProvider)
        assert p.num_features == 4
        assert p.num_classes == 3
        bx, by = next(iter(p.train))
        assert bx.shape == (16, 4)
        assert by.shape == (16,)

    @pytest.mark.parametrize(
        "name,expected_cls",
        [
            ("scan", SCANDataProvider),
            ("deepmind_math", MathDataProvider),
            ("cvr_odd_one_out", CVROddOneOutDataProvider),
            ("syneval_odd_one_out", SynevalOddOneOutDataProvider),
        ],
    )
    def test_from_config_offline_stub_providers(self, name, expected_cls):
        cfg = types.SimpleNamespace(dataset_name=name, batch_size=8, model_type="fc")
        p = DataProviderFactory.from_config(cfg, num_workers=0)
        assert isinstance(p, expected_cls)
        bx, by = next(iter(p.train))
        assert bx.shape[0] == 8
        assert by.shape[0] == 8

    def test_from_config_cvr_flattens_when_fc(self):
        cfg_fc = types.SimpleNamespace(
            dataset_name="cvr_odd_one_out", batch_size=4, model_type="fc"
        )
        p_fc = DataProviderFactory.from_config(cfg_fc, num_workers=0)
        bx_fc, _ = next(iter(p_fc.train))
        # Each of the 4 images (3, 64, 64) flattened -> (4, 12288)
        assert bx_fc.shape == (4, 4, 3 * 64 * 64)

        cfg_cnn = types.SimpleNamespace(
            dataset_name="cvr_odd_one_out", batch_size=4, model_type="cnn"
        )
        p_cnn = DataProviderFactory.from_config(cfg_cnn, num_workers=0)
        bx_cnn, _ = next(iter(p_cnn.train))
        assert bx_cnn.shape == (4, 4, 3, 64, 64)


# ---------------------------------------------------------------------------
# 5. Offline Concrete Providers (Iris, OddOneOut, HIGGS w/ mock memmap)
# ---------------------------------------------------------------------------

class TestOfflineConcreteProviders:
    def test_iris_direct(self):
        p = IrisDataProvider(batch_size=15, num_workers=0)
        assert p.name == "iris"
        assert p.num_features == 4
        assert p.num_classes == 3
        assert len(p.dataset) == 150
        bx, by = next(iter(p.train))
        assert bx.shape == (15, 4)
        assert by.dtype == torch.long

    def test_oddoneout_direct(self):
        p = OddOneOutDataProvider(batch_size=8, num_workers=0)
        assert p.name == "odd_one_out"
        assert p.num_classes == 4
        bx, by = next(iter(p.train))
        assert bx.shape == (8, 4, 3, 64, 64)
        assert by.shape == (8,)

    def test_higgs_provider_three_way_memmap_and_normalization(self, mock_higgs_dir):
        root, total = mock_higgs_dir
        p = HIGGSDataProvider(
            root=str(root),
            batch_size=16,
            epoch_size=0.2,
            normalize=True,
            split=[0.7, 0.15, 0.15],
            num_workers=0,
        )
        assert p.name == "higgs"
        assert p.num_features == 28
        assert p.num_classes == 2
        assert p.x_shape == (28,)
        assert p.y_shape == ()

        # Disjoint index splits summing to total
        tr_set = set(p.train_indices.tolist())
        va_set = set(p.val_indices.tolist())
        te_set = set(p.test_indices.tolist())
        assert len(tr_set) + len(va_set) + len(te_set) == total
        assert tr_set.isdisjoint(va_set)
        assert tr_set.isdisjoint(te_set)
        assert va_set.isdisjoint(te_set)

        # Normalization stats computed from train and shared with val/test
        assert p.dataset_train.mean is not None
        assert torch.allclose(p.dataset_val.mean, p.dataset_train.mean)
        assert torch.allclose(p.dataset_test.std, p.dataset_train.std)

        # Batches are normalized and properly shaped
        bx, by = next(iter(p.train))
        assert bx.shape == (16, 28)
        assert by.shape == (16,)
        assert set(by.tolist()).issubset({0, 1})

        groups = p.get_feature_groups()
        assert groups["low_level"] == list(range(21))
        assert groups["high_level"] == list(range(21, 28))

    def test_higgs_set_epoch_resamples_train_val_and_keeps_test_fixed(self, mock_higgs_dir):
        root, _ = mock_higgs_dir
        p = HIGGSDataProvider(
            root=str(root),
            batch_size=16,
            epoch_size=0.2,
            normalize=False,
            split=[0.7, 0.15, 0.15],
            num_workers=0,
        )
        test_x_initial = p.dataset_test.x.clone()

        p.set_epoch(0)
        train_e0 = p.dataset_train.x.clone()
        val_e0 = p.dataset_val.x.clone()

        p.set_epoch(1)
        train_e1 = p.dataset_train.x.clone()
        val_e1 = p.dataset_val.x.clone()

        assert not torch.equal(train_e0, train_e1)
        assert not torch.equal(val_e0, val_e1)
        assert torch.equal(p.dataset_test.x, test_x_initial)

        # Deterministic when re-setting epoch 0
        p.set_epoch(0)
        assert torch.equal(p.dataset_train.x, train_e0)

    def test_higgs_dataset_invalid_epoch_size(self, mock_higgs_dir):
        root, _ = mock_higgs_dir
        full_x, full_y = _load_or_build_npy(root)
        with pytest.raises(ValueError, match="epoch_size as float must be"):
            HIGGSDataset(full_x, full_y, epoch_size=0.0, normalize=False)
        with pytest.raises(ValueError, match="epoch_size as float must be"):
            HIGGSDataset(full_x, full_y, epoch_size=1.5, normalize=False)
        with pytest.raises(TypeError, match="epoch_size must be int or float"):
            HIGGSDataset(full_x, full_y, epoch_size="bad", normalize=False)


# ---------------------------------------------------------------------------
# 6. Confirmed Bugs (strict xfail regression pins)
# ---------------------------------------------------------------------------

class TestConfirmedBugs:
    @pytest.mark.xfail(
        strict=True,
        reason="B2: DataProviderFactory.from_config() accesses config.model_type, crashing on DataConfig.",
    )
    def test_b2_factory_accepts_data_config(self):
        cfg = DataConfig(dataset_name="iris", batch_size=16)
        p = DataProviderFactory.from_config(cfg, num_workers=0)
        assert isinstance(p, IrisDataProvider)

    @pytest.mark.xfail(
        strict=True,
        reason="B3: DataProviderFactory.from_config(..., split=None) overrides HIGGS 3-way split default with [0.7, 0.3].",
    )
    def test_b3_factory_preserves_higgs_three_way_split(self, mock_higgs_dir, monkeypatch):
        root, _ = mock_higgs_dir
        monkeypatch.chdir(root.parent)
        # Point HIGGSDataProvider default root to mock_higgs_dir
        orig_init = HIGGSDataProvider.__init__

        def _patched_init(self, root=str(mock_higgs_dir[0]), **kwargs):
            return orig_init(self, root=root, **kwargs)

        monkeypatch.setattr(HIGGSDataProvider, "__init__", _patched_init)
        cfg = types.SimpleNamespace(dataset_name="higgs", batch_size=16, model_type="fc")
        p = DataProviderFactory.from_config(cfg, num_workers=0)
        assert p.dataset_test is not None

    @pytest.mark.xfail(
        strict=True,
        reason="B5: _load_or_build_npy() uses np.save(Path('HIGGS.npy.tmp'), ...) which writes HIGGS.npy.tmp.npy and fails rename().",
    )
    def test_b5_higgs_csv_to_npy_conversion(self, tmp_path: Path):
        # Create a tiny valid 5-row HIGGS.csv (1 label + 28 features)
        rows = np.zeros((5, 29), dtype=np.float32)
        np.savetxt(tmp_path / "HIGGS.csv", rows, delimiter=",")
        full_x, full_y = _load_or_build_npy(tmp_path)
        assert full_x.shape == (5, 28)
        assert full_y.shape == (5,)

    @pytest.mark.xfail(
        strict=True,
        reason="B6: IrisDataProvider assigns identical unsplit dataset to train/val/test (100% leakage).",
    )
    def test_b6_iris_splits_are_disjoint(self):
        p = IrisDataProvider(batch_size=16, split=[0.7, 0.3], num_workers=0)
        assert p.dataset_train is not p.dataset_val
        assert len(p.dataset_train) + len(p.dataset_val) == len(p.dataset)

    @pytest.mark.xfail(
        strict=True,
        reason="B7: 9 providers unconditionally unpack 2 splits and crash on 3-way split=[0.7, 0.15, 0.15].",
    )
    def test_b7_three_way_split_supported_on_math_provider(self):
        p = MathDataProvider(batch_size=8, split=[0.7, 0.15, 0.15], num_workers=0)
        assert p.dataset_test is not None

    @pytest.mark.xfail(
        strict=True,
        reason="B9: HIGGS 2-way split uses contiguous np.arange() without shuffling and ignores split_seed.",
    )
    def test_b9_higgs_two_way_split_respects_split_seed(self, mock_higgs_dir):
        root, _ = mock_higgs_dir
        p1 = HIGGSDataProvider(
            root=str(root),
            batch_size=16,
            epoch_size=1.0,
            normalize=False,
            split=[0.8, 0.2],
            split_seed=1,
            num_workers=0,
        )
        p2 = HIGGSDataProvider(
            root=str(root),
            batch_size=16,
            epoch_size=1.0,
            normalize=False,
            split=[0.8, 0.2],
            split_seed=99,
            num_workers=0,
        )
        assert not np.array_equal(p1.train_indices, p2.train_indices)

    @pytest.mark.xfail(
        strict=True,
        reason="B10: Datasets call self.transforms.apply(x) instead of self.transforms(x), crashing on torchvision.transforms.Compose.",
    )
    def test_b10_torchvision_compose_accepted(self):
        compose = tv_transforms.Compose([tv_transforms.Lambda(lambda x: x)])
        p = MathDataProvider(batch_size=8, num_workers=0, transforms=compose)
        _ = p.dataset_train[0]
