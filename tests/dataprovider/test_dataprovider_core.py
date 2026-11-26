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


@pytest.fixture
def mock_pt_data_root(tmp_path: Path) -> Path:
    """Create small preprocessed ``.pt`` files for SCAN and DeepMind Math."""
    for subdir, x_shape, y_shape, x_dtype in (
        ("scan", (40, 20), (40, 30), torch.long),
        ("math", (40, 50), (40, 10), torch.long),
    ):
        d = tmp_path / subdir
        d.mkdir(parents=True, exist_ok=True)
        for split in ("train", "test"):
            x = torch.zeros(x_shape, dtype=x_dtype)
            y = torch.zeros(y_shape, dtype=torch.long)
            torch.save({"inputs": x, "targets": y}, d / f"{split}.pt")
    return tmp_path


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
    def test_from_config_offline_stub_providers(self, name, expected_cls, mock_pt_data_root):
        cfg = types.SimpleNamespace(dataset_name=name, batch_size=8, model_type="fc")
        extra = {"root": mock_pt_data_root} if name in ("scan", "deepmind_math") else {}
        if name == "deepmind_math":
            extra["download"] = False
        p = DataProviderFactory.from_config(cfg, num_workers=0, **extra)
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

    def test_missing_files_raise_no_dummy_fallback(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="SCAN data file not found"):
            SCANDataProvider(root=tmp_path, batch_size=8, num_workers=0, download=False)
        with pytest.raises(FileNotFoundError, match="Missing DeepMind Math files"):
            MathDataProvider(root=tmp_path, batch_size=8, num_workers=0, download=False)

    def test_scan_github_autodownload_and_cache(self, tmp_path, monkeypatch):
        from positronixlab.baselines import csordas
        from positronixlab.dataprovider import scan_dataprovider as sp

        sample_scan = (
            "IN: jump twice OUT: I_JUMP I_JUMP\n"
            "IN: walk left OUT: I_TURN_LEFT I_WALK\n"
            "IN: run right OUT: I_TURN_RIGHT I_RUN\n"
            "IN: look around left OUT: I_TURN_LEFT I_LOOK I_TURN_LEFT I_LOOK I_TURN_LEFT I_LOOK I_TURN_LEFT I_LOOK\n"
        ).encode("utf-8")
        urls: list[str] = []

        class _FakeResp:
            def raise_for_status(self) -> None:
                return None

            def iter_content(self, chunk_size: int = 1024):
                yield sample_scan

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def _fake_get(url: str, stream: bool = True, timeout: int = 60):
            urls.append(url)
            return _FakeResp()

        monkeypatch.setattr(sp.requests, "get", _fake_get)

        p = csordas.scan(
            root=tmp_path,
            split_type="jump",
            max_input_len=12,
            max_output_len=16,
            batch_size=2,
            num_workers=0,
        )
        assert urls == [sp.SCAN_URLS["jump"]["train"], sp.SCAN_URLS["jump"]["test"]]
        assert len(p.dataset) == 4
        assert len(p.dataset_test) == 4
        x0, y0 = p.dataset[0]
        assert p.in_vocab.decode(x0) == "jump twice"
        assert p.out_vocab.decode(y0) == "I_JUMP I_JUMP"

        # Second call hits cache -> 0 HTTP requests
        urls.clear()
        p2 = csordas.scan(
            root=tmp_path,
            split_type="jump",
            max_input_len=12,
            max_output_len=16,
            batch_size=2,
            num_workers=0,
        )
        assert urls == []
        assert len(p2.dataset) == 4

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
# 6. Regression tests for previously confirmed bugs (B1-B14) & new features
# ---------------------------------------------------------------------------

class TestFixedBugsAndNewFeatures:
    def test_b2_factory_accepts_data_config(self):
        cfg = DataConfig(dataset_name="iris", batch_size=16)
        p = DataProviderFactory.from_config(cfg, num_workers=0)
        assert isinstance(p, IrisDataProvider)

    def test_b3_factory_preserves_higgs_three_way_split(self, mock_higgs_dir):
        root, _ = mock_higgs_dir
        cfg = DataConfig(dataset_name="higgs", batch_size=16, root=str(root))
        p = DataProviderFactory.from_config(cfg, num_workers=0)
        assert p.dataset_test is not None
        assert p.dataset_split == [0.7, 0.15, 0.15]

    def test_b5_higgs_csv_to_npy_conversion(self, tmp_path: Path):
        rows = np.zeros((5, 29), dtype=np.float32)
        np.savetxt(tmp_path / "HIGGS.csv", rows, delimiter=",")
        full_x, full_y = _load_or_build_npy(tmp_path)
        assert full_x.shape == (5, 28)
        assert full_y.shape == (5,)
        assert (tmp_path / "HIGGS.npy").exists()

    def test_b6_iris_splits_are_disjoint_and_support_transforms(self):
        p = IrisDataProvider(
            batch_size=16,
            split=[0.7, 0.3],
            num_workers=0,
            transforms=lambda x: x * 2.0,
        )
        assert p.dataset_train is not p.dataset_val
        assert len(p.dataset_train) + len(p.dataset_val) == len(p.dataset)
        assert set(p.dataset_train.indices.tolist()).isdisjoint(
            set(p.dataset_val.indices.tolist())
        )
        x0, _ = p.dataset_train[0]
        raw_idx = int(p.dataset_train.indices[0])
        assert torch.allclose(x0, p.dataset._full_x[raw_idx] * 2.0)

    def test_b7_three_way_split_supported_on_math_provider(self, mock_pt_data_root):
        p = MathDataProvider(
            root=mock_pt_data_root,
            batch_size=8,
            split=[0.7, 0.15, 0.15],
            num_workers=0,
            download=False,
        )
        assert p.dataset_test is not None
        assert len(p.dataset_train) + len(p.dataset_val) + len(p.dataset_test) == len(p.dataset)

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

    def test_b10_torchvision_compose_and_metatransform_callable(self, mock_pt_data_root):
        from positronixlab.metamodel.metatransform import MetaTransform

        compose = tv_transforms.Compose([tv_transforms.Lambda(lambda x: x)])
        p = MathDataProvider(
            root=mock_pt_data_root,
            batch_size=8,
            num_workers=0,
            transforms=compose,
            download=False,
        )
        _ = p.dataset_train[0]

        mt = MetaTransform(["flatten"])
        out = mt(torch.zeros(1, 4, 4))
        assert out.shape == (16,)

    def test_base_tensor_dataset_zero_copy_and_idempotent_normalization(self):
        from positronixlab.dataprovider import BaseTensorDataset

        x = torch.arange(100, dtype=torch.float32).reshape(20, 5)
        y = torch.arange(20, dtype=torch.long)
        ds = BaseTensorDataset(x, y, normalize=True, norm_cols=slice(0, 2))
        tr, va = ds.random_split([0.8, 0.2], seed=0)
        # Zero-copy: backing tensor is shared
        assert tr._full_x is ds._full_x
        assert va._full_x is ds._full_x

        mean, std = tr.compute_normalization()
        first_x0, _ = tr[0]
        # Calling compute_normalization / apply_normalization twice does not double-normalize
        tr.compute_normalization()
        va.apply_normalization(mean, std)
        va.apply_normalization(mean, std)
        second_x0, _ = tr[0]
        assert torch.allclose(first_x0, second_x0)

    def test_higgs_sampler_resamples_across_epochs_with_loader(self, mock_higgs_dir):
        root, _ = mock_higgs_dir
        p = HIGGSDataProvider(
            root=str(root),
            batch_size=16,
            epoch_size=0.2,
            normalize=False,
            num_workers=0,
        )
        p.set_epoch(0)
        b0 = next(iter(p.train))[0].clone()
        p.set_epoch(1)
        b1 = next(iter(p.train))[0].clone()
        assert not torch.equal(b0, b1)
        p.set_epoch(0)
        b0_again = next(iter(p.train))[0].clone()
        assert torch.equal(b0, b0_again)

    def test_get_loader_overrides_and_drop_last_default(self):
        p = IrisDataProvider(batch_size=16, num_workers=0)
        assert p.train.drop_last is True
        assert p.val.drop_last is True
        custom = p.get_loader("val", batch_size=10, drop_last=False, shuffle=False)
        assert custom.batch_size == 10
        assert custom.drop_last is False

    def test_registry_lookup_and_aliases(self, mock_pt_data_root):
        assert DataProvider.get("iris") is IrisDataProvider
        assert "iris" in DataProviderFactory.list_available()
        assert "scan" in DataProviderFactory.list_available()
        p = DataProviderFactory.create(
            "scan", root=mock_pt_data_root, batch_size=8, num_workers=0, download=False
        )
        assert isinstance(p, SCANDataProvider)


# ---------------------------------------------------------------------------
# 7. Unified Baseline Compositional DataProviders under dataprovider
# ---------------------------------------------------------------------------

class TestUnifiedCompositionalProviders:
    def test_class_removal_from_provider_and_factory(self):
        from positronixlab.dataprovider import ClassRemovalDataProvider, CompositionalDataProvider

        iris = IrisDataProvider(batch_size=8, num_workers=0)
        cr = ClassRemovalDataProvider.from_provider(iris)
        assert isinstance(cr, CompositionalDataProvider)
        assert isinstance(cr, DataProvider)
        assert cr.list_subtasks() == ["all", "without_0", "without_1", "without_2"]
        assert set(cr.eval_matrix_spec().keys()) == {"without_0", "without_1", "without_2"}
        for _, by in cr.get_subtask("without_1").train_loader():
            assert (by != 1).all()
        # Both standard DataProvider (.train) and Compositional (.train_loader()) work
        bx, _ = next(iter(cr.train))
        assert bx.shape == (8, 4)

    def test_addmul_and_tuple_arithmetic_subtasks_and_modes(self):
        from positronixlab.dataprovider import AddMulDataProvider, TupleArithmeticDataProvider

        # Original per-digit mode
        am_digits = AddMulDataProvider(
            n_digits=2, n_train=64, n_val=16, n_test=16, batch_size=16, target_mode="digits"
        )
        assert am_digits.list_subtasks() == ["full", "add", "mul"]
        assert am_digits.eval_matrix_spec() == {
            "add": ["full", "add", "mul"],
            "mul": ["full", "add", "mul"],
        }
        bx, by = next(iter(am_digits.train))
        assert bx.shape == (16, 42)
        assert by.shape == (16, 2)

        # Single-head integer mode via factory alias
        am_int = DataProviderFactory.create(
            "csordas:add_mul",
            n_digits=1,
            n_train=64,
            n_val=16,
            n_test=16,
            batch_size=16,
            target_mode="integer",
        )
        bx_i, by_i = next(iter(am_int.get_subtask("add").val_loader()))
        assert bx_i.shape == (16, 22)
        assert by_i.shape == (16,)
        assert (bx_i[:, -2] == 1).all()

        # TupleArithmetic subtasks via factory alias
        ta = DataProviderFactory.create(
            "csordas:tuple_arithmetic",
            n_digits=1,
            n_tuples=2,
            n_train=64,
            n_val=16,
            n_test=16,
            batch_size=16,
            target_mode="integer",
        )
        assert ta.list_subtasks() == ["full", "tuple_0", "tuple_1"]
        t1_x, t1_y = next(iter(ta.get_subtask("tuple_1").val_loader()))
        assert (t1_x[:, :20] == 0).all()
        assert t1_y.shape == (16,)

    def test_lepori_cvr_and_syneval_via_dataprovider_factory(self):
        from positronixlab.dataprovider import (
            CVR_FAMILIES,
            SYNEVAL_FAMILIES,
            CompositionalDataProvider,
        )

        cvr = DataProviderFactory.create(
            "lepori:cvr_inside_count",
            n_train=16,
            n_val=8,
            n_test=8,
            image_size=32,
            batch_size=4,
            use_real_generator=False,
        )
        assert isinstance(cvr, CompositionalDataProvider)
        assert set(cvr.list_subtasks()) == set(CVR_FAMILIES["inside_count"].keys())
        assert set(cvr.eval_matrix_spec().keys()) == {"inside", "count"}
        bx, by = next(iter(cvr.train))
        assert bx.shape == (4, 4, 3, 32, 32)
        assert by.shape == (4,)

        syn = DataProviderFactory.create(
            "lepori:syneval_pro_ant_plural",
            n_train=16,
            n_val=8,
            n_test=8,
            max_length=16,
            batch_size=4,
        )
        assert isinstance(syn, CompositionalDataProvider)
        assert set(syn.list_subtasks()) == set(SYNEVAL_FAMILIES["pro_ant_plural"].keys())
        assert syn.get_subtask("ablate_pro").train_loader() is None
        sx, sy = next(iter(syn.train))
        assert sx.shape == (4, 2, 4, 16)
        assert sy.shape == (4,)


class TestMathDataProviderAutoDownload:
    """Offline unit tests for MathDataProvider Hugging Face auto-download & caching."""

    def test_math_vocab_round_trip(self):
        from positronixlab.dataprovider.math_dataprovider import MathVocab

        vocab = MathVocab()
        assert len(vocab) == 99
        q = "What is 2 + 2?"
        a = "4"
        q_ids = vocab.encode(q, max_len=32, add_sos=False, add_eos=True)
        a_ids = vocab.encode(a, max_len=8, add_sos=True, add_eos=True)
        assert q_ids.shape == (32,)
        assert a_ids.shape == (8,)
        assert int(a_ids[0]) == vocab.SOS_IDX
        assert int(a_ids[2]) == vocab.EOS_IDX
        assert int(a_ids[3]) == vocab.PAD_IDX
        assert vocab.decode(q_ids) == q
        assert vocab.decode(a_ids) == a

    def test_hf_autodownload_parse_and_cache(self, tmp_path, monkeypatch):
        from positronixlab.baselines import csordas
        from positronixlab.dataprovider import math_dataprovider as mp

        easy_txt = (
            "What is 1 + 1?\n2\n"
            "What is 2 + 3?\n5\n"
            "What is 10 - 4?\n6\n"
            "What is 7 + 8?\n15\n"
            "What is 9 - 2?\n7\n"
            "What is 3 + 4?\n7\n"
        ).encode("utf-8")
        # Include one duplicate question from easy_txt to verify cross-split deduplication
        extrap_txt = (
            "What is 1 + 1?\n2\n"
            "What is 1000 + 2000?\n3000\n"
            "What is 5000 - 1234?\n3766\n"
            "What is 9999 + 1?\n10000\n"
        ).encode("utf-8")

        requested_urls: list[str] = []

        class _FakeResponse:
            def __init__(self, payload: bytes) -> None:
                self._payload = payload
                self.headers = {"content-length": str(len(payload))}

            def raise_for_status(self) -> None:
                return None

            def iter_content(self, chunk_size: int = 1024):
                yield self._payload

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def _fake_get(url: str, stream: bool = True, timeout: int = 60):
            requested_urls.append(url)
            payload = extrap_txt if "/extrapolate/" in url else easy_txt
            return _FakeResponse(payload)

        monkeypatch.setattr(mp.requests, "get", _fake_get)

        provider = csordas.deepmind_math(
            root=tmp_path,
            tasks=["arithmetic__add_or_sub"],
            difficulties=("easy",),
            eval_split="extrapolate",
            include_extrapolate=True,
            max_question_len=40,
            max_answer_len=12,
            max_samples_per_file=5,
            batch_size=2,
            num_workers=0,
            show_progress=False,
        )

        # Should download train-easy/arithmetic__add_or_sub.txt and extrapolate/arithmetic__add_or_sub_big.txt
        assert requested_urls == [
            f"https://huggingface.co/datasets/{mp.HF_DM_MATH_REPO}/resolve/main/train-easy/arithmetic__add_or_sub.txt",
            f"https://huggingface.co/datasets/{mp.HF_DM_MATH_REPO}/resolve/main/extrapolate/arithmetic__add_or_sub_big.txt",
        ]
        assert (tmp_path / "math" / "raw" / "train-easy" / "arithmetic__add_or_sub.txt").exists()
        assert (tmp_path / "math" / "raw" / "extrapolate" / "arithmetic__add_or_sub_big.txt").exists()

        # 5 easy samples -> 1 carved to IID test, 4 remain in train
        assert len(provider.get_subtask("easy").dataset_train) == 4
        assert len(provider.get_subtask("iid").dataset_test) == 1
        # Extrapolate had 1 duplicate ("What is 1 + 1?") removed -> 3 unique samples
        assert len(provider.get_subtask("extrapolate").dataset_test) == 3
        assert set(provider.list_subtasks()) >= {"full", "easy", "iid", "extrapolate"}
        assert "easy" in provider.eval_matrix_spec()
        assert provider.vocab_size == 99
        assert provider.x_shape == (40,)
        assert provider.y_shape == (12,)

        # Second call hits compositional cache -> zero additional HTTP requests
        requested_urls.clear()
        provider2 = csordas.deepmind_math(
            root=tmp_path,
            tasks=["arithmetic__add_or_sub"],
            difficulties=("easy",),
            eval_split="extrapolate",
            include_extrapolate=True,
            max_question_len=40,
            max_answer_len=12,
            max_samples_per_file=5,
            batch_size=2,
            num_workers=0,
            show_progress=False,
        )
        assert requested_urls == []
        assert len(provider2.get_subtask("easy").dataset_train) == 4
        assert len(provider2.get_subtask("extrapolate").dataset_test) == 3

    def test_multi_task_subtasks_and_seq2seq_metrics(self, tmp_path):
        from positronixlab.baselines import DataProviderLoader, ModelLoader
        from positronixlab.dataprovider.math_dataprovider import (
            MathSeq2SeqLoss,
            MathVocab,
            math_exact_accuracy,
            math_token_accuracy,
        )

        # Seed local raw .txt files for two modules (no network needed)
        for task, qa in (
            (
                "arithmetic__add_or_sub",
                "1+1?\n2\n2+2?\n4\n3+3?\n6\n4+4?\n8\n5+5?\n10\n",
            ),
            (
                "arithmetic__mul",
                "2*2?\n4\n3*3?\n9\n4*4?\n16\n5*5?\n25\n6*6?\n36\n",
            ),
        ):
            for split in ("train-easy", "train-hard", "interpolate"):
                p = tmp_path / "math" / "raw" / split / f"{task}.txt"
                p.parent.mkdir(parents=True, exist_ok=True)
                prefix = f"[{split}] "
                lines = qa.strip().splitlines()
                unique_qa = "\n".join(
                    f"{prefix}{lines[i]}\n{lines[i + 1]}" for i in range(0, len(lines), 2)
                )
                p.write_text(unique_qa + "\n", encoding="utf-8")

        dp = DataProviderLoader.create(
            "csordas:deepmind_math",
            root=tmp_path,
            tasks=["arithmetic__add_or_sub", "arithmetic__mul"],
            difficulties=("easy", "hard"),
            eval_split="interpolate",
            max_question_len=32,
            max_answer_len=10,
            batch_size=2,
            num_workers=0,
            download=False,
        )
        assert set(dp.list_subtasks()) == {
            "full",
            "easy",
            "hard",
            "iid",
            "interpolate",
            "arithmetic__add_or_sub",
            "arithmetic__mul",
        }
        assert set(dp.eval_matrix_spec().keys()) == {"arithmetic__add_or_sub", "arithmetic__mul"}

        # Verify end-to-end compatibility with CsordasTransformerEncDec + MathSeq2SeqLoss
        model = ModelLoader.create(
            "csordas:transformer",
            n_input_tokens=dp.vocab_size,
            n_output_tokens=dp.vocab_size,
            state_size=16,
            n_heads=2,
            n_encoder_layers=1,
            n_decoder_layers=1,
        )
        bx, by = next(iter(dp.train_loader()))
        logits = model(bx, by[:, :-1])
        loss_fn = MathSeq2SeqLoss(pad_idx=MathVocab.PAD_IDX)
        loss = loss_fn(logits, by)
        assert torch.isfinite(loss)
        assert 0.0 <= math_token_accuracy(logits, by) <= 1.0
        assert 0.0 <= math_exact_accuracy(logits, by) <= 1.0

        # Perfect logits -> 1.0 token and exact accuracy
        perfect = torch.full((2, by.size(1) - 1, dp.vocab_size), -10.0)
        for b in range(2):
            for t in range(by.size(1) - 1):
                perfect[b, t, int(by[b, t + 1])] = 10.0
        assert math_token_accuracy(perfect, by) == 1.0
        assert math_exact_accuracy(perfect, by) == 1.0

    def test_explicit_tasks_without_download_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Missing DeepMind Math files"):
            MathDataProvider(
                root=tmp_path,
                tasks=["arithmetic__mul"],
                difficulties=("easy",),
                download=False,
                num_workers=0,
            )

    def test_legacy_pt_files_loaded_without_network(self, tmp_path):
        math_dir = tmp_path / "math"
        math_dir.mkdir(parents=True)
        torch.save(
            {"inputs": torch.ones((12, 20), dtype=torch.long), "targets": torch.ones((12, 6), dtype=torch.long)},
            math_dir / "train.pt",
        )
        torch.save(
            {"inputs": torch.ones((4, 20), dtype=torch.long), "targets": torch.ones((4, 6), dtype=torch.long)},
            math_dir / "test.pt",
        )
        p = MathDataProvider(root=tmp_path, batch_size=4, num_workers=0, download=False)
        assert len(p.dataset) == 12
        assert len(p.dataset_test) == 4
        assert p.x_shape == (20,)
        assert p.y_shape == (6,)
