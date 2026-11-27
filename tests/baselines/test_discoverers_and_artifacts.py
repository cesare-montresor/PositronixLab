"""Unit tests for native discoverers, loader helpers, and portable baseline artifacts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from positronixlab.baselines import (
    BaselineArtifactLoader,
    CheckpointStore,
    ContinuousSparsificationDiscoverer,
    CsordasGumbelDiscoverer,
    DataProviderLoader,
    ModelLoader,
    NeuroSurgeonDiscoverer,
)
from positronixlab.dataprovider import AddMulDataProvider, make_fast_loader
from positronixlab.tasks import SubnetworkMask, resolve_subtask_criterion


def test_csordas_and_native_continuous_sparsification_discoverers() -> None:
    dp = AddMulDataProvider(n_digits=1, n_train=64, n_val=32, n_test=32, batch_size=16, seed=1)
    kwargs = ModelLoader.infer_kwargs("csordas:mlp", dp, {"layer_sizes": [32, 16]})
    model = ModelLoader.create("csordas:mlp", device="cpu", **kwargs)
    fast_loader = make_fast_loader(dp.train_loader(), device="cpu", batch_size=16, shuffle=True)

    loss_fn = resolve_subtask_criterion(dp)
    csordas = CsordasGumbelDiscoverer(
        epochs=3,
        lr=5e-2,
        mask_loss_weight=1e-3,
        loss_fn=loss_fn,
        early_stop_patience=2,
        early_stop_tol=1.0,
    )
    m_cs = csordas.discover(model, fast_loader, mask_name="csordas:test")
    assert m_cs.granularity == "weight"
    assert csordas.last_stats["backend"] == "native_fast"
    assert 1 <= csordas.last_stats["epochs_run"] <= 3

    lepori_w = ContinuousSparsificationDiscoverer(
        mask_unit="weight",
        epochs=2,
        lr=5e-2,
        l0_lambda=1e-4,
        max_temp=20.0,
        loss_fn=loss_fn,
    )
    m_lw = lepori_w.discover(model, fast_loader, mask_name="lepori:w")
    assert m_lw.granularity == "weight"
    assert lepori_w.last_stats["backend"] == "native_fast"

    lepori_n = NeuroSurgeonDiscoverer(
        mask_method="continuous_sparsification",
        mask_unit="neuron",
        epochs=2,
        lr=5e-2,
        l0_lambda=1e-4,
        max_temp=20.0,
        loss_fn=loss_fn,
        backend="native",
    )
    m_ln = lepori_n.discover(model, fast_loader, mask_name="lepori:n")
    assert m_ln.granularity == "unit"
    assert lepori_n.last_stats["backend"] == "native_fast"


def test_discoverers_zero_epochs_and_invalid_backend() -> None:
    dp = AddMulDataProvider(n_digits=1, n_train=32, n_val=16, n_test=16, batch_size=16, seed=0)
    kwargs = ModelLoader.infer_kwargs("csordas:mlp", dp, {"layer_sizes": [16]})
    model = ModelLoader.create("csordas:mlp", device="cpu", **kwargs)
    loader = dp.train_loader()

    cs0 = CsordasGumbelDiscoverer(epochs=0, loss_fn=resolve_subtask_criterion(dp))
    cs0.discover(model, loader)
    assert cs0.last_stats["epochs_run"] == 0

    lepori0 = ContinuousSparsificationDiscoverer(epochs=0, loss_fn=resolve_subtask_criterion(dp))
    lepori0.discover(model, loader)
    assert lepori0.last_stats["epochs_run"] == 0

    with pytest.raises(ValueError, match="backend='native' only supports"):
        NeuroSurgeonDiscoverer(mask_method="hard_concrete", backend="native")

    with pytest.raises(ValueError, match="backend must be"):
        NeuroSurgeonDiscoverer(backend="invalid")  # type: ignore[arg-type]


def test_checkpoint_store_and_portable_artifact_loader(tmp_path: Path) -> None:
    dp = AddMulDataProvider(n_digits=1, n_train=32, n_val=16, n_test=16, batch_size=16, seed=0)
    kwargs = ModelLoader.infer_kwargs("csordas:mlp", dp, {"layer_sizes": [16]})
    model = ModelLoader.create("csordas:mlp", device="cpu", **kwargs)

    orig_dir = tmp_path / "runs" / "run_01"
    store = CheckpointStore(orig_dir)
    assert not store.is_model_done()
    ModelLoader.save(model, store.model_bundle_path, arch="csordas:mlp", model_kwargs=kwargs)
    store.mark_model_done(store.model_bundle_path, metrics={"val_accuracy": 0.9}, duration_sec=0.5)
    assert store.is_model_done()

    mask = SubnetworkMask(
        name="csordas:add",
        granularity="unit",
        masks={"layers.0": torch.ones(16, dtype=torch.bool)},
    )
    store.save_subtask_mask("csordas", "add", mask, duration_sec=0.1)
    assert store.is_subtask_mask_done("csordas", "add")
    assert store.is_method_done("csordas", ["add"])
    store.mark_method_done("csordas", total_duration_sec=0.1)

    report_path = orig_dir / "report.json"
    report_path.write_text("{}", encoding="utf-8")
    manifest_path = store.write_manifest(
        config_dict={"dataset_name": "csordas:add_mul"},
        subtasks=["add"],
        report_json_path=report_path,
    )
    assert manifest_path.is_file()

    # Manifest must store portable relative paths
    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert raw_manifest["work_dir"] == "."
    assert raw_manifest["model"]["bundle_path"] == "model_bundle.pt"
    assert raw_manifest["methods"]["csordas"]["subtasks"]["add"]["path"] == "masks/csordas/add.pt"
    assert raw_manifest["report_json_path"] == "report.json"

    # Relocate the entire directory and load via parametrized root_dir
    relocated_root = tmp_path / "moved_root"
    relocated_dir = relocated_root / "run_01"
    shutil.move(str(orig_dir), str(relocated_dir))

    artifact = BaselineArtifactLoader.load("run_01", root_dir=relocated_root)
    assert artifact.subtasks == ["add"]
    assert artifact.available_methods() == ["csordas"]
    assert artifact.report_json_path == relocated_dir / "report.json"
    loaded_model, info = artifact.load_model(device="cpu")
    assert info["arch"] == "csordas:mlp"
    assert isinstance(loaded_model, nn.Module)
    loaded_masks = artifact.load_all_masks()
    assert "add" in loaded_masks["csordas"]

    # Instance API with bound root_dir
    loader = BaselineArtifactLoader(root_dir=relocated_root)
    artifact2 = loader.load_artifact("run_01")
    assert artifact2.model_bundle_path == relocated_dir / "model_bundle.pt"


def test_dataprovider_loader_resolve_subtasks() -> None:
    dp = AddMulDataProvider(n_digits=1, n_train=32, n_val=16, n_test=16, batch_size=16, seed=0)
    resolved = DataProviderLoader.resolve_subtasks(dp)
    assert "add" in resolved and "mul" in resolved
    assert "full" not in resolved and "all" not in resolved
    assert DataProviderLoader.resolve_subtasks(dp, ["add"]) == ["add"]
