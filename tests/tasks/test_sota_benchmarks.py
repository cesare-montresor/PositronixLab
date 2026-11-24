"""Unit tests for ModelLoader.save/load_bundle, ModularObserver generalization,
3D sequence reducers, AtlasDiscoverer decoupled attribution, and ModularityBenchmark."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from positronixlab.dataprovider.sota import DataProviderLoader
from positronixlab.metamodel.sota import MASKABLE_LAYER_TYPES, ModelLoader
from positronixlab.modularobserver.modularobserver import ModularObserver
from positronixlab.pipeline import PipelineConfig
from positronixlab.spacemath.reducers import get_reducer
from positronixlab.tasks.sota import (
    AtlasDiscoverer,
    CsordasGumbelDiscoverer,
    ModularityBenchmark,
)


def test_sequence_reducers_3d():
    x = torch.randn(4, 7, 16)
    for name in ("linear", "seq_mean", "seq_max", "seq_cls", "seq_energy"):
        out = get_reducer(name)(x)
        assert out.shape == (4, 16), f"Reducer {name} returned {out.shape}"


def test_modular_observer_supports_conv_and_target_layers():
    model = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8 * 8 * 8, 4),
    )
    obs = ModularObserver(
        config=PipelineConfig(),
        model=model,
        observable_types=MASKABLE_LAYER_TYPES,
    )
    obs.map(torch.randn(2, 3, 8, 8))
    names = obs.layer_names()
    assert "0" in names and "3" in names
    obs.clear()
    assert obs.hooks == {}


def test_model_loader_save_and_load_bundle(tmp_path: Path):
    model = ModelLoader.create(
        "csordas:mlp", n_inputs=12, n_outputs=6, layer_sizes=[16, 12], device="cpu"
    )
    bundle_path = tmp_path / "mlp_bundle.pt"
    ModelLoader.save(
        model,
        bundle_path,
        arch="csordas:mlp",
        model_kwargs={"n_inputs": 12, "n_outputs": 6, "layer_sizes": [16, 12]},
        dataset_name="csordas:add_mul",
        metrics={"val_acc": 0.91},
    )
    loaded, info = ModelLoader.load_bundle(bundle_path, device="cpu")
    assert info["arch"] == "csordas:mlp"
    assert info["dataset_name"] == "csordas:add_mul"
    assert info["metrics"]["val_acc"] == 0.91

    x = torch.randn(3, 12)
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x))


def test_atlas_decoupled_attribution_and_benchmark(tmp_path: Path):
    torch.manual_seed(0)
    dp = DataProviderLoader.create(
        "csordas:add_mul", n_train=64, n_val=32, n_test=32, n_digits=1, batch_size=16
    )
    model = ModelLoader.create(
        "csordas:mlp",
        n_inputs=dp.num_features,
        n_outputs=dp.num_classes,
        layer_sizes=[24, 16],
        device="cpu",
    )

    val_add = dp.get_subtask("add").val_loader()
    val_mul = dp.get_subtask("mul").val_loader()

    atlas = AtlasDiscoverer(atlas_epochs=1, space_dim=6, k_min=2, k_max=3)
    _, clusters = atlas.fit_atlas(model, dp.val_loader())

    # Decoupled attribution from pre-computed clusters (no second fit_atlas)
    atlas_masks = atlas.attribute_from_clusters(
        model, clusters, {"add": val_add, "mul": val_mul}
    )
    assert set(atlas_masks.keys()) == {"add", "mul"}

    # Threshold sweep (zero retraining)
    energy_add = atlas._collect_mean_unit_energy(model, val_add, list(clusters.keys()))
    sweep = atlas.sweep_fractions(clusters, energy_add, fractions=[0.25, 0.5, 0.75], mask_name="atlas:add")
    assert len(sweep) == 3

    csordas = CsordasGumbelDiscoverer(mask_loss_weight=1e-2, lr=5e-2, epochs=1)
    cs_add = csordas.discover(model, val_add, mask_name="csordas:add")
    cs_mul = csordas.discover(model, val_mul, mask_name="csordas:mul")

    bench = ModularityBenchmark(modes=("retain", "zero_ablate"))
    report = bench.compare_methods(
        model,
        masks_by_method={
            "atlas": {"add": sweep, "mul": atlas_masks["mul"]},
            "csordas": {"add": cs_add, "mul": cs_mul},
        },
        subtask_loaders={"add": val_add, "mul": val_mul},
        dataset_name="csordas:add_mul",
        model_arch="csordas:mlp",
    )

    assert len(report.results) == 6  # 3 atlas sweep (add) + 1 atlas (mul) + 2 csordas
    assert len(report.alignments) == 2
    summary = report.summary_metrics()
    assert "atlas_retain_acc" in summary and "csordas_unit_sparsity" in summary

    json_path = report.save_json(tmp_path / "report.json")
    csv_path = report.save_csv(tmp_path / "report.csv")
    loaded_json = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded_json["dataset_name"] == "csordas:add_mul"
    assert csv_path.read_text(encoding="utf-8").count("\n") >= 7
