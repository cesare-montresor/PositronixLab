import pytest
import torch
import torch.nn.functional as F

from positronixlab.modularspace.config import LayerSpaceConfig
from positronixlab.modularspace.layerspace import LayerSpace
from positronixlab.modularspace.presets import get_preset, list_presets
from positronixlab.modularspace.space_step import SpaceStepSpec
from positronixlab.pipeline.config import BaseConfig


def _make_observed_layers(num_cus: int = 24, layer_type: str = "linear"):
    return {
        "layer_0": {
            "out_units": num_cus,
            "layer_type": layer_type,
            "activation_type": "relu",
        }
    }


def test_base_config_independent_of_nas(tmp_path):
    cfg = BaseConfig(name="unit_test")
    assert cfg.to_dict() == {"name": "unit_test"}
    assert len(cfg.signature()) == 32

    path = tmp_path / "cfg.json"
    cfg.save(str(path))
    loaded = BaseConfig.load(str(path))
    assert loaded == cfg


def test_layerspace_config_to_step_spec():
    cfg = LayerSpaceConfig.from_dict(get_preset("softmax_ce", modular_space_tau=0.2))
    spec = SpaceStepSpec.from_config(cfg)
    assert spec.contrast_method == "softmax_ce"
    assert spec.weight_method == "inv_freq"
    assert spec.tau == pytest.approx(0.2)
    assert spec.sigreg_target == "positions"


@pytest.mark.parametrize("preset_name", list_presets())
@pytest.mark.parametrize("stats_enabled", [False, True])
def test_layerspace_runs_all_presets(preset_name, stats_enabled):
    torch.manual_seed(0)
    cfg = LayerSpaceConfig.from_dict(get_preset(preset_name))
    space = LayerSpace(
        cfg,
        _make_observed_layers(24),
        device=torch.device("cpu"),
        stats_enabled=stats_enabled,
    )
    activations = torch.randn(16, 24)
    labels = torch.arange(24) % 3

    losses, positions, stats = space.process_activations("layer_0", activations, clusters_labels=labels)
    assert "loss" in losses
    assert torch.isfinite(torch.tensor(float(losses["loss"])))
    assert isinstance(positions, torch.Tensor)
    assert positions.shape == (24, cfg.modular_space_size)
    if stats_enabled:
        assert stats is not None and stats.shape == (24, 4)
    else:
        assert stats is None


def test_layerspace_conditional_overrides_skip_zero_weight_terms():
    torch.manual_seed(1)
    cfg = LayerSpaceConfig()
    space = LayerSpace(
        cfg,
        _make_observed_layers(20),
        device=torch.device("cpu"),
        score_method="sign_preserving_row",
        apply_activation=True,
        weight_method="uniform",
        contrast_method="distance_linear",
        loss_cluster=0.0,
        loss_sigreg=0.0,
        loss_centers=0.5,
        loss_batch=0.25,
    )
    losses, _, _ = space.process_activations("layer_0", torch.randn(12, 20))

    assert "loss_global" in losses
    assert "loss_centers" in losses
    assert "loss_batch" in losses
    assert "loss_cluster" not in losses
    assert "loss_sigreg" not in losses


def test_layerspace_conv_activations_and_dynamic_unit_mask():
    torch.manual_seed(2)
    cfg = LayerSpaceConfig(modular_space_scoring="top_k", modular_space_reducer="conv_l2")
    space = LayerSpace(cfg, _make_observed_layers(16, layer_type="conv"), device=torch.device("cpu"))

    unit_mask = torch.zeros(16)
    unit_mask[:10] = 1.0
    fmaps = torch.randn(8, 16, 5, 5)

    losses, positions, _ = space.process_activations("layer_0", fmaps, unit_mask=unit_mask)
    assert torch.isfinite(torch.tensor(float(losses["loss"])))
    assert positions.shape == (16, cfg.modular_space_size)


def test_softmax_ce_separates_synthetic_modules():
    torch.manual_seed(0)
    N, D, bs = 60, 8, 64
    groups = torch.arange(N) // 10  # 6 modules of 10 units
    on_mod = torch.stack([torch.randperm(6)[:2] for _ in range(bs)])
    fires = (groups[None, :, None] == on_mod[:, None, :]).any(-1).float()
    scores = fires * torch.rand(bs, N) - (1.0 - fires) * torch.rand(bs, N)

    cfg = LayerSpaceConfig.from_dict(
        get_preset("softmax_ce", modular_space_loss_sigreg=0.0, modular_space_lr=0.02, modular_space_size=D)
    )
    space = LayerSpace(cfg, _make_observed_layers(N), device=torch.device("cpu"))
    for _ in range(80):
        _, positions, _ = space.process_scores("layer_0", scores, clusters_labels=groups)

    P = F.normalize(positions, dim=-1)
    sim = P @ P.T
    same = (groups[:, None] == groups[None, :]).float()
    within = float((sim * same).sum() / same.sum())
    between = float((sim * (1.0 - same)).sum() / (1.0 - same).sum())
    assert within > between + 0.3
