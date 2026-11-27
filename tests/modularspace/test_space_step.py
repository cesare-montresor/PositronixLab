import pytest
import torch

from positronixlab.modularspace.config import LayerSpaceConfig
from positronixlab.modularspace.layerspace import LayerSpace
from positronixlab.modularspace.presets import get_preset, get_preset_spec, list_presets
from positronixlab.modularspace.space_step import SpaceStep, SpaceStepSpec


@pytest.mark.parametrize("preset_name", list_presets())
def test_space_step_runs_all_presets_and_backprops(preset_name):
    torch.manual_seed(0)
    step = SpaceStep(get_preset_spec(preset_name))
    positions = torch.randn(20, 6, requires_grad=True)
    scores = step.score_activations(torch.randn(12, 20), layer_type="linear")
    labels = torch.arange(20) % 4

    loss, metrics = step.compute_loss(positions, scores, clusters_labels=labels)
    assert torch.isfinite(loss)
    assert "loss" in metrics
    loss.backward()
    assert positions.grad is not None
    assert torch.isfinite(positions.grad).all()


def test_space_step_cluster_single_cluster_skips_cluster_contrast():
    torch.manual_seed(1)
    spec = SpaceStepSpec(loss_global=1.0, loss_cluster=1.0, loss_sigreg=0.0)
    step = SpaceStep(spec)
    positions = torch.randn(12, 4, requires_grad=True)
    scores = step.score_activations(torch.randn(8, 12))
    # Only 1 cluster (K < 2) -> cluster contrast returns 0
    labels = torch.zeros(12, dtype=torch.long)

    loss, metrics = step.compute_loss(positions, scores, clusters_labels=labels)
    assert float(metrics["loss_cluster"]) == pytest.approx(0.0)
    assert "frac_mixed_cluster" not in metrics
    assert torch.isfinite(loss)


def test_space_step_invalid_sigreg_target_raises():
    spec = SpaceStepSpec(loss_sigreg=1.0, sigreg_target="invalid_target")
    step = SpaceStep(spec)
    positions = torch.randn(10, 4, requires_grad=True)
    scores = step.score_activations(torch.randn(6, 10))
    with pytest.raises(ValueError, match="Unknown sigreg_target"):
        step.compute_loss(positions, scores)


def test_layerspace_invalid_sigreg_target_raises_eagerly():
    with pytest.raises(ValueError, match="Unknown sigreg_target"):
        LayerSpace(
            LayerSpaceConfig(),
            {"l0": {"out_units": 8, "layer_type": "linear"}},
            device=torch.device("cpu"),
            sigreg_target="invalid_target",
        )


def test_space_step_matches_layerspace_compute_loss():
    torch.manual_seed(7)
    cfg = LayerSpaceConfig.from_dict(get_preset("v1_centers"))
    spec = SpaceStepSpec.from_config(cfg)
    step = SpaceStep(spec)
    space = LayerSpace(
        cfg,
        {"l0": {"out_units": 16, "layer_type": "linear"}},
        device=torch.device("cpu"),
    )
    positions = space.positions["l0"].detach().clone().requires_grad_(True)
    scores = step.score_activations(torch.randn(10, 16))

    loss_step, m_step = step.compute_loss(positions, scores)
    loss_ls, m_ls = space.compute_loss(positions, scores)
    assert float(loss_step) == pytest.approx(float(loss_ls), rel=1e-5, abs=1e-6)
    assert set(m_step.keys()) == set(m_ls.keys())


def test_layerspace_all_zero_losses_does_not_crash():
    cfg = LayerSpaceConfig(
        modular_space_loss_global=0.0,
        modular_space_loss_cluster=0.0,
        modular_space_loss_centers=0.0,
        modular_space_loss_batch=0.0,
        modular_space_loss_sigreg=0.0,
    )
    space = LayerSpace(
        cfg,
        {"l0": {"out_units": 8, "layer_type": "linear"}},
        device=torch.device("cpu"),
    )
    losses, positions, _ = space.process_activations("l0", torch.randn(4, 8))
    assert float(losses["loss"]) == pytest.approx(0.0)
    assert positions.shape == (8, cfg.modular_space_size)
