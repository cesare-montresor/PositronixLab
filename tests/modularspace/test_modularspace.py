import pytest
import torch

pytest.importorskip(
    "parallelazy",
    reason="ModularSpace imports PipelineConfig at top-level (modularspace.py:8), which requires parallelazy",
)

from positronixlab.modularspace import ModularSpace
from positronixlab.pipeline import PipelineConfigV1


def _make_layers(num_cus: int = 16):
    return {
        "layer_0": {
            "out_units": num_cus,
            "layer_type": "linear",
            "activation_type": "relu",
        }
    }


@pytest.mark.parametrize("distance", ["cosine", "euclidean"])
def test_modularspace_process_activations_with_stats_enabled(distance):
    torch.manual_seed(0)
    cfg = PipelineConfigV1(
        modular_space_size=4,
        modular_space_lr=0.01,
        modular_space_distance=distance,
    )
    space = ModularSpace(
        cfg,
        _make_layers(16),
        device=torch.device("cpu"),
        stats_enabled=True,
    )
    losses, positions, stats = space.process_activations("layer_0", torch.randn(8, 16))
    assert set(losses.keys()) == {"loss", "loss_on", "loss_off", "loss_centers"}
    assert torch.isfinite(torch.tensor(float(losses["loss"])))
    assert positions.shape == (16, 4)
    assert stats is not None and stats.shape == (16, 4)


def test_modularspace_unknown_layer_raises():
    cfg = PipelineConfigV1()
    space = ModularSpace(cfg, _make_layers(8), device=torch.device("cpu"), stats_enabled=True)
    with pytest.raises(ValueError, match="not found in ModularSpace"):
        space.get_space("missing_layer")


@pytest.mark.xfail(
    strict=True,
    reason="Bug B1: ModularSpace.get_space checks `None in (layer_info, positions, optimizer, stats)` and raises ValueError when stats_enabled=False (modularspace.py:59)",
)
def test_modularspace_works_with_stats_disabled():
    cfg = PipelineConfigV1(modular_space_size=4)
    space = ModularSpace(
        cfg,
        _make_layers(12),
        device=torch.device("cpu"),
        stats_enabled=False,
    )
    losses, positions, stats = space.process_activations("layer_0", torch.randn(6, 12))
    assert torch.isfinite(torch.tensor(float(losses["loss"])))
    assert positions.shape == (12, 4)
    assert stats is None
