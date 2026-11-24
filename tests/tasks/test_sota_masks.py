"""Unit tests for SubnetworkMask, MaskApplier, and SOTA discoverers."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from positronixlab.metamodel.sota import ModelLoader, is_neurosurgeon_available
from positronixlab.tasks.sota import (
    CsordasGumbelDiscoverer,
    MaskApplier,
    NeuroSurgeonDiscoverer,
    RandomMaskDiscoverer,
    SubnetworkMask,
)


def _make_tiny_mlp():
    torch.manual_seed(0)
    return ModelLoader.create(
        "csordas:mlp", n_inputs=8, n_outputs=3, layer_sizes=[12], device="cpu"
    )


def _make_tiny_loader():
    torch.manual_seed(0)
    x = torch.randn(32, 8)
    y = (x[:, 0] > 0).long() + (x[:, 1] > 0).long()
    return DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False)


def test_subnetwork_mask_projection_set_algebra_and_serialization(tmp_path):
    mlp = _make_tiny_mlp()
    unit_a = SubnetworkMask(
        name="atlas_a",
        granularity="unit",
        masks={
            "layers.0": torch.tensor([True] * 6 + [False] * 6),
            "layers.1": torch.tensor([True, True, False]),
        },
    )
    unit_b = SubnetworkMask(
        name="atlas_b",
        granularity="unit",
        masks={
            "layers.0": torch.tensor([False] * 3 + [True] * 6 + [False] * 3),
            "layers.1": torch.tensor([False, True, True]),
        },
    )

    # Set algebra & structural metrics
    inter = unit_a & unit_b
    union = unit_a | unit_b
    diff = unit_a - unit_b
    assert inter.l0() == 4  # 3 in layers.0 + 1 in layers.1
    assert union.l0() == 12  # 9 in layers.0 + 3 in layers.1
    assert diff.l0() == 4  # 3 in layers.0 + 1 in layers.1
    assert pytest.approx(unit_a.iou(unit_b), rel=1e-6) == 4 / 12
    assert pytest.approx(unit_a.overlap_fraction(unit_b), rel=1e-6) == 4 / 8
    assert pytest.approx(unit_a.sparsity(), rel=1e-6) == 1.0 - (8 / 15)

    # Unit -> Weight -> Unit round-trip
    weight_a = unit_a.to_weight_mask(mlp)
    assert weight_a.granularity == "weight"
    assert weight_a.masks["layers.0"].shape == mlp.layers[0].weight.shape
    assert weight_a.masks["layers.1"].shape == mlp.layers[1].weight.shape
    back_to_unit = weight_a.to_unit_mask()
    assert torch.equal(back_to_unit.masks["layers.0"], unit_a.masks["layers.0"])
    assert torch.equal(back_to_unit.masks["layers.1"], unit_a.masks["layers.1"])

    # Serialization round-trip
    saved = unit_a.save(tmp_path / "mask_a.pt")
    loaded = SubnetworkMask.load(saved)
    assert loaded.name == unit_a.name
    assert loaded.granularity == unit_a.granularity
    assert torch.equal(loaded.masks["layers.0"], unit_a.masks["layers.0"])


@pytest.mark.parametrize(
    "mode",
    ["retain", "zero_ablate", "random_ablate", "randomly_sampled", "complement_sampled"],
)
def test_mask_applier_modes_and_exact_restoration(mode):
    mlp = _make_tiny_mlp()
    orig_w0 = mlp.layers[0].weight.detach().clone()
    orig_w1 = mlp.layers[1].weight.detach().clone()

    # Keep density <= 0.5 so complement_sampled is valid
    unit_mask = SubnetworkMask(
        name="half",
        granularity="unit",
        masks={
            "layers.0": torch.tensor([True] * 4 + [False] * 8),
            "layers.1": torch.tensor([True, False, False]),
        },
    )

    with MaskApplier(mlp, unit_mask, mode=mode, seed=123):
        w0 = mlp.layers[0].weight.detach()
        if mode == "retain":
            assert torch.equal(w0[:4], orig_w0[:4])
            assert (w0[4:] == 0).all()
        elif mode == "zero_ablate":
            assert (w0[:4] == 0).all()
            assert torch.equal(w0[4:], orig_w0[4:])
        elif mode == "random_ablate":
            assert not torch.equal(w0[:4], orig_w0[:4])
            assert torch.equal(w0[4:], orig_w0[4:])
        elif mode == "randomly_sampled":
            assert int((w0 == 0).sum().item()) == 4 * 8
        elif mode == "complement_sampled":
            assert torch.equal(w0[:4], orig_w0[:4])
            assert int((w0[4:] == 0).sum().item()) == 4 * 8

    # Guaranteed exact restoration after exit
    assert torch.equal(mlp.layers[0].weight, orig_w0)
    assert torch.equal(mlp.layers[1].weight, orig_w1)


def test_mask_applier_restores_on_exception():
    mlp = _make_tiny_mlp()
    orig_w0 = mlp.layers[0].weight.detach().clone()
    mask = RandomMaskDiscoverer(density=0.5, seed=0).discover(mlp)

    with pytest.raises(RuntimeError, match="boom"):
        with MaskApplier(mlp, mask, mode="zero_ablate"):
            raise RuntimeError("boom")

    assert torch.equal(mlp.layers[0].weight, orig_w0)


def test_csordas_gumbel_discoverer_runs_without_mutating_model():
    mlp = _make_tiny_mlp()
    orig_w0 = mlp.layers[0].weight.detach().clone()
    loader = _make_tiny_loader()

    discoverer = CsordasGumbelDiscoverer(mask_loss_weight=1e-2, lr=5e-2, epochs=2, init_logit=1.0)
    mask = discoverer.discover(mlp, loader, mask_name="csordas_sub")

    assert mask.granularity == "weight"
    assert set(mask.masks.keys()) == {"layers.0", "layers.1"}
    assert mask.scores is not None
    assert torch.equal(mlp.layers[0].weight, orig_w0)


@pytest.mark.skipif(not is_neurosurgeon_available(), reason="NeuroSurgeon not accessible in sandbox")
def test_neurosurgeon_discoverer_continuous_sparsification_and_neuron_mode():
    mlp = _make_tiny_mlp()
    orig_w0 = mlp.layers[0].weight.detach().clone()
    loader = _make_tiny_loader()

    ns_weight = NeuroSurgeonDiscoverer(
        mask_method="continuous_sparsification",
        mask_unit="weight",
        l0_lambda=1e-3,
        lr=5e-2,
        epochs=2,
        max_temp=10.0,
    )
    w_mask = ns_weight.discover(mlp, loader, mask_name="ns_weight")
    assert w_mask.granularity == "weight"
    assert set(w_mask.masks.keys()) == {"layers.0", "layers.1"}

    ns_neuron = NeuroSurgeonDiscoverer(
        mask_method="continuous_sparsification",
        mask_unit="neuron",
        l0_lambda=1e-3,
        lr=5e-2,
        epochs=2,
    )
    u_mask = ns_neuron.discover(mlp, loader, mask_name="ns_neuron")
    assert u_mask.granularity == "unit"
    assert u_mask.masks["layers.0"].shape == (12,)
    assert u_mask.masks["layers.1"].shape == (3,)
    assert torch.equal(mlp.layers[0].weight, orig_w0)
