"""Unit tests for AtlasDiscoverer (LayerSpace + ModularClusterKmeans -> SubnetworkMask)."""

from __future__ import annotations

import torch

from positronixlab.baselines import DataProviderLoader, ModelLoader
from positronixlab.tasks import AtlasDiscoverer, MaskApplier


def test_atlas_discoverer_mlp_and_subtasks_no_hook_leaks():
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
    orig_w0 = model.layers[0].weight.detach().clone()

    atlas = AtlasDiscoverer(atlas_epochs=2, space_dim=6, k_min=2, k_max=4)
    mask = atlas.discover(model, dp.val_loader(), mask_name="atlas_full")

    assert mask.granularity == "unit"
    assert set(mask.masks.keys()) == {"layers.0", "layers.1", "layers.2"}
    assert mask.masks["layers.0"].shape == (24,)
    assert mask.masks["layers.1"].shape == (16,)
    assert mask.l0() > 0
    assert mask.scores is not None

    # Verify zero forward hooks leaked onto any module
    for mod in model.modules():
        assert len(mod._forward_hooks) == 0

    # Verify shared-Atlas multi-subtask discovery
    subtask_masks = atlas.discover_subtasks(
        model,
        reference_dataloader=dp.val_loader(),
        subtask_dataloaders={
            "add": dp.get_subtask("add").val_loader(),
            "mul": dp.get_subtask("mul").val_loader(),
        },
    )
    assert set(subtask_masks.keys()) == {"add", "mul"}
    assert 0.0 <= subtask_masks["add"].iou(subtask_masks["mul"]) <= 1.0

    # Verify unit mask works seamlessly inside MaskApplier
    with MaskApplier(model, subtask_masks["add"], mode="zero_ablate"):
        out = model(torch.randn(4, dp.num_features))
        assert out.shape == (4, dp.num_classes)
    assert torch.equal(model.layers[0].weight, orig_w0)


def test_atlas_discoverer_supports_conv2d_models():
    torch.manual_seed(0)
    conv = ModelLoader.create(
        "csordas:conv", in_channels=3, out_channels=4, dropout=False, device="cpu"
    )
    x = torch.randn(16, 3, 16, 16)
    y = torch.randint(0, 4, (16,))
    loader = [(x[:8], y[:8]), (x[8:], y[8:])]

    atlas = AtlasDiscoverer(atlas_epochs=1, space_dim=4, k_min=2, k_max=3)
    mask = atlas.discover(conv, loader, target_layers=["features.0", "out_layer"])
    assert mask.granularity == "unit"
    assert mask.masks["features.0"].shape == (32,)
    assert mask.masks["out_layer"].shape == (4,)

    # Broadcasts cleanly to 4D Conv2d weights
    w_mask = mask.to_weight_mask(conv)
    assert w_mask.masks["features.0"].shape == conv.features[0].weight.shape


def test_reduce_activation_to_2d_conv1d_and_conv3d():
    import torch.nn as nn
    from positronixlab.tasks.atlas_discoverer import _reduce_activation_to_2d

    conv1d = nn.Conv1d(3, 5, 3)
    out1d = conv1d(torch.randn(2, 3, 16))  # (2, 5, 14), C=5 != L=14
    assert _reduce_activation_to_2d(out1d, module=conv1d).shape == (2, 5)

    conv3d = nn.Conv3d(2, 6, 3)
    out3d = conv3d(torch.randn(2, 2, 4, 4, 4))  # (2, 6, 2, 2, 2)
    assert _reduce_activation_to_2d(out3d, module=conv3d).shape == (2, 6)

    # Sequence tensor without conv module still averages over T (dim=1)
    seq = torch.randn(2, 10, 8)
    assert _reduce_activation_to_2d(seq).shape == (2, 8)


def test_atlas_discoverer_supports_conv1d_models():
    import torch.nn as nn

    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv1d(3, 6, kernel_size=3),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(6, 4),
    )
    x = torch.randn(16, 3, 12)  # L=12 != C=6
    y = torch.randint(0, 4, (16,))
    loader = [(x[:8], y[:8]), (x[8:], y[8:])]

    atlas = AtlasDiscoverer(atlas_epochs=1, space_dim=4, k_min=2, k_max=3)
    mask = atlas.discover(model, loader, target_layers=["0", "4"])
    assert mask.masks["0"].shape == (6,)
    assert mask.masks["4"].shape == (4,)


def test_atlas_discoverer_rejects_exhausted_and_empty_loaders():
    import pytest

    torch.manual_seed(0)
    model = ModelLoader.create(
        "csordas:mlp", n_inputs=8, n_outputs=3, layer_sizes=[8], device="cpu"
    )
    x = torch.randn(8, 8)
    y = torch.randint(0, 3, (8,))
    loader = [(x[:4], y[:4]), (x[4:], y[4:])]

    atlas = AtlasDiscoverer(atlas_epochs=1, space_dim=4, k_min=2, k_max=3)

    # One-shot generator is rejected upfront
    gen = (b for b in loader)
    with pytest.raises(ValueError, match="re-iterable"):
        atlas.discover(model, gen)

    # Empty loader raises instead of returning a silent empty SubnetworkMask
    with pytest.raises(ValueError, match="no samples"):
        atlas.discover(model, [])


def test_atlas_discoverer_differential_attribution_avoids_identical_subtask_masks():
    """Regression test (B2): selectivity/contrastive attribution separates subtasks where raw energy collapses."""
    clusters = {"fc": torch.tensor([0, 0, 1, 1], dtype=torch.long)}
    # Cluster 0 has higher raw magnitude on BOTH subtasks, but subtask B boosts cluster 1 relative to ref
    ref_energy = {"fc": torch.tensor([10.0, 10.0, 2.0, 2.0])}
    subtask_energies = {
        "a": {"fc": torch.tensor([12.0, 12.0, 1.0, 1.0])},
        "b": {"fc": torch.tensor([8.0, 8.0, 4.0, 4.0])},
    }

    raw_masks = AtlasDiscoverer(
        top_cluster_fraction=0.5, attribution_mode="raw"
    ).attribute_from_unit_energy(clusters, subtask_energies, reference_energy=ref_energy)
    assert raw_masks["a"].iou(raw_masks["b"]) == 1.0  # raw collapses to identical masks

    sel_masks = AtlasDiscoverer(
        top_cluster_fraction=0.5, attribution_mode="selectivity"
    ).attribute_from_unit_energy(clusters, subtask_energies, reference_energy=ref_energy)
    assert sel_masks["a"].iou(sel_masks["b"]) == 0.0
    assert sel_masks["a"].metadata["pairwise_iou"]["b"]["overall"] == 0.0
    assert sel_masks["a"].metadata["attribution_mode"] == "selectivity"

    con_masks = AtlasDiscoverer(
        top_cluster_fraction=0.5, attribution_mode="contrastive"
    ).attribute_from_unit_energy(clusters, subtask_energies)
    assert con_masks["a"].iou(con_masks["b"]) == 0.0
    assert con_masks["b"].metadata["pairwise_iou"]["a"]["overall"] == 0.0
