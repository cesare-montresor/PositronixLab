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
