"""Unit tests for the unified SOTA ModelLoader (Csordas, Lepori, NeuroSurgeon)."""

from __future__ import annotations

import pytest
import torch

from positronixlab.metamodel.sota import (
    CVRResNetModel,
    ModelLoader,
    OddOneOutHead,
    is_neurosurgeon_available,
    strip_csordas_mask_keys,
    strip_lepori_l0_keys,
    strip_neurosurgeon_mask_keys,
)


def test_registry_lists_csordas_and_lepori_models():
    all_models = ModelLoader.list_models()
    assert "csordas:mlp" in all_models
    assert "csordas:conv" in all_models
    assert "csordas:lstm" in all_models
    assert "csordas:resnet110" in all_models
    assert "csordas:transformer" in all_models
    assert "lepori:cvr_resnet18" in all_models
    assert "lepori:cvr_resnet50" in all_models
    assert "lepori:cvr_wideresnet50" in all_models
    assert "lepori:cvr_vit" in all_models
    assert "lepori:syneval_bert" in all_models

    assert all(m.startswith("csordas:") for m in ModelLoader.list_models(source="csordas"))
    assert all(m.startswith("lepori:") for m in ModelLoader.list_models(source="lepori"))

    with pytest.raises(ValueError, match="Unknown architecture"):
        ModelLoader.create("unknown:arch")


def test_csordas_models_forward_and_maskable_layers():
    mlp = ModelLoader.create("csordas:mlp", n_inputs=16, n_outputs=4, layer_sizes=[32, 16], device="cpu")
    y = mlp(torch.randn(3, 16))
    assert y.shape == (3, 4)
    assert mlp.maskable_layers() == ["layers.0", "layers.1", "layers.2"]

    conv = ModelLoader.create("csordas:conv", in_channels=3, out_channels=10, dropout=False, device="cpu")
    y_conv = conv(torch.randn(2, 3, 32, 32))
    assert y_conv.shape == (2, 10)
    assert "out_layer" in conv.maskable_layers()

    lstm = ModelLoader.create("csordas:lstm", n_inputs=10, n_outputs=5, state_size=16, n_layers=1, device="cpu")
    y_lstm = lstm(torch.randn(2, 6, 10))
    assert y_lstm.shape == (2, 6, 5)
    assert "output_projection" in lstm.maskable_layers()

    resnet = ModelLoader.create("csordas:resnet110", num_classes=10, num_blocks=(1, 1, 1), device="cpu")
    y_res = resnet(torch.randn(2, 3, 32, 32))
    assert y_res.shape == (2, 10)
    assert "linear" in resnet.maskable_layers()

    trafo = ModelLoader.create(
        "csordas:transformer",
        n_input_tokens=20,
        n_output_tokens=20,
        state_size=16,
        n_heads=2,
        n_encoder_layers=1,
        n_decoder_layers=1,
        device="cpu",
    )
    src = torch.randint(0, 20, (2, 5))
    tgt = torch.randint(0, 20, (2, 4))
    y_trafo = trafo(src, tgt)
    assert y_trafo.shape == (2, 4, 20)
    assert "output_map" in trafo.maskable_layers()


def test_lepori_cvr_resnet_vit_and_syneval_bert_forward():
    cvr_res = ModelLoader.create(
        "lepori:cvr_resnet18", mlp_hidden_dim=64, mlp_out_dim=16, device="cpu"
    )
    y_res = cvr_res(torch.randn(2, 4, 3, 64, 64))
    assert y_res.shape == (2, 4)
    assert "mlp.fc1" in cvr_res.maskable_layers()

    cvr_vit = ModelLoader.create(
        "lepori:cvr_vit",
        image_size=32,
        patch_size=16,
        num_hidden_layers=1,
        hidden_size=32,
        num_attention_heads=2,
        intermediate_size=64,
        mlp_hidden_dim=32,
        mlp_out_dim=16,
        device="cpu",
    )
    y_vit = cvr_vit(torch.randn(2, 4, 3, 32, 32))
    assert y_vit.shape == (2, 4)
    assert not any("embeddings" in n for n in cvr_vit.maskable_layers())

    bert = ModelLoader.create(
        "lepori:syneval_bert",
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        device="cpu",
    )
    stacked = torch.stack(
        [torch.randint(0, 100, (2, 4, 8)), torch.ones(2, 4, 8, dtype=torch.long)], dim=1
    )
    y_bert = bert(stacked)
    assert y_bert.shape == (2, 4)
    assert not any("embeddings" in n for n in bert.maskable_layers())


def test_ooo_head_identifies_outlier():
    head = OddOneOutHead()
    # 3 identical vectors + 1 orthogonal outlier at index 2
    same = torch.tensor([1.0, 0.0, 0.0, 0.0])
    outlier = torch.tensor([0.0, 1.0, 0.0, 0.0])
    emb = torch.stack([same, same, outlier, same], dim=0).unsqueeze(0)  # (1, 4, 4)
    logits = head(emb)
    assert logits.shape == (1, 4)
    assert int(logits.argmax(dim=1).item()) == 2


def test_checkpoint_loading_auto_detects_all_sota_formats(tmp_path):
    # 1. Csordas MaskedModel checkpoint
    base_mlp = ModelLoader.create("csordas:mlp", n_inputs=8, n_outputs=3, layer_sizes=[12], device="cpu")
    csordas_ckpt = {f"model.{k}": v.clone() for k, v in base_mlp.state_dict().items()}
    csordas_ckpt["masks.0.layers.0"] = torch.ones_like(base_mlp.layers[0].weight)
    csordas_ckpt["masks.0.layers.1"] = -torch.ones_like(base_mlp.layers[1].weight)

    loaded_mlp, csordas_mask = ModelLoader.load(
        "csordas:mlp",
        checkpoint=csordas_ckpt,
        return_mask=True,
        n_inputs=8,
        n_outputs=3,
        layer_sizes=[12],
        device="cpu",
    )
    assert torch.allclose(loaded_mlp.layers[0].weight, base_mlp.layers[0].weight)
    assert csordas_mask is not None
    assert csordas_mask.masks["layers.0"].all()
    assert not csordas_mask.masks["layers.1"].any()

    # 2. Lepori split backbone + mlp checkpoints with *.mask_weight
    cvr = ModelLoader.create("lepori:cvr_resnet18", mlp_hidden_dim=32, mlp_out_dim=16, device="cpu")
    assert isinstance(cvr, CVRResNetModel)
    bb_state = dict(cvr.backbone.state_dict())
    bb_state["conv1.mask_weight"] = torch.ones_like(cvr.backbone.conv1.weight)
    mlp_state = dict(cvr.mlp.state_dict())
    mlp_state["fc1.mask_weight"] = -torch.ones_like(cvr.mlp.fc1.weight)

    bb_path = tmp_path / "run_backbone.pt"
    mlp_path = tmp_path / "run_mlp.pt"
    torch.save(bb_state, bb_path)
    torch.save(mlp_state, mlp_path)

    loaded_cvr, lepori_mask = ModelLoader.load(
        "lepori:cvr_resnet18",
        backbone_checkpoint=bb_path,
        mlp_checkpoint=mlp_path,
        return_mask=True,
        mlp_hidden_dim=32,
        mlp_out_dim=16,
        device="cpu",
    )
    assert torch.allclose(loaded_cvr.backbone.conv1.weight, cvr.backbone.conv1.weight)
    assert lepori_mask is not None
    assert lepori_mask.masks["backbone.conv1"].all()
    assert not lepori_mask.masks["mlp.fc1"].any()

    # 3. NeuroSurgeon CircuitModel checkpoint
    ns_ckpt = {f"circuit_model.wrapped_model.model.{k}": v.clone() for k, v in base_mlp.state_dict().items()}
    ns_ckpt["circuit_model.wrapped_model.model.layers.0.weight_mask_params"] = torch.full_like(
        base_mlp.layers[0].weight, 1.5
    )
    loaded_ns, ns_mask = ModelLoader.load(
        "csordas:mlp",
        checkpoint=ns_ckpt,
        return_mask=True,
        n_inputs=8,
        n_outputs=3,
        layer_sizes=[12],
        device="cpu",
    )
    assert torch.allclose(loaded_ns.layers[0].weight, base_mlp.layers[0].weight)
    assert ns_mask is not None
    assert ns_mask.masks["layers.0"].all()


@pytest.mark.skipif(not is_neurosurgeon_available(), reason="NeuroSurgeon not accessible in sandbox")
def test_neurosurgeon_wrapper_roundtrip():
    mlp = ModelLoader.create("csordas:mlp", n_inputs=8, n_outputs=3, layer_sizes=[12], device="cpu")
    orig_w = mlp.layers[0].weight.detach().clone()

    wrapped = ModelLoader.wrap_neurosurgeon(
        mlp,
        mask_method="continuous_sparsification",
        mask_unit="weight",
        deepcopy_model=True,
    )
    out = wrapped(torch.randn(4, 8))
    assert out.shape == (4, 3)
    assert len(wrapped.mask_parameters()) == 2
    assert wrapped.l0_loss().ndim == 0

    mask = wrapped.extract_subnetwork_mask(name="ns_test")
    assert mask.granularity == "weight"
    assert set(mask.masks.keys()) == {"layers.0", "layers.1"}
    # Original model untouched
    assert torch.equal(mlp.layers[0].weight, orig_w)
