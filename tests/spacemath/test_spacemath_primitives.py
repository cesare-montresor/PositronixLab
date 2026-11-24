import math
import pytest
import torch
from torch import nn

from positronixlab.spacemath import (
    CONTRASTS,
    LOSSES,
    REDUCERS,
    SCORERS,
    WEIGHTERS,
    SpaceType,
    activations_to_scores,
    distance_elementwise,
    distance_pairwise,
    get_contrast,
    get_reducer,
    get_scorer,
    get_weighter,
    loss_batch_repulsion,
    loss_centers_repulsion,
    loss_layer_alignment,
    loss_sigreg,
    loss_sigreg_centers,
    mask_cluster_membership,
    mask_global_scope,
    space_normalize,
    space_sigreg_prepare,
    stats_update_,
    weight_raw,
)


def test_lookup_helpers_reject_unknown_names():
    with pytest.raises(KeyError):
        get_reducer("missing")
    with pytest.raises(KeyError):
        get_scorer("missing")
    with pytest.raises(KeyError):
        get_weighter("missing")
    with pytest.raises(KeyError):
        get_contrast("missing")


@pytest.mark.parametrize("space_type", [SpaceType.COSINE, SpaceType.EUCLIDEAN])
def test_space_type_distances_and_sigreg_prepare(space_type):
    torch.manual_seed(0)
    x2 = torch.randn(4, 6)
    y2 = torch.randn(8, 6)
    x3 = torch.randn(3, 2, 6)

    d_el = distance_elementwise(x2, x2, space_type=space_type)
    assert d_el.shape == (4,)
    assert torch.allclose(d_el, torch.zeros_like(d_el), atol=1e-3)

    d_pw2 = distance_pairwise(x2, y2, space_type=space_type)
    assert d_pw2.shape == (4, 8)
    assert (d_pw2 >= 0).all()

    d_pw3 = distance_pairwise(x3, y2, space_type=space_type)
    assert d_pw3.shape == (3, 2, 8)
    assert (d_pw3 >= 0).all()

    prep = space_sigreg_prepare(y2, space_type=space_type)
    assert prep.shape == y2.shape
    if space_type == SpaceType.COSINE:
        norms = prep.norm(dim=-1)
        assert torch.allclose(norms, torch.full_like(norms, math.sqrt(6)), atol=1e-5)
        unit = space_normalize(y2, space_type=space_type)
        assert torch.allclose(unit.norm(dim=-1), torch.ones(8), atol=1e-5)


@pytest.mark.parametrize("score_method", sorted(SCORERS))
def test_scoring_methods_linear(score_method):
    torch.manual_seed(1)
    act = torch.randn(16, 32)
    scores = activations_to_scores(act, layer_type="linear", score_method=score_method, active_q=0.75)
    assert scores.shape == (16, 32)
    assert torch.isfinite(scores).all()
    assert scores.min() >= -1.0 - 1e-5
    assert scores.max() <= 1.0 + 1e-5


@pytest.mark.parametrize("reducer", [r for r in sorted(REDUCERS) if r.startswith("conv_")])
def test_scoring_conv_reducers(reducer):
    torch.manual_seed(2)
    fmap = torch.randn(8, 12, 7, 7)
    scores = activations_to_scores(
        fmap,
        layer_type="conv",
        reducer=reducer,
        score_method="top_k",
        active_q=0.75,
        activation_fn=nn.ReLU(),
        apply_activation=True,
    )
    assert scores.shape == (8, 12)
    assert torch.isfinite(scores).all()


@pytest.mark.parametrize("weight_method", sorted(WEIGHTERS))
def test_weighting_methods_and_stats(weight_method):
    torch.manual_seed(3)
    scores = torch.randn(10, 16).clamp(-1.0, 1.0)
    stats = torch.zeros(16, 4)
    stats_update_(stats, scores)
    assert (stats[:, 0] + stats[:, 1]).eq(10).all()

    w_on, w_off = get_weighter(weight_method)(scores, stats=stats, freq_alpha=0.5)
    assert w_on.shape == (10, 16)
    assert w_off.shape == (10, 16)
    assert (w_on >= 0).all() and (w_off >= 0).all()


def test_masks_support_noise_and_unit_mask():
    unit_mask = torch.tensor([1, 1, 1, 1, 0, 0], dtype=torch.float32)
    g_mask = mask_global_scope(6, torch.device("cpu"), torch.float32, unit_mask=unit_mask)
    assert g_mask.shape == (1, 6)
    assert g_mask[0, 4:].sum() == 0.0

    labels = torch.tensor([0, 0, 1, -1, 1, 0])
    m = mask_cluster_membership(
        labels, num_units=6, device=torch.device("cpu"), dtype=torch.float32, unit_mask=unit_mask
    )
    assert m.shape == (2, 6)
    assert m[:, 3].sum() == 0.0
    assert m[:, 4:].sum() == 0.0

    assert mask_cluster_membership(
        torch.zeros(6, dtype=torch.long), device=torch.device("cpu"), dtype=torch.float32
    ) is None


@pytest.mark.parametrize("contrast_method", sorted(CONTRASTS))
@pytest.mark.parametrize("space_type", [SpaceType.COSINE, SpaceType.EUCLIDEAN])
def test_contrast_methods_forward_and_backward(contrast_method, space_type):
    torch.manual_seed(4)
    U, D, bs = 24, 6, 12
    positions = torch.randn(U, D, requires_grad=True)
    scores = torch.randn(bs, U).clamp(-1.0, 1.0)
    w_on, w_off = weight_raw(scores)
    labels = torch.arange(U) % 3
    m = mask_cluster_membership(labels, num_units=U, device=positions.device, dtype=positions.dtype)

    res = get_contrast(contrast_method)(
        positions, w_on, w_off, m, space_type=space_type, tau=0.1
    )
    assert torch.isfinite(res.loss)
    assert res.centers_on.shape == (bs, 3, D)
    assert res.centers_off.shape == (bs, 3, D)
    assert res.mixed.shape == (bs, 3)

    res.loss.backward()
    assert positions.grad is not None
    assert torch.isfinite(positions.grad).all()


@pytest.mark.parametrize("space_type", [SpaceType.COSINE, SpaceType.EUCLIDEAN])
def test_additional_losses(space_type):
    torch.manual_seed(5)
    centers_on = torch.randn(8, 2, 6, requires_grad=True)
    centers_off = torch.randn(8, 2, 6, requires_grad=True)
    mixed = torch.ones(8, 2)

    l_c = loss_centers_repulsion(centers_on, centers_off, mixed=mixed, space_type=space_type)
    l_b = loss_batch_repulsion(centers_on, mixed=mixed, space_type=space_type)
    l_sw = loss_sigreg(centers_on.reshape(-1, 6), space_type=space_type, method="weak")
    l_ss = loss_sigreg(centers_on.reshape(-1, 6), space_type=space_type, method="strong")
    l_sc = loss_sigreg_centers(centers_on, mixed, space_type=space_type, method="weak")
    l_al = loss_layer_alignment([centers_on[:, 0], centers_off[:, 0]], space_type=space_type)

    total = l_c + l_b + l_sw + l_ss + l_sc + l_al
    assert torch.isfinite(total)
    total.backward()
    assert torch.isfinite(centers_on.grad).all()
    assert {"centers", "batch", "sigreg", "sigreg_centers", "alignment"}.issubset(LOSSES)
