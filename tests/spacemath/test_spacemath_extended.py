import pytest
import torch
from torch import nn

from positronixlab.spacemath import (
    ActivationType,
    LayerType,
    SpaceMath,
    SpaceType,
    contrast_distance_linear,
    contrast_distance_log,
    contrast_softmax_ce,
    distance_elementwise,
    distance_pairwise,
    loss_batch_repulsion,
    loss_layer_alignment,
    loss_sigreg,
    loss_sigreg_centers,
    mask_cluster_membership,
    mask_global_scope,
    reduce_conv_l2,
    reduce_conv_locality,
    reduce_conv_max,
    reduce_conv_mean_abs,
    reduce_linear,
    score_quantile,
    score_sign_preserving_batch,
    score_sign_preserving_row,
    score_softmax,
    score_top_k,
    sigreg,
    sigreg_strong,
    sigreg_weak,
    space_is_cosine,
    space_normalize,
    space_sigreg_prepare,
    stats_activation_frequency,
    stats_update_,
    weight_inv_freq,
    weight_raw,
    weight_uniform,
)
from positronixlab.spacemath.common import (
    classmethod_compile,
    infer_shape,
    is_activation,
    is_conv,
    is_layer,
    is_linear,
    spow,
    spow_,
    staticmethod_compile,
    zclamp,
    zclamp_,
)

# ---------------------------------------------------------------------------
# common.py
# ---------------------------------------------------------------------------


def test_zclamp_and_spow_basic_and_monkeypatch():
    x = torch.tensor([-1.0, -1e-10, 0.0, 1e-10, 2.0])

    # Out-of-place zclamp with force_zero=False pushes near-zero non-zero values to +/-eps
    z_eps = zclamp(x, eps=1e-6, force_zero=False)
    assert torch.allclose(z_eps, torch.tensor([-1.0, -1e-6, 0.0, 1e-6, 2.0]))
    assert torch.equal(x, torch.tensor([-1.0, -1e-10, 0.0, 1e-10, 2.0]))

    # Out-of-place zclamp with force_zero=True zeroes near-zero values
    z_zero = zclamp(x, eps=1e-6, force_zero=True)
    assert torch.allclose(z_zero, torch.tensor([-1.0, 0.0, 0.0, 0.0, 2.0]))

    # Monkey-patched Tensor methods match free functions
    assert torch.equal(x.zclamp(eps=1e-6), z_eps)
    assert torch.equal(x.spow(2.0, eps=1e-6), spow(x, 2.0, eps=1e-6))

    # In-place variants mutate tensor
    y = x.clone()
    zclamp_(y, eps=1e-6, force_zero=True)
    assert torch.equal(y, z_zero)

    y2 = torch.tensor([1e-10, 2.0])
    spow_(y2, 2.0, eps=1e-6, force_zero=True)
    assert torch.allclose(y2, torch.tensor([0.0, 4.0]))


def test_module_predicates():
    assert is_linear(nn.Linear(4, 4))
    assert not is_linear(nn.Conv2d(3, 4, 3))
    assert is_conv(nn.Conv2d(3, 4, 3))
    assert is_conv(nn.Conv1d(3, 4, 3))
    assert is_activation(nn.ReLU())
    assert is_activation(nn.Sigmoid())
    assert not is_activation(nn.Linear(4, 4))
    assert is_layer(nn.Linear(4, 4))
    assert is_layer(nn.Conv2d(3, 4, 3))
    assert is_layer(nn.LSTM(4, 4))
    assert not is_layer(nn.ReLU())


def test_compile_method_decorators():
    class Helper:
        @staticmethod_compile
        def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a + b

        @classmethod_compile
        def scale(cls, a: torch.Tensor) -> torch.Tensor:
            return a * 2.0

    t = torch.tensor([1.0, 2.0])
    assert torch.allclose(Helper.add(t, t), torch.tensor([2.0, 4.0]))
    assert torch.allclose(Helper.scale(t), torch.tensor([2.0, 4.0]))


def test_clear_cache_docstring_safety_and_compile_env_switch(monkeypatch):
    from pathlib import Path

    from positronixlab.spacemath.common import (
        _is_safe_cache_dir,
        clear_torch_compile_cache,
        torch_compile,
    )

    assert clear_torch_compile_cache.__doc__ is not None
    assert "cache" in clear_torch_compile_cache.__doc__.lower()

    assert not _is_safe_cache_dir("/")
    assert not _is_safe_cache_dir("/tmp")
    assert not _is_safe_cache_dir(str(Path.home()))
    assert _is_safe_cache_dir("/tmp/torchinductor_someone")
    assert _is_safe_cache_dir("/tmp/triton_cache")
    assert _is_safe_cache_dir(str(Path.home() / ".triton" / "cache"))

    monkeypatch.setenv("POSITRONIXLAB_COMPILE", "0")

    def raw_fn(x: torch.Tensor) -> torch.Tensor:
        return x + 1.0

    assert torch_compile(raw_fn) is raw_fn


@pytest.mark.xfail(strict=True, reason="common.py:49 infer_shape has stray `self` parameter")
def test_bug_infer_shape_stray_self():
    x_in = (torch.randn(4, 8),)
    x_out = torch.randn(4, 16)
    bs, in_shape, out_shape = infer_shape(x_in, x_out)
    assert bs == 4 and in_shape == (8,) and out_shape == (16,)


def test_bug_is_linear_matches_identity():
    assert not is_linear(nn.Identity())


@pytest.mark.xfail(strict=True, reason="common.py:30 spow defaults to force_zero=True, producing inf for negative powers")
def test_bug_spow_negative_power_near_zero():
    out = spow(torch.tensor([1e-10]), -1.0)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# space_type.py
# ---------------------------------------------------------------------------


def test_space_type_normalize_name_and_errors():
    assert SpaceType.normalize_name("cosine") == SpaceType.COSINE
    assert SpaceType.normalize_name("  COS ") == SpaceType.COSINE
    assert SpaceType.normalize_name("euclidean") == SpaceType.EUCLIDEAN
    assert SpaceType.normalize_name("L2") == SpaceType.EUCLIDEAN
    assert SpaceType.normalize_name("euclid") == SpaceType.EUCLIDEAN
    assert space_is_cosine("cosine") is True
    assert space_is_cosine("l2") is False

    with pytest.raises(ValueError):
        SpaceType.normalize_name("manhattan")


def test_distance_pairwise_batched_3d():
    torch.manual_seed(10)
    x = torch.randn(4, 3, 5)
    y = torch.randn(4, 7, 5)
    for st in SpaceType.ALL:
        d = distance_pairwise(x, y, space_type=st)
        assert d.shape == (4, 3, 7)
        assert (d >= 0).all()
        assert torch.isfinite(d).all()


def test_space_normalize_euclidean_is_identity():
    x = torch.randn(5, 4) * 3.0
    assert torch.equal(space_normalize(x, space_type=SpaceType.EUCLIDEAN), x)
    assert torch.equal(space_sigreg_prepare(x, space_type=SpaceType.EUCLIDEAN), x)


@pytest.mark.xfail(
    strict=True,
    reason="space_type.py:70 Euclidean distance_elementwise(x, x) returns sqrt(eps)=1e-4 instead of 0.0",
)
def test_bug_distance_elementwise_euclidean_zero_floor():
    x = torch.randn(4, 6)
    d = distance_elementwise(x, x, space_type=SpaceType.EUCLIDEAN)
    assert torch.allclose(d, torch.zeros_like(d), atol=1e-6)


# ---------------------------------------------------------------------------
# reducers.py & scoring.py
# ---------------------------------------------------------------------------


def test_reducers_shapes_and_3d_linear():
    x3 = torch.randn(4, 3, 5)
    assert reduce_linear(x3).shape == (4, 5)

    # Conv1d / Conv3d work with l2, mean_abs, max
    c1 = torch.randn(4, 8, 10)
    assert reduce_conv_l2(c1).shape == (4, 8)
    assert reduce_conv_mean_abs(c1).shape == (4, 8)
    assert reduce_conv_max(c1).shape == (4, 8)

    c2 = torch.randn(4, 8, 5, 5)
    out_loc = reduce_conv_locality(c2, locality_weight=0.5, spatial_radius=3)
    assert out_loc.shape == (4, 8)
    assert torch.isfinite(out_loc).all()


def test_scoring_sign_preserving_and_softmax():
    x = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 4.0], [-4.0, 0.0, 0.0, 2.0, 2.0]])
    s_row = score_sign_preserving_row(x)
    assert torch.allclose(s_row[0], torch.tensor([-1.0, -0.5, 0.0, 0.25, 1.0]), atol=1e-6)
    assert torch.allclose(s_row[1], torch.tensor([-1.0, 0.0, 0.0, 1.0, 1.0]), atol=1e-6)

    s_batch = score_sign_preserving_batch(x)
    assert s_batch.min().item() == pytest.approx(-1.0, abs=1e-6)
    assert s_batch.max().item() == pytest.approx(1.0, abs=1e-6)

    s_sm = score_softmax(x)
    assert torch.equal(s_sm[0], torch.tensor([-1.0, -1.0, -1.0, -1.0, 1.0]))
    assert torch.equal(s_sm[1], torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0]))


def test_score_top_k_single_unit_edge_case():
    x = torch.tensor([[2.0], [5.0]])
    s = score_top_k(x, active_q=0.8)
    assert s.shape == (2, 1)
    assert torch.isfinite(s).all()


def test_bug_score_normalize_by_threshold_all_negative_row():
    values = torch.tensor([[-10.0, -5.0, -2.0, -1.0]])
    scores = score_quantile(values, active_q=0.5)
    assert scores.min() >= -1.0 - 1e-5
    assert scores.max() <= 1.0 + 1e-5
    # Top values (-2.0, -1.0) should be >= 0 (active)
    assert (scores[0, 2:] >= 0.0).all()


# ---------------------------------------------------------------------------
# weighting.py
# ---------------------------------------------------------------------------


def test_weighting_nonzero_threshold_and_stats_frequency():
    scores = torch.tensor([[-0.5, 0.1, 0.5, 0.9]])
    w_on, w_off = weight_raw(scores, threshold=0.2)
    assert torch.allclose(w_on, torch.tensor([[0.0, 0.0, 0.3, 0.7]]))
    assert torch.allclose(w_off, torch.tensor([[0.7, 0.1, 0.0, 0.0]]))

    u_on, u_off = weight_uniform(scores, threshold=0.2)
    assert torch.equal(u_on, torch.tensor([[0.0, 0.0, 1.0, 1.0]]))
    assert torch.equal(u_off, torch.tensor([[1.0, 1.0, 0.0, 0.0]]))

    stats = torch.zeros(4, 4)
    # Before any update, frequency is 0.0 (safe clamp_min)
    assert torch.equal(stats_activation_frequency(stats), torch.zeros(4))
    stats_update_(stats, scores)
    freq = stats_activation_frequency(stats)
    assert torch.allclose(freq, torch.tensor([0.0, 1.0, 1.0, 1.0]))

    # Explicit freq override works without stats
    w_on_if, _ = weight_inv_freq(scores, freq=torch.tensor([0.1, 0.1, 0.9, 0.9]), freq_alpha=0.5)
    assert w_on_if.shape == (1, 4)
    assert (w_on_if >= 0).all()


def test_mask_cluster_membership_edge_cases():
    dev = torch.device("cpu")
    assert mask_cluster_membership(None, device=dev, dtype=torch.float32) is None
    assert mask_cluster_membership(torch.tensor([], dtype=torch.long), device=dev, dtype=torch.float32) is None
    with pytest.raises(ValueError):
        mask_cluster_membership(torch.tensor([0, 1]), num_units=5, device=dev, dtype=torch.float32)


@pytest.mark.xfail(
    strict=True,
    reason="weighting.py:74 mask_cluster_membership uses max()+1 instead of unique non-negative cluster count",
)
def test_bug_mask_cluster_membership_single_nonzero_cluster_id():
    labels = torch.tensor([-1, -1, 2, 2])
    m = mask_cluster_membership(labels, num_units=4, device=torch.device("cpu"), dtype=torch.float32)
    assert m is None


# ---------------------------------------------------------------------------
# sigreg.py & losses.py
# ---------------------------------------------------------------------------


def test_sigreg_weak_and_strong_gaussian_vs_collapsed():
    torch.manual_seed(42)
    iso = torch.randn(512, 8)
    collapsed = torch.randn(512, 1).expand(512, 8) + 1e-4 * torch.randn(512, 8)

    assert sigreg_weak(iso, sketch_dim=64) < sigreg_weak(collapsed, sketch_dim=64)
    assert sigreg_strong(iso, sketch_dim=32) < sigreg_strong(collapsed, sketch_dim=32)

    # Sketching path when D > sketch_dim
    high_d = torch.randn(64, 32)
    l_sketched = sigreg(high_d, sketch_dim=8, method="weak")
    assert torch.isfinite(l_sketched) and l_sketched >= 0

    with pytest.raises(ValueError):
        sigreg(iso, method="invalid")


def test_losses_degenerate_and_small_inputs():
    x1 = torch.randn(1, 6)
    assert loss_sigreg(x1).item() == 0.0
    assert loss_batch_repulsion(x1).item() == 0.0

    centers = torch.randn(4, 2, 6)
    mixed_zero = torch.zeros(4, 2)
    assert loss_sigreg_centers(centers, mixed_zero).item() == 0.0
    assert loss_batch_repulsion(centers, mixed=mixed_zero).item() == 0.0

    # Single-layer list returns 0.0
    assert loss_layer_alignment([torch.randn(4, 6)]).item() == 0.0


@pytest.mark.xfail(strict=True, reason="losses.py:120 loss_layer_alignment indexes [0] on empty sequence")
def test_bug_loss_layer_alignment_empty_list():
    out = loss_layer_alignment([])
    assert out.item() == 0.0


@pytest.mark.xfail(strict=True, reason="losses.py:45-54 loss_batch_repulsion ignores `mixed` for 2D inputs")
def test_bug_loss_batch_repulsion_2d_ignores_mixed():
    pts = torch.randn(4, 6)
    mixed = torch.zeros(4)
    out = loss_batch_repulsion(pts, mixed=mixed, space_type=SpaceType.COSINE)
    assert out.item() == 0.0


# ---------------------------------------------------------------------------
# contrast.py edge cases
# ---------------------------------------------------------------------------


def test_contrast_all_inactive_or_all_active_is_finite_and_zero():
    torch.manual_seed(7)
    positions = torch.randn(8, 4, requires_grad=True)
    scope = mask_global_scope(8, positions.device, positions.dtype)
    w_on = torch.ones(4, 8)
    w_off = torch.zeros(4, 8)  # no inactive mass -> mixed == 0

    for fn in (contrast_distance_log, contrast_distance_linear, contrast_softmax_ce):
        res = fn(positions, w_on, w_off, scope, space_type=SpaceType.COSINE)
        assert torch.isfinite(res.loss)
        assert res.loss.item() == pytest.approx(0.0, abs=1e-6)
        assert res.mixed.sum().item() == 0.0


def test_bug_contrast_distance_cosine_not_scale_invariant():
    torch.manual_seed(8)
    positions = torch.randn(12, 4)
    scales = torch.linspace(0.1, 10.0, 12).unsqueeze(1)
    scaled_positions = positions * scales
    scores = torch.randn(6, 12).clamp(-1.0, 1.0)
    w_on, w_off = weight_raw(scores)
    scope = mask_global_scope(12, positions.device, positions.dtype)

    res1 = contrast_distance_log(positions, w_on, w_off, scope, space_type=SpaceType.COSINE)
    res2 = contrast_distance_log(scaled_positions, w_on, w_off, scope, space_type=SpaceType.COSINE)
    assert torch.allclose(res1.loss, res2.loss, atol=1e-5)


# ---------------------------------------------------------------------------
# spacemath.py (legacy v1)
# ---------------------------------------------------------------------------


def test_legacy_layer_and_activation_types():
    assert LayerType.get_layer_type(nn.Linear(4, 4)) == LayerType.LINEAR
    assert LayerType.get_layer_type(nn.Conv2d(3, 4, 3)) == LayerType.CONV
    assert LayerType.get_layer_type(nn.ReLU()) == "unknown"

    assert ActivationType.get_activation_type(nn.ReLU()) == ActivationType.RELU
    assert ActivationType.get_activation_type(nn.LeakyReLU()) == ActivationType.LEAKY_RELU
    assert ActivationType.get_activation_type(nn.Sigmoid()) == ActivationType.LOGISTIC
    assert ActivationType.get_activation_type(nn.Softmax(dim=1)) == ActivationType.SOFTMAX
    assert ActivationType.get_activation_type(nn.Tanh()) == "unknown"

    assert isinstance(ActivationType.get_function(ActivationType.RELU), nn.ReLU)
    assert isinstance(ActivationType.get_function(ActivationType.SOFTMAX), nn.Softmax)
    with pytest.raises(ValueError):
        ActivationType.get_function("gelu")


def test_legacy_spacemath_distance_and_linear_scoring():
    torch.manual_seed(9)
    x = torch.randn(4, 6)
    y = torch.randn(5, 6)

    d_cos_pw = SpaceMath.distance(x, y, distance_type="cosine", pairwise=True)
    assert d_cos_pw.shape == (4, 5)
    d_cos_el = SpaceMath.distance(x, x, distance_type="cosine", pairwise=False)
    assert d_cos_el.shape == (4,)
    assert torch.allclose(d_cos_el, torch.zeros(4), atol=1e-5)

    d_euc_pw = SpaceMath.distance(x, y, distance_type="euclidean", pairwise=True)
    assert d_euc_pw.shape == (4, 5)
    d_euc_el = SpaceMath.distance(x, x, distance_type="euclidean", pairwise=False)
    assert d_euc_el.shape == (4,)

    # Positive linear activations work with SpaceMath.activations_to_scores
    act = torch.relu(torch.randn(8, 16)) + 0.1
    scores = SpaceMath.activations_to_scores(act, layer_type=LayerType.LINEAR, x_quantile=0.75)
    assert scores.shape == (8, 16)
    assert scores.min() >= -1.0 - 1e-5 and scores.max() <= 1.0 + 1e-5

    # Softmax branch
    sm_scores = SpaceMath.activations_to_scores(act, layer_type=LayerType.LINEAR, activation_fn=nn.Softmax(dim=1))
    assert set(sm_scores.unique().tolist()).issubset({-1.0, 1.0})

    # count_active helper
    on, off = SpaceMath.split_active_group(scores)
    assert SpaceMath.count_active(on, off) is True


@pytest.mark.xfail(strict=True, reason="spacemath.py:112 unpacks (B, 1, k) tensor into `q, scores`")
def test_bug_legacy_spacemath_conv_activations_to_scores():
    fmap = torch.randn(4, 8, 5, 5)
    scores = SpaceMath.activations_to_scores(fmap, layer_type=LayerType.CONV, x_quantile=0.75)
    assert scores.shape == (4, 8)


@pytest.mark.xfail(
    strict=True,
    reason="spacemath.py:262 get_quantile_k_kernels_per_sample indexes sorted_indices with unsorted mask positions",
)
def test_bug_legacy_spacemath_get_quantile_k_kernels_mismatch():
    torch.manual_seed(11)
    fmap = torch.randn(3, 10, 5, 5)
    _, full_scores = SpaceMath.rank_kernels_by_activation_per_sample(fmap)
    top_idx, top_scores = SpaceMath.get_quantile_k_kernels_per_sample(fmap, quantile=0.7)
    # Every non-zero returned score must match full_scores at the returned kernel index
    for b in range(3):
        gathered = full_scores[b, top_idx[b]]
        assert torch.allclose(gathered, top_scores[b], atol=1e-5)


@pytest.mark.xfail(
    strict=True,
    reason="spacemath.py:85 SpaceMath.distance Euclidean pairwise calls .squeeze(), collapsing (1, N) to (N,)",
)
def test_bug_legacy_spacemath_euclidean_squeeze():
    x = torch.randn(1, 6)
    y = torch.randn(5, 6)
    d = SpaceMath.distance(x, y, distance_type="euclidean", pairwise=True)
    assert d.shape == (1, 5)
