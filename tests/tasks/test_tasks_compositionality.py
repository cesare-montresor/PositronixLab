from types import SimpleNamespace
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from positronixlab.pipeline import PipelineConfigV1
from positronixlab.tasks.compositionality.atlas_check import (
    AtlasCheck,
    DiscreteDistribution,
    TensorDistribution,
)
from positronixlab.tasks.compositionality.weightmask_check import (
    MultiClassWeightmaskManager,
    WeightMaskCriterion,
    WeightMasksMetrics,
    WeightmaskCheck,
)
from positronixlab.tasks.compositionality.compositionality import (
    CompositionalityCheck,
    CompositionalityPipeline,
    StaticMaskApplier,
)


# ---------------------------------------------------------------------------
# Synthetic helpers (CPU-only, no external checkpoints)
# ---------------------------------------------------------------------------

class ModularMLP(nn.Module):
    """Simple MLP with .device attribute for synthetic modularity tests."""

    def __init__(self, in_features: int = 6, hidden: int = 6, num_classes: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, num_classes)
        self.device = torch.device("cpu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class NestedMLP(nn.Module):
    """MLP with nested nn.Sequential layers (dotted names like 'net.0')."""

    def __init__(self, in_features: int = 6, hidden: int = 4, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )
        self.device = torch.device("cpu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_synthetic_dataprovider(
    num_samples: int = 36,
    in_features: int = 6,
    num_classes: int = 3,
    batch_size: int = 18,
    indexed: bool = False,
    include_test: bool = True,
):
    """Creates a balanced synthetic dataset where each class activates a distinct feature block."""
    torch.manual_seed(42)
    xs_list, ys_list = [], []
    per_class = num_samples // num_classes
    block = in_features // num_classes
    for c in range(num_classes):
        x_c = torch.randn(per_class, in_features) * 0.1
        x_c[:, c * block : (c + 1) * block] += 3.0
        y_c = torch.full((per_class,), c, dtype=torch.long)
        xs_list.append(x_c)
        ys_list.append(y_c)

    xs = torch.cat(xs_list, dim=0)
    ys = torch.cat(ys_list, dim=0)
    # Interleave classes so every batch contains all classes
    perm = torch.arange(num_samples).reshape(num_classes, per_class).T.reshape(-1)
    xs, ys = xs[perm], ys[perm]

    if indexed:
        ds = TensorDataset(xs, ys, torch.arange(num_samples))
    else:
        ds = TensorDataset(xs, ys)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return SimpleNamespace(
        num_classes=num_classes,
        train=loader,
        val=loader,
        test=loader if include_test else None,
    )


def make_modular_model(num_classes: int = 3) -> ModularMLP:
    """Constructs a deterministic block-diagonal MLP with clean class specialization."""
    model = ModularMLP(in_features=num_classes * 2, hidden=num_classes * 2, num_classes=num_classes)
    with torch.no_grad():
        model.fc1.weight.zero_()
        model.fc1.bias.zero_()
        model.fc2.weight.zero_()
        model.fc2.bias.zero_()
        for c in range(num_classes):
            model.fc1.weight[2 * c : 2 * c + 2, 2 * c : 2 * c + 2] = torch.eye(2)
            model.fc2.weight[c, 2 * c : 2 * c + 2] = 1.0
    return model


# ---------------------------------------------------------------------------
# Distribution tests
# ---------------------------------------------------------------------------

class TestDistributions:
    def test_discrete_distribution_scalar_vector_and_add(self):
        d1 = DiscreteDistribution(num_classes=3)
        assert torch.allclose(d1.get_distribution(), torch.zeros(3))

        d1.update(0)
        d1.update(torch.tensor(1), torch.tensor(3.0))
        d1.update(torch.tensor([2, 2]))
        assert torch.allclose(d1.get_distribution(), torch.tensor([1 / 6, 3 / 6, 2 / 6]))

        d2 = DiscreteDistribution(num_classes=3)
        d2.update(torch.tensor([0, 1]), torch.tensor([2.0, 2.0]))
        d1.add(d2)
        assert torch.allclose(d1.get_distribution(), torch.tensor([3 / 10, 5 / 10, 2 / 10]))

    def test_discrete_distribution_add_validation(self):
        d1 = DiscreteDistribution(num_classes=3)
        with pytest.raises(TypeError):
            d1.add(TensorDistribution(torch.Size([3])))
        with pytest.raises(ValueError):
            d1.add(DiscreteDistribution(num_classes=4))

    def test_tensor_distribution_update_and_add(self):
        t1 = TensorDistribution(torch.Size([2, 2]))
        assert torch.allclose(t1.get_distribution(), torch.zeros(2, 2))

        t1.update(torch.ones(2, 2))
        t1.update(torch.full((2, 2), 3.0))
        assert torch.allclose(t1.get_distribution(), torch.full((2, 2), 2.0))

        t2 = TensorDistribution(torch.Size([2, 2]))
        t2.update(torch.full((2, 2), 5.0))
        t1.add(t2)
        assert torch.allclose(t1.get_distribution(), torch.full((2, 2), 3.0))

        with pytest.raises(TypeError):
            t1.add(DiscreteDistribution(num_classes=2))
        with pytest.raises(ValueError):
            t1.add(TensorDistribution(torch.Size([3])))


# ---------------------------------------------------------------------------
# AtlasCheck tests
# ---------------------------------------------------------------------------

class TestAtlasCheck:
    def test_atlas_run_modularity_and_extract_masks(self):
        model = make_modular_model(num_classes=3)
        dp = make_synthetic_dataprovider(num_samples=36, in_features=6, num_classes=3, batch_size=18)
        config = PipelineConfigV1()
        clusters = {
            "fc1": torch.tensor([0, 0, 1, 1, 2, 2]),
            "fc2": torch.tensor([0, 1, 2]),
        }

        atlas = AtlasCheck(config=config, model=model, dataprovider=dp, clusters=clusters)
        try:
            atlas.run()
            scores_y = atlas.modularity_score(kld_target="y")
            scores_u = atlas.modularity_score(kld_target="uniform")

            assert "fc1" in scores_y and "fc2" in scores_y and "model" in scores_y
            assert 0.5 < scores_y["model"]["modularity_purity"] <= 1.0
            assert scores_y["model"]["modularity_kld"] >= 0.0
            assert 0.0 <= scores_y["model"]["modularity_kld_exp"] <= 1.0
            assert abs(scores_y["model"]["modularity_kld"] - scores_u["model"]["modularity_kld"]) < 1e-5

            with pytest.raises(ValueError, match="Unknown kld_target"):
                atlas.modularity_score(kld_target="invalid")

            masks = atlas.extract_weightmasks(threshold=0.05)
            assert set(masks.keys()) == {0, 1, 2}
            for c in range(3):
                assert masks[c]["fc1"].dtype == torch.bool
                assert masks[c]["fc1"].shape == (6,)
                assert masks[c]["fc1"][2 * c : 2 * c + 2].all()
        finally:
            atlas.observer.on_forward_layer(None)

    def test_atlas_supports_indexed_batches(self):
        model = make_modular_model(num_classes=3)
        dp = make_synthetic_dataprovider(indexed=True)
        clusters = {
            "fc1": torch.tensor([0, 0, 1, 1, 2, 2]),
            "fc2": torch.tensor([0, 1, 2]),
        }
        atlas = AtlasCheck(config=PipelineConfigV1(), model=model, dataprovider=dp, clusters=clusters)
        try:
            atlas.run()
            assert "fc1" in atlas.cluster_class_stats
        finally:
            atlas.observer.on_forward_layer(None)

    @pytest.mark.xfail(
        strict=True,
        reason="Confirmed bug (atlas_check.py:310-315): modularity_score() divides by len(stats)==0 when empty.",
    )
    def test_modularity_score_before_run_handles_empty_stats(self):
        model = make_modular_model(num_classes=3)
        dp = make_synthetic_dataprovider()
        clusters = {"fc1": torch.tensor([0, 0, 1, 1, 2, 2]), "fc2": torch.tensor([0, 1, 2])}
        atlas = AtlasCheck(config=PipelineConfigV1(), model=model, dataprovider=dp, clusters=clusters)
        try:
            scores = atlas.modularity_score()
            assert "model" in scores
        finally:
            atlas.observer.on_forward_layer(None)

    @pytest.mark.xfail(
        strict=True,
        reason="Confirmed bug (atlas_check.py:117): AtlasCheck never detaches forward hooks after run().",
    )
    def test_atlas_detaches_hooks_after_run(self):
        model = make_modular_model(num_classes=3)
        dp = make_synthetic_dataprovider()
        clusters = {"fc1": torch.tensor([0, 0, 1, 1, 2, 2]), "fc2": torch.tensor([0, 1, 2])}
        atlas = AtlasCheck(config=PipelineConfigV1(), model=model, dataprovider=dp, clusters=clusters)
        try:
            atlas.run()
            assert len(model.fc1._forward_hooks) == 0
        finally:
            atlas.observer.on_forward_layer(None)


# ---------------------------------------------------------------------------
# WeightmaskCheck & WeightMasksMetrics tests
# ---------------------------------------------------------------------------

class TestWeightmaskCheck:
    def test_weightmask_criterion_and_metrics(self):
        criterion = WeightMaskCriterion(lambda_l1=1e-2)
        logits = torch.tensor([[3.0, 0.0], [0.0, 3.0]], requires_grad=True)
        targets = torch.tensor([0, 1])
        m = torch.zeros((2, 2), requires_grad=True)
        out = criterion(logits, targets, [m])
        assert set(out.keys()) == {"total_loss", "task_loss", "overall_weightmaskloss"}
        assert out["overall_weightmaskloss"].item() == pytest.approx(2.0, rel=1e-5)

        empty_out = criterion(logits, targets, [])
        assert empty_out["overall_weightmaskloss"].item() == 0.0

        a = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
        b = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        assert WeightMasksMetrics.compute_iou(a, b) == pytest.approx(1 / 3)
        assert WeightMasksMetrics.compute_iomin(a, b) == pytest.approx(1 / 2)
        assert WeightMasksMetrics.compute_sparsity(a) == pytest.approx(0.5)

        net_mod = WeightMasksMetrics.evaluate_network_modularity({"l1": a}, {"l1": b})
        assert net_mod["network_iou"] == pytest.approx(1 / 3)
        assert net_mod["network_iomin"] == pytest.approx(1 / 2)

        assert WeightMasksMetrics.compute_accuracy(logits.detach(), targets) == 1.0

        preds_seq = torch.tensor([[1, 2, 9], [1, 3, 0]])
        targs_seq = torch.tensor([[1, 2, 0], [1, 2, 0]])
        assert WeightMasksMetrics.compute_sequence_accuracy(preds_seq, targs_seq, pad_token_id=0) == 0.5

    def test_weightmask_check_runs_on_balanced_batches(self):
        torch.manual_seed(0)
        model = make_modular_model(num_classes=3)
        dp = make_synthetic_dataprovider(num_samples=36, in_features=6, num_classes=3, batch_size=18)
        target_layers = {"fc1": model.fc1, "fc2": model.fc2}

        checker = WeightmaskCheck(model=model, dataprovider=dp)
        masks = checker.run(target_layers=target_layers, num_epochs=2, lr=0.1)

        assert set(masks.keys()) == {0, 1, 2}
        for c in range(3):
            assert masks[c]["fc1"].shape == model.fc1.weight.shape
            assert masks[c]["fc2"].shape == model.fc2.weight.shape
            assert not torch.isnan(masks[c]["fc1"]).any()
        # Original nn.Parameter restored
        assert isinstance(model.fc1.weight, nn.Parameter)
        assert len(model.fc1._forward_pre_hooks) == 0

    @pytest.mark.xfail(
        strict=True,
        reason="Confirmed bug (weightmask_check.py:18,33): num_epochs is ignored; only 1 epoch executes.",
    )
    def test_weightmask_respects_num_epochs(self):
        model = make_modular_model(num_classes=3)
        dp = make_synthetic_dataprovider(num_samples=18, in_features=6, num_classes=3, batch_size=18)

        iter_count = 0
        orig_loader = dp.train

        class CountingLoader:
            def __iter__(self):
                nonlocal iter_count
                iter_count += 1
                return iter(orig_loader)

            def __len__(self):
                return len(orig_loader)

        dp.train = CountingLoader()
        checker = WeightmaskCheck(model=model, dataprovider=dp)
        checker.run(target_layers={"fc1": model.fc1}, num_epochs=3, lr=0.05)
        assert iter_count == 3

    @pytest.mark.xfail(
        strict=True,
        reason="Confirmed bug (weightmask_check.py:41-54): empty class slice produces NaN task_loss/total_loss.",
    )
    def test_weightmask_handles_batch_missing_a_class(self):
        model = make_modular_model(num_classes=3)
        # Batch with only classes 0 and 1 (class 2 absent)
        xs = torch.randn(6, 6)
        ys = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

        target_layers = {"fc1": model.fc1}
        mgr = MultiClassWeightmaskManager(model, target_layers, num_classes=3)
        try:
            crit = WeightMaskCriterion()
            for c in range(3):
                mask = ys == c
                mgr.current_class = c
                logits = mgr(xs[mask])
                loss_dict = crit(logits, ys[mask], mgr.active_layer_masks)
                assert not torch.isnan(loss_dict["total_loss"]), f"NaN loss for missing class {c}"
        finally:
            mgr.remove_hooks()

    @pytest.mark.xfail(
        strict=True,
        reason="Confirmed bug (weightmask_check.py:95-96,144-159): remove_hooks() does not restore requires_grad.",
    )
    def test_weightmask_restores_requires_grad(self):
        model = make_modular_model(num_classes=3)
        assert all(p.requires_grad for p in model.parameters())
        dp = make_synthetic_dataprovider(num_samples=18, batch_size=18)
        WeightmaskCheck(model=model, dataprovider=dp).run(target_layers={"fc1": model.fc1}, num_epochs=1)
        assert all(p.requires_grad for p in model.parameters())

    @pytest.mark.xfail(
        strict=True,
        reason="Confirmed bug (weightmask_check.py:98-117): target layer without weight raises KeyError in pre_hook.",
    )
    def test_weightmask_manager_ignores_weightless_layers(self):
        model = make_modular_model(num_classes=3)
        mgr = MultiClassWeightmaskManager(model, {"fc1": model.fc1, "act": model.act}, num_classes=3)
        try:
            _ = mgr(torch.randn(4, 6))
        finally:
            mgr.remove_hooks()


# ---------------------------------------------------------------------------
# StaticMaskApplier, CompositionalityCheck & CompositionalityPipeline tests
# ---------------------------------------------------------------------------

class TestCompositionality:
    def test_static_mask_applier_masks_and_restores_on_exit_and_exception(self):
        model = make_modular_model(num_classes=3)
        orig_w = model.fc1.weight.clone()
        zero_mask = {"fc1": torch.zeros_like(model.fc1.weight)}

        x = torch.ones(2, 6)
        with StaticMaskApplier(model, zero_mask):
            out_masked = model(x)
            assert torch.allclose(out_masked, torch.zeros_like(out_masked))

        assert isinstance(model.fc1.weight, nn.Parameter)
        assert torch.allclose(model.fc1.weight, orig_w)

        with pytest.raises(RuntimeError, match="boom"):
            with StaticMaskApplier(model, zero_mask):
                raise RuntimeError("boom")
        assert isinstance(model.fc1.weight, nn.Parameter)
        assert torch.allclose(model.fc1.weight, orig_w)

    def test_compositionality_check_end_to_end_with_adapted_masks(self):
        model = make_modular_model(num_classes=3)
        dp = make_synthetic_dataprovider(num_samples=36, in_features=6, num_classes=3, batch_size=18)

        # Build compatible 2D (or broadcastable (out, 1)) masks per class
        atlas_masks = {}
        csordas_masks = {}
        for c in range(3):
            m1 = torch.zeros(6, 1)
            m1[2 * c : 2 * c + 2, 0] = 1.0
            m2 = torch.zeros(3, 1)
            m2[c, 0] = 1.0
            atlas_masks[c] = {"fc1": m1, "fc2": m2}
            csordas_masks[c] = {
                "fc1": m1.expand_as(model.fc1.weight).clone(),
                "fc2": m2.expand_as(model.fc2.weight).clone(),
            }

        checker = CompositionalityCheck(model=model, dataloader=dp.test, num_classes=3)
        results = checker.run(atlas_masks=atlas_masks, csordas_masks=csordas_masks)

        assert results["untainted"]["accuracy"] == pytest.approx(1.0)
        assert set(results["comparison"].keys()) == {0, 1, 2}
        for c in range(3):
            assert results["atlas"][c]["target_recall"] == pytest.approx(1.0)
            assert results["comparison"][c]["cm_frobenius_diff"] == pytest.approx(0.0, abs=1e-5)

    def test_compositionality_check_skips_missing_class_mask(self):
        model = make_modular_model(num_classes=3)
        dp = make_synthetic_dataprovider(num_samples=18, batch_size=18)
        checker = CompositionalityCheck(model=model, dataloader=dp.test, num_classes=3)
        partial = {0: {"fc1": torch.ones_like(model.fc1.weight)}}
        res = checker.evaluate_masked_runs(partial, method_name="Partial")
        assert list(res.keys()) == [0]

    @pytest.mark.xfail(
        strict=True,
        reason="Confirmed bug (compositionality.py:151): run() divides by zero on empty dataloader.",
    )
    def test_compositionality_check_empty_dataloader(self):
        model = make_modular_model(num_classes=3)
        empty_loader = DataLoader(TensorDataset(torch.empty(0, 6), torch.empty(0, dtype=torch.long)), batch_size=4)
        checker = CompositionalityCheck(model=model, dataloader=empty_loader, num_classes=3)
        res = checker.run(atlas_masks={}, csordas_masks={})
        assert res["untainted"]["accuracy"] == 0.0

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Confirmed bug (atlas_check.py:235-240 + compositionality.py:28-36): "
            "Atlas 1D mask (out_features,) fails to broadcast against non-square weight (out, in) "
            "and uses dotted keys ('net.0') ignored by StaticMaskApplier ('net_0')."
        ),
    )
    def test_static_mask_applier_accepts_raw_atlas_masks(self):
        model = NestedMLP(in_features=6, hidden=4, num_classes=2)
        # Raw AtlasCheck mask format: dotted key + 1D bool tensor of length out_features
        raw_atlas_mask = {"net.0": torch.tensor([True, True, False, False])}
        x = torch.ones(2, 6)
        out_unmasked = model(x)
        with StaticMaskApplier(model, raw_atlas_mask):
            out_masked = model(x)
        assert not torch.allclose(out_unmasked, out_masked)

    @pytest.mark.xfail(
        strict=True,
        reason="Confirmed bug (compositionality.py:229): CompositionalityPipeline crashes when dataprovider.test is None.",
    )
    def test_compositionality_pipeline_handles_missing_test_split(self):
        model = make_modular_model(num_classes=3)
        dp = make_synthetic_dataprovider(include_test=False)
        clusters = {"fc1": torch.tensor([0, 0, 1, 1, 2, 2]), "fc2": torch.tensor([0, 1, 2])}
        pipe = CompositionalityPipeline(model=model, dataprovider=dp, config=PipelineConfigV1())
        pipe.run_full_evaluation(clusters=clusters, atlas_threshold=0.05, csordas_epochs=1)
