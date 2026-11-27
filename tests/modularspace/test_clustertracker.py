import pytest
import torch

from positronixlab.modularspace import (
    ClusterIdentityTracker,
    ClusterTracker,
    LayerSpaceConfig,
)
from positronixlab.modularspace.custertracker import cluster_sizes, overlap_matrix


def test_alias_and_basic_helpers():
    assert ClusterIdentityTracker is ClusterTracker

    old_a = torch.tensor([0, 0, 1, 1, -1])
    new_a = torch.tensor([1, 1, 0, -1, 0])
    ov = overlap_matrix(old_a, 2, new_a, 2)
    assert ov.shape == (2, 2)
    assert ov[1, 0].item() == pytest.approx(2.0)
    assert ov[0, 1].item() == pytest.approx(1.0)
    assert cluster_sizes(new_a, 2).tolist() == [2, 2]


def test_clustertracker_init_and_match_preserves_outer_coords():
    tracker = ClusterTracker(d_out=2, seed=7)
    labels0 = torch.tensor([0, 0, 0, 1, 1, 1])
    centroids0 = torch.tensor([[0.0, 0.0], [3.0, 3.0]])
    state0, ev0 = tracker.step(labels0, centroids0)
    assert [e.kind for e in ev0] == ["init", "init"]
    assert state0.K == 2
    assert state0.outer.shape == (2, 2)

    # Swap cluster IDs 0 <-> 1; outer positions must follow identities
    labels1 = torch.tensor([1, 1, 1, 0, 0, 0])
    centroids1 = torch.tensor([[3.0, 3.0], [0.0, 0.0]])
    state1, ev1 = tracker.step(labels1, centroids1)
    assert all(e.kind == "match" for e in ev1)
    assert torch.allclose(state1.outer[0], state0.outer[1])
    assert torch.allclose(state1.outer[1], state0.outer[0])


@pytest.mark.parametrize("d_out", [2, 5])
def test_clustertracker_split_event(d_out):
    tracker = ClusterTracker(d_out=d_out, seed=3, gap_scale=0.5)
    labels0 = torch.tensor([0, 0, 0, 0, 1, 1])
    centroids0 = torch.tensor([[0.0, 0.0], [5.0, 5.0]])
    state0, _ = tracker.step(labels0, centroids0)

    # Old cluster 0 splits into new clusters 0 and 1; old 1 matches new 2
    labels1 = torch.tensor([0, 0, 1, 1, 2, 2])
    centroids1 = torch.tensor([[-1.0, 0.0], [1.0, 0.0], [5.0, 5.0]])
    state1, ev1 = tracker.step(labels1, centroids1)

    kinds = {e.kind for e in ev1}
    assert "split" in kinds
    assert "match" in kinds
    assert state1.outer.shape == (3, d_out)
    # Children should be separated around parent_outer
    assert not torch.allclose(state1.outer[0], state1.outer[1])
    assert torch.allclose(state1.outer[2], state0.outer[1])


def test_clustertracker_merge_add_remove_events():
    tracker = ClusterTracker(d_out=2, seed=11)
    # 3 old clusters + 2 unassigned items
    labels0 = torch.tensor([0, 0, 1, 1, 2, 2, -1, -1])
    centroids0 = torch.tensor([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
    state0, _ = tracker.step(labels0, centroids0)

    # New 0 merges old 0 & 1; new 1 is brand-new from unassigned items; old 2 is removed
    labels1 = torch.tensor([0, 0, 0, 0, -1, -1, 1, 1])
    centroids1 = torch.tensor([[1.0, 0.0], [10.0, 10.0]])
    state1, ev1 = tracker.step(labels1, centroids1)

    kinds = [e.kind for e in ev1]
    assert "merge" in kinds
    assert "add" in kinds
    assert "remove" in kinds
    expected_merge = 0.5 * (state0.outer[0] + state0.outer[1])
    assert torch.allclose(state1.outer[0], expected_merge)


def test_clustertracker_switch_event_emitted():
    tracker = ClusterTracker(d_out=2, seed=5, switch_threshold=0.05)
    # 10 items in cluster 0, 10 items in cluster 1
    labels0 = torch.tensor([0] * 10 + [1] * 10)
    centroids0 = torch.tensor([[0.0, 0.0], [4.0, 4.0]])
    tracker.step(labels0, centroids0)

    # 2 items jump from cluster 1 into cluster 0 (2/12 = 0.167 > 0.05)
    labels1 = torch.tensor([0] * 12 + [1] * 8)
    centroids1 = torch.tensor([[0.2, 0.2], [4.0, 4.0]])
    _, ev1 = tracker.step(labels1, centroids1)
    assert any(e.kind == "switch" for e in ev1)


def test_modular_overspace_update_and_queries():
    pytest.importorskip(
        "parallelazy",
        reason="ModularOverSpace imports PipelineConfig at top-level (modularoverspace.py:10), which requires parallelazy",
    )
    from positronixlab.modularspace import ModularOverSpace

    cfg = LayerSpaceConfig(modular_overspace_size=4)
    overspace = ModularOverSpace(
        cfg,
        observed_layers={"l0": {"out_units": 6}},
        device=torch.device("cpu"),
        stats_enabled=True,
        seed=0,
    )
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    centroids = torch.randn(2, 3)

    state, events = overspace.update("l0", labels, centroids)
    assert state.outer.shape == (2, 4)
    assert torch.equal(overspace.get_positions("l0"), state.outer)
    assert overspace.get_state("l0") is state
    assert overspace.get_events("l0") == events
    assert overspace.stats["l0"] == {"init": 2}

    with pytest.raises(ValueError, match="not found in ModularOverSpace"):
        overspace.get_state("unknown")
    assert overspace.get_events("unknown") == []


def test_clustertracker_recovers_from_empty_previous_step():
    tracker = ClusterTracker(d_out=2, seed=0)
    tracker.step(torch.tensor([-1, -1, -1]), torch.zeros(0, 3))
    state1, ev1 = tracker.step(torch.tensor([0, 0, 0]), torch.ones(1, 3))
    assert state1.K == 1
    assert [e.kind for e in ev1] == ["add"]


def test_clustertracker_split_positions_single_child_offsets_from_parent():
    tracker = ClusterTracker(d_out=2, seed=0, gap_scale=0.5)
    parent = torch.tensor([1.0, 2.0])
    child_centroid = torch.tensor([[5.0, 6.0]])
    split_pos = tracker._split_positions(parent, child_centroid, seed=123)[0]
    assert not torch.allclose(split_pos, parent)
