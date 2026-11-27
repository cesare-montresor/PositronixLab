"""
Cluster identity tracking for modular overspace.

This module is a cleaned-up implementation of the logic in
``custertracking.example.py``. It keeps a stable outer-space position for
each cluster identity while handling the common transitions between
successive clusterings.

The outer space is intentionally decoupled from the embedding-space
centroids: centroids are only used when a cluster splits, to estimate the
gap between the split children.

Supported transitions:

* match
* split
* merge
* switch
* add
* remove

The expensive part is the overlap matrix, which is fully vectorized with
torch. Everything else stays at cluster granularity because K is small.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class ClusterState:
    """Snapshot of clustering plus outer-space projection at one timestep."""

    assignment: Tensor
    centroids: Tensor
    sizes: Tensor
    outer: Tensor

    @property
    def K(self) -> int:
        return int(self.centroids.shape[0])


@dataclass
class ClusterEvent:
    """Human-readable record of one identity transition."""

    kind: str
    new_ids: tuple[int, ...] = ()
    old_ids: tuple[int, ...] = ()
    detail: str = ""

    def __repr__(self) -> str:
        return f"<{self.kind} new={self.new_ids} old={self.old_ids} {self.detail}>"


def _one_hot(assignment: Tensor, k: int) -> Tensor:
    """[N] long (-1 allowed) -> [N, K] float one-hot."""

    n_items = int(assignment.shape[0])
    one_hot = torch.zeros(n_items, k, device=assignment.device, dtype=torch.float32)
    valid = assignment >= 0
    if valid.any():
        one_hot[valid, assignment[valid]] = 1.0
    return one_hot


def overlap_matrix(old_assignment: Tensor, k_old: int, new_assignment: Tensor, k_new: int) -> Tensor:
    """Return overlap[new_cluster, old_cluster] item counts."""

    old_one_hot = _one_hot(old_assignment, k_old)
    new_one_hot = _one_hot(new_assignment, k_new)
    return new_one_hot.T @ old_one_hot


def cluster_sizes(assignment: Tensor, k: int) -> Tensor:
    return _one_hot(assignment, k).sum(dim=0).long()


def _random_outer(k: int, d_out: int, bounds: float, device, generator=None) -> Tensor:
    return (torch.rand(k, d_out, device=device, generator=generator) * 2.0 - 1.0) * bounds


def _seeded_unit_directions(seed: int, m: int, d_out: int, device) -> Tensor:
    """Deterministic unit directions for split placement."""

    if d_out == 2:
        angles = torch.arange(m, device=device, dtype=torch.float32) * (2.0 * math.pi / max(m, 1))
        phase = (seed % 360) * math.pi / 180.0
        angles = angles + phase
        return torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    vectors = torch.randn(m, d_out, generator=generator).to(device)
    return vectors / vectors.norm(dim=1, keepdim=True).clamp_min(1e-8)


class ClusterTracker:
    """Track cluster identities over time and project them into outer space."""

    def __init__(
        self,
        d_out: int = 2,
        match_threshold: float = 0.5,
        switch_threshold: float = 0.05,
        gap_scale: float = 0.5,
        init_bounds: float = 1.0,
        device: torch.device | None = None,
        seed: int = 0,
    ):
        self.d_out = d_out
        self.match_threshold = match_threshold
        self.switch_threshold = switch_threshold
        self.gap_scale = gap_scale
        self.init_bounds = init_bounds
        self.device = device or torch.device("cpu")
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(seed)

        self.state: ClusterState | None = None
        self.history: list[list[ClusterEvent]] = []

    def step(self, assignment: Tensor, centroids: Tensor) -> tuple[ClusterState, list[ClusterEvent]]:
        """Advance the tracker by one timestep.

        The `assignment` tensor is expected to use contiguous cluster ids
        in `[0, K_new - 1]` with `-1` reserved for unassigned items.
        """

        assignment = assignment.to(self.device).long()
        centroids = centroids.to(self.device)
        k_new = int(centroids.shape[0])
        sizes_new = cluster_sizes(assignment, k_new)

        if self.state is None:
            return self._init(assignment, centroids, sizes_new)

        events, outer_new = self._match_and_project(assignment, centroids, sizes_new)
        state = ClusterState(assignment=assignment, centroids=centroids, sizes=sizes_new, outer=outer_new)
        self.state = state
        self.history.append(events)
        return state, events

    def _init(self, assignment: Tensor, centroids: Tensor, sizes: Tensor) -> tuple[ClusterState, list[ClusterEvent]]:
        k = int(centroids.shape[0])
        outer = _random_outer(k, self.d_out, self.init_bounds, self.device, self._generator)
        state = ClusterState(assignment=assignment, centroids=centroids, sizes=sizes, outer=outer)
        events = [ClusterEvent(kind="init", new_ids=(j,)) for j in range(k)]
        self.state = state
        self.history.append(events)
        return state, events

    def _match_and_project(self, assignment: Tensor, centroids: Tensor, sizes_new: Tensor) -> tuple[list[ClusterEvent], Tensor]:
        old = self.state
        if old is None:
            raise RuntimeError("ClusterTracker._match_and_project called before initial state was set.")

        k_old, k_new = old.K, int(centroids.shape[0])
        overlap = overlap_matrix(old.assignment, k_old, assignment, k_new)
        frac = overlap / sizes_new.clamp_min(1).unsqueeze(1).float()

        primary: dict[int, int] = {}
        secondary: dict[int, list[tuple[int, float]]] = {j: [] for j in range(k_new)}
        merge_candidates: dict[int, list[tuple[int, float]]] = {}
        no_lineage: list[int] = []

        order = torch.argsort(sizes_new, descending=True).tolist()
        for j in order:
            row = frac[j]
            best_i = int(torch.argmax(row).item())
            best_frac = float(row[best_i].item())

            if best_frac > self.match_threshold:
                primary[j] = best_i
                for i in range(k_old):
                    if i == best_i:
                        continue
                    share = float(row[i].item())
                    if share > self.switch_threshold:
                        secondary[j].append((i, share))
                continue

            contributors = [(i, float(row[i].item())) for i in range(k_old) if float(row[i].item()) > self.switch_threshold]
            contributors.sort(key=lambda item: -item[1])
            total_frac = sum(frac_value for _, frac_value in contributors)
            if contributors and total_frac > self.match_threshold:
                merge_candidates[j] = contributors
            else:
                no_lineage.append(j)

        children_of: dict[int, list[int]] = {}
        for j, i in primary.items():
            children_of.setdefault(i, []).append(j)

        events: list[ClusterEvent] = []
        outer_new = torch.zeros(k_new, self.d_out, device=self.device)

        for old_id, children in children_of.items():
            if len(children) == 1:
                child = children[0]
                outer_new[child] = old.outer[old_id]
                events.append(ClusterEvent(kind="match", new_ids=(child,), old_ids=(old_id,)))
            else:
                child_positions = self._split_positions(old.outer[old_id], centroids[children], seed=old_id)
                for child_id, position in zip(children, child_positions, strict=False):
                    outer_new[child_id] = position
                events.append(
                    ClusterEvent(
                        kind="split",
                        new_ids=tuple(children),
                        old_ids=(old_id,),
                        detail=f"n_children={len(children)}",
                    )
                )

        for child_id, contributors in merge_candidates.items():
            old_ids = [old_id for old_id, _ in contributors]
            weights = old.sizes[old_ids].float()
            outer_new[child_id] = (old.outer[old_ids] * weights.unsqueeze(1)).sum(dim=0) / weights.sum().clamp_min(1)
            events.append(ClusterEvent(kind="merge", new_ids=(child_id,), old_ids=tuple(old_ids)))

        for child_id, contribs in secondary.items():
            if child_id not in primary or not contribs:
                continue

            base_pos = outer_new[child_id].clone()
            base_weight = float(sizes_new[child_id].item()) - sum(float(overlap[child_id, old_id].item()) for old_id, _ in contribs)
            base_weight = max(base_weight, 1.0)
            acc_pos = base_pos * base_weight
            acc_weight = base_weight
            switch_old_ids: list[int] = []

            for old_id, _ in contribs:
                split_pos = self._split_positions(old.outer[old_id], centroids[[child_id]], seed=old_id * 1_000_003 + child_id)[0]
                weight = float(overlap[child_id, old_id].item())
                # A small linear interpolation keeps the transition smooth when
                # items drift from one cluster lineage into another.
                drift = 0.15 * (weight / float(sizes_new[child_id].item()))
                interp_pos = base_pos * (1.0 - drift) + split_pos * drift
                acc_pos = acc_pos + interp_pos * weight
                acc_weight += weight
                switch_old_ids.append(old_id)

            outer_new[child_id] = acc_pos / acc_weight
            events.append(
                ClusterEvent(
                    kind="switch",
                    new_ids=(child_id,),
                    old_ids=tuple(switch_old_ids),
                    detail=f"contrib_frac={[round(frac_value, 3) for _, frac_value in contribs]}",
                )
            )

        if no_lineage:
            fresh = _random_outer(len(no_lineage), self.d_out, self.init_bounds, self.device, self._generator)
            for index, child_id in enumerate(no_lineage):
                outer_new[child_id] = fresh[index]
                events.append(ClusterEvent(kind="add", new_ids=(child_id,)))

        used_old = set(children_of.keys())
        for contributors in merge_candidates.values():
            used_old.update(old_id for old_id, _ in contributors)
        for contribs in secondary.values():
            used_old.update(old_id for old_id, _ in contribs)

        for old_id in range(k_old):
            if old_id not in used_old:
                events.append(ClusterEvent(kind="remove", old_ids=(old_id,)))

        return events, outer_new

    def _split_positions(self, parent_outer: Tensor, children_centroids: Tensor, seed: int) -> Tensor:
        """Place children around their parent using only split-gap geometry.

        Centroids do not steer the outer-space layout beyond estimating the
        split gap. The final directions are deterministic but independent of
        the embedding space.
        """

        m = int(children_centroids.shape[0])
        if m == 1:
            return parent_outer.unsqueeze(0)

        pairwise = torch.cdist(children_centroids, children_centroids)
        mask = ~torch.eye(m, dtype=torch.bool, device=pairwise.device)
        distance = pairwise[mask].mean() if mask.any() else torch.tensor(0.0, device=pairwise.device)
        radius = self.gap_scale * (distance / 2.0)
        directions = _seeded_unit_directions(seed, m, self.d_out, self.device)
        return parent_outer.unsqueeze(0) + radius * directions


ClusterIdentityTracker = ClusterTracker
