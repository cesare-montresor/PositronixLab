"""
cluster_identity_tracker.py

Temporal cluster identity tracking for an evolving, contrastively-trained
embedding space (ModularCluster). See cluster_identity_tracking.md for the
full write-up of the algorithm this implements.

Fixed number of items `N`. At every timestep you:
    1. Re-embed items                              (external)
    2. Re-cluster embeddings                        (external)
    3. tracker.step(assignment, centroids)  ->  (ClusterState, List[ClusterEvent])

Design goals:
    - The only thing that scales with N (one-hot + overlap matmul) is fully
      vectorized torch; everything cluster-level (K is small) is a plain
      Python loop for clarity, since K << N and it is not worth the
      complexity of vectorizing it.
    - No item-level Python loops anywhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------

@dataclass
class ClusterState:
    """Snapshot of clustering + outer-space projection at one timestep."""
    assignment: Tensor   # LongTensor [N]      item -> cluster id (-1 = unassigned)
    centroids: Tensor    # FloatTensor [K, D]  centroid in embedding space
    sizes: Tensor        # LongTensor [K]      item count per cluster
    outer: Tensor        # FloatTensor [K, d_out]  stable (u, v, ...) coords

    @property
    def K(self) -> int:
        return self.centroids.shape[0]


@dataclass
class ClusterEvent:
    """Human-readable record of what happened to one cluster identity."""
    kind: str                         # "init" | "match" | "split" | "merge" | "switch" | "add" | "remove"
    new_ids: Tuple[int, ...] = ()
    old_ids: Tuple[int, ...] = ()
    detail: str = ""

    def __repr__(self) -> str:
        return f"<{self.kind} new={self.new_ids} old={self.old_ids} {self.detail}>"


# --------------------------------------------------------------------------
# Small vectorized helpers
# --------------------------------------------------------------------------

def _one_hot(assignment: Tensor, K: int) -> Tensor:
    """[N] long (-1 allowed) -> [N, K] float one-hot, unassigned rows are all-zero."""
    N = assignment.shape[0]
    oh = torch.zeros(N, K, device=assignment.device, dtype=torch.float32)
    valid = assignment >= 0
    if valid.any():
        oh[valid, assignment[valid]] = 1.0
    return oh


def overlap_matrix(old_assignment: Tensor, K_old: int,
                    new_assignment: Tensor, K_new: int) -> Tensor:
    """overlap[j, i] = number of items shared between new cluster j and old cluster i."""
    oh_old = _one_hot(old_assignment, K_old)   # [N, K_old]
    oh_new = _one_hot(new_assignment, K_new)   # [N, K_new]
    return oh_new.T @ oh_old                    # [K_new, K_old]


def cluster_sizes(assignment: Tensor, K: int) -> Tensor:
    return _one_hot(assignment, K).sum(dim=0).long()


def _random_outer(K: int, d_out: int, bounds: float, device, generator=None) -> Tensor:
    return (torch.rand(K, d_out, device=device, generator=generator) * 2 - 1) * bounds


def _seeded_unit_directions(seed: int, m: int, d_out: int, device) -> Tensor:
    """m deterministic (seeded) unit vectors in d_out dims.

    For d_out == 2 these are placed evenly on a circle (exact, matches the
    sketch's radial split layout). For d_out > 2 we fall back to seeded
    random directions (normalized), since there is no canonical even
    spacing in higher dimensions.
    """
    if d_out == 2:
        angles = torch.arange(m, device=device, dtype=torch.float32) * (2 * math.pi / m)
        # small deterministic phase offset per seed so repeated splits of the
        # same parent don't always align on the same axis
        phase = (seed % 360) * math.pi / 180.0
        angles = angles + phase
        return torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    v = torch.randn(m, d_out, generator=gen).to(device)
    return v / v.norm(dim=1, keepdim=True).clamp_min(1e-8)


# --------------------------------------------------------------------------
# Tracker
# --------------------------------------------------------------------------

class ClusterIdentityTracker:
    """Matches successive independent clusterings of the same N items and
    maintains a temporally-stable low-dimensional projection (outer space)
    per cluster identity.

    Parameters
    ----------
    d_out : dimensionality of the outer (u, v, ...) space.
    match_threshold : overlap fraction (of the new cluster's size) above
        which an old cluster is accepted as the new cluster's primary lineage.
    switch_threshold : overlap fraction above which a *secondary* contribution
        is considered a real "switch" rather than noise.
    gap_scale : scales the embedding-space distance down (or up) into an
        outer-space split radius.
    init_bounds : half-width of the uniform init / add box in outer space.
    device : torch device.
    """

    def __init__(
        self,
        d_out: int = 2,
        match_threshold: float = 0.5,
        switch_threshold: float = 0.05,
        gap_scale: float = 0.5,
        init_bounds: float = 1.0,
        device: Optional[torch.device] = None,
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
        self._split_seed_counter = seed

        self.state: Optional[ClusterState] = None
        self.history: List[List[ClusterEvent]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, assignment: Tensor, centroids: Tensor) -> Tuple[ClusterState, List[ClusterEvent]]:
        """Advance the tracker by one timestep.

        assignment : LongTensor [N], item -> new cluster id (-1 allowed)
        centroids  : FloatTensor [K_new, D]
        """
        assignment = assignment.to(self.device)
        centroids = centroids.to(self.device)
        K_new = centroids.shape[0]
        sizes_new = cluster_sizes(assignment, K_new)

        if self.state is None:
            return self._init(assignment, centroids, sizes_new)

        events, outer_new = self._match_and_project(assignment, centroids, sizes_new)

        new_state = ClusterState(
            assignment=assignment, centroids=centroids, sizes=sizes_new, outer=outer_new,
        )
        self.state = new_state
        self.history.append(events)
        return new_state, events

    # ------------------------------------------------------------------
    # t0 initialization
    # ------------------------------------------------------------------

    def _init(self, assignment: Tensor, centroids: Tensor, sizes: Tensor):
        K = centroids.shape[0]
        outer = _random_outer(K, self.d_out, self.init_bounds, self.device, self._generator)
        state = ClusterState(assignment=assignment, centroids=centroids, sizes=sizes, outer=outer)
        events = [ClusterEvent(kind="init", new_ids=(j,)) for j in range(K)]
        self.state = state
        self.history.append(events)
        return state, events

    # ------------------------------------------------------------------
    # Matching + classification + coordinate update
    # ------------------------------------------------------------------

    def _match_and_project(self, assignment: Tensor, centroids: Tensor, sizes_new: Tensor):
        old = self.state
        K_old, K_new = old.K, centroids.shape[0]

        overlap = overlap_matrix(old.assignment, K_old, assignment, K_new)  # [K_new, K_old]
        frac = overlap / sizes_new.clamp_min(1).unsqueeze(1).float()        # [K_new, K_old]

        # --- pass 1: per-new-cluster primary / secondary contributions ---
        primary: Dict[int, int] = {}                 # new_j -> old_i
        secondary: Dict[int, List[Tuple[int, float]]] = {j: [] for j in range(K_new)}
        merge_candidates: Dict[int, List[Tuple[int, float]]] = {}
        no_lineage: List[int] = []

        # process new clusters biggest-first (sketch: sort groups by size)
        order = torch.argsort(sizes_new, descending=True).tolist()

        for j in order:
            row = frac[j]
            best_i = int(torch.argmax(row).item())
            best_frac = float(row[best_i].item())

            if best_frac > self.match_threshold:
                primary[j] = best_i
                for i in range(K_old):
                    if i == best_i:
                        continue
                    f = float(row[i].item())
                    if f > self.switch_threshold:
                        secondary[j].append((i, f))
                continue

            # no single old cluster explains > 50%: candidate merge, or add
            contributors = [(i, float(row[i].item())) for i in range(K_old)
                             if float(row[i].item()) > self.switch_threshold]
            contributors.sort(key=lambda t: -t[1])
            total_frac = sum(f for _, f in contributors)
            if contributors and total_frac > self.match_threshold:
                merge_candidates[j] = contributors
            else:
                no_lineage.append(j)

        # --- pass 2: aggregate on the old-cluster side (match vs split) ---
        children_of: Dict[int, List[int]] = {}
        for j, i in primary.items():
            children_of.setdefault(i, []).append(j)

        events: List[ClusterEvent] = []
        outer_new = torch.zeros(K_new, self.d_out, device=self.device)

        # matches & splits
        for i, children in children_of.items():
            if len(children) == 1:
                j = children[0]
                outer_new[j] = old.outer[i]
                events.append(ClusterEvent(kind="match", new_ids=(j,), old_ids=(i,)))
            else:
                positions = self._split_positions(old.outer[i], centroids[children], seed=i)
                for j, pos in zip(children, positions):
                    outer_new[j] = pos
                events.append(ClusterEvent(
                    kind="split", new_ids=tuple(children), old_ids=(i,),
                    detail=f"n_children={len(children)}",
                ))

        # merges
        for j, contributors in merge_candidates.items():
            old_ids = [i for i, _ in contributors]
            weights = old.sizes[old_ids].float()
            outer_new[j] = (old.outer[old_ids] * weights.unsqueeze(1)).sum(0) / weights.sum().clamp_min(1)
            events.append(ClusterEvent(kind="merge", new_ids=(j,), old_ids=tuple(old_ids)))

        # switches: secondary contributions layered on top of an already-placed primary
        for j, contribs in secondary.items():
            if not contribs or j not in primary:
                continue
            base_pos = outer_new[j].clone()
            base_weight = float(sizes_new[j].item()) - sum(
                float(overlap[j, i].item()) for i, _ in contribs
            )
            base_weight = max(base_weight, 1.0)
            acc_pos = base_pos * base_weight
            acc_weight = base_weight
            switch_old_ids = []
            for i, f in contribs:
                # step 1 (split-off): temporary position for the jumping subgroup
                split_pos = self._split_positions(
                    old.outer[i], centroids[[j]], seed=i * 1_000_003 + j,
                )[0]
                w = float(overlap[j, i].item())
                acc_pos = acc_pos + split_pos * w
                acc_weight += w
                switch_old_ids.append(i)
            outer_new[j] = acc_pos / acc_weight
            events.append(ClusterEvent(
                kind="switch", new_ids=(j,), old_ids=tuple(switch_old_ids),
                detail=f"contrib_frac={[round(f, 3) for _, f in contribs]}",
            ))

        # brand-new clusters, no usable lineage
        if no_lineage:
            fresh = _random_outer(len(no_lineage), self.d_out, self.init_bounds,
                                   self.device, self._generator)
            for k, j in enumerate(no_lineage):
                outer_new[j] = fresh[k]
                events.append(ClusterEvent(kind="add", new_ids=(j,)))

        # removed identities: old clusters that are neither a primary parent
        # nor a merge contributor for anyone
        used_old = set(children_of.keys())
        for contributors in merge_candidates.values():
            used_old.update(i for i, _ in contributors)
        for contribs in secondary.values():
            used_old.update(i for i, _ in contribs)
        removed = [i for i in range(K_old) if i not in used_old]
        for i in removed:
            events.append(ClusterEvent(kind="remove", old_ids=(i,)))

        return events, outer_new

    # ------------------------------------------------------------------
    # Split placement
    # ------------------------------------------------------------------

    def _split_positions(self, parent_outer: Tensor, children_centroids: Tensor, seed: int) -> Tensor:
        """Place m >= 1 children around `parent_outer`.

        Magnitude comes from the mean pairwise distance of the children's
        embedding-space centroids (GAP = DISTANCE / 2); direction is a
        deterministic (seeded) even circular spread in 2D, or seeded random
        unit vectors otherwise.
        """
        m = children_centroids.shape[0]
        if m == 1:
            return parent_outer.unsqueeze(0)

        pdist = torch.cdist(children_centroids, children_centroids)  # [m, m]
        mask = ~torch.eye(m, dtype=torch.bool, device=pdist.device)
        distance = pdist[mask].mean() if mask.any() else torch.tensor(0.0, device=pdist.device)
        gap = distance / 2.0
        radius = self.gap_scale * gap

        directions = _seeded_unit_directions(seed, m, self.d_out, self.device)
        return parent_outer.unsqueeze(0) + radius * directions


# --------------------------------------------------------------------------
# Self-test / demo
# --------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    N, D = 512, 16
    device = torch.device("cpu")
    tracker = ClusterIdentityTracker(d_out=2, device=device, seed=42)

    # --- t0: 4 clusters ---
    assignment0 = torch.randint(0, 4, (N,))
    centroids0 = torch.randn(4, D)
    state0, ev0 = tracker.step(assignment0, centroids0)
    print("t0:", ev0)
    print("outer0:\n", state0.outer)

    # --- t1: cluster 0 splits into 0 and 1; clusters 1,2,3 -> 2,3,4 (plain matches) ---
    assignment1 = assignment0.clone()
    mask0 = assignment0 == 0
    idx0 = mask0.nonzero(as_tuple=True)[0]
    half = idx0[: len(idx0) // 2]
    assignment1[half] = 5          # new child cluster id
    remap = {1: 2, 2: 3, 3: 4}
    for old_id, new_id in remap.items():
        assignment1[assignment0 == old_id] = new_id
    # relabel cluster 0's remainder (unsplit half) and the split child to 0,1
    remaining0 = idx0[len(idx0) // 2:]
    assignment1[remaining0] = 0
    assignment1[half] = 1

    # renumber contiguous 0..K-1 and recompute centroids from (shifted) embeddings
    uniq = sorted(assignment1.unique().tolist())
    remap2 = {u: k for k, u in enumerate(uniq)}
    assignment1 = torch.tensor([remap2[int(a)] for a in assignment1])
    centroids1 = torch.randn(len(uniq), D)  # stand-in for "re-embedded, re-centroided"
    # nudge child centroids of the split apart so DISTANCE is meaningful
    centroids1[0] += 1.0
    centroids1[1] -= 1.0

    state1, ev1 = tracker.step(assignment1, centroids1)
    print("\nt1 events:")
    for e in ev1:
        print(" ", e)
    print("outer1:\n", state1.outer)