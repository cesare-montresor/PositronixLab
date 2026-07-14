"""Minimal ClusterTracker example.

This script shows the smallest useful way to use the tracker in your own
project:

1. create a ``ClusterTracker``
2. pass in cluster assignments and centroids for each step
3. read back the tracked outer-space positions and events

There is no visualization here.
"""

from __future__ import annotations

import torch

from modularnet.modularspace import ClusterTracker


def centroids_from_points(points: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    groups = []
    for cluster_id in range(int(labels.max().item()) + 1):
        mask = labels == cluster_id
        groups.append(points[mask].mean(dim=0))
    return torch.stack(groups, dim=0)


def main() -> None:
    tracker = ClusterTracker(d_out=2, seed=7)

    points_t0 = torch.tensor([
        [0.0, 0.0],
        [0.1, 0.1],
        [3.0, 3.0],
        [3.1, 3.2],
    ])
    labels_t0 = torch.tensor([0, 0, 1, 1])
    centroids_t0 = centroids_from_points(points_t0, labels_t0)
    state0, events0 = tracker.step(labels_t0, centroids_t0)

    print("step 0 events:", events0)
    print("step 0 outer:\n", state0.outer)

    points_t1 = torch.tensor([
        [0.0, 0.0],
        [0.2, 0.1],
        [3.0, 3.0],
        [3.2, 3.1],
    ])
    labels_t1 = torch.tensor([0, 0, 1, 1])
    centroids_t1 = centroids_from_points(points_t1, labels_t1)
    state1, events1 = tracker.step(labels_t1, centroids_t1)

    print("step 1 events:", events1)
    print("step 1 outer:\n", state1.outer)


if __name__ == "__main__":
    main()
