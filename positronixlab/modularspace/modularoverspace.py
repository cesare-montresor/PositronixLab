
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional

import torch

from .custertracker import ClusterEvent, ClusterState, ClusterTracker
from positronixlab.pipeline.pipeline_config import PipelineConfig


class ModularOverSpace:
    """Per-layer wrapper around `ClusterTracker`.

    The tracker stores stable outer-space positions for the cluster identities
    of each layer. This class owns one tracker per layer_id and exposes a small
    API for the overspace plugin:

    * `update(...)` advances the state for one layer
    * `get_state(...)` returns the most recent tracked state
    * `get_positions(...)` returns the current outer-space coordinates
    """

    def __init__(
        self,
        config: PipelineConfig,
        observed_layers: dict[str, dict],
        device: torch.device | None = None,
        stats_enabled: bool = False,
        seed: int = 0,
    ):
        self.config = config
        self.observed_layers = observed_layers
        self.device = device if device is not None else torch.device("cpu")
        self.stats_enabled = stats_enabled
        self.d_out = config.modular_overspace_size
        self.seed = seed

        self.trackers: Dict[str, ClusterTracker] = {}
        self.states: Dict[str, ClusterState] = {}
        self.events: Dict[str, List[ClusterEvent]] = {}
        self.positions: Dict[str, torch.Tensor] = {}
        self.stats: Dict[str, Dict[str, int]] = {}

    def _tracker_for(self, layer_id: str) -> ClusterTracker:
        tracker = self.trackers.get(layer_id)
        if tracker is None:
            tracker = ClusterTracker(d_out=self.d_out, device=self.device, seed=self.seed + len(self.trackers))
            self.trackers[layer_id] = tracker
        return tracker

    def update(self, layer_id: str, labels: torch.Tensor, centroids: torch.Tensor) -> tuple[ClusterState, List[ClusterEvent]]:
        labels = labels.to(self.device).long()
        centroids = centroids.to(self.device)

        tracker = self._tracker_for(layer_id)
        state, events = tracker.step(labels, centroids)

        self.states[layer_id] = state
        self.events[layer_id] = events
        self.positions[layer_id] = state.outer

        if self.stats_enabled:
            event_counts = Counter(event.kind for event in events)
            self.stats[layer_id] = {kind: int(count) for kind, count in event_counts.items()}

        return state, events

    def get_state(self, layer_id: str) -> ClusterState:
        state = self.states.get(layer_id)
        if state is None:
            raise ValueError(f"Layer {layer_id} not found in ModularOverSpace.")
        return state

    def get_positions(self, layer_id: str) -> torch.Tensor:
        state = self.get_state(layer_id)
        return state.outer

    def get_events(self, layer_id: str) -> List[ClusterEvent]:
        return self.events.get(layer_id, [])



    