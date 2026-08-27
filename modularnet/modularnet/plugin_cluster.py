import logging
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np
import faiss

from modularnet.modularcluster.streaming_cluster_tracker import StreamingClusterTracker
from modularnet.plugin.plugin_worker import PluginWorker
from modularnet.plugin import BasePlugin, PluginDescriptor

logger = logging.getLogger(__name__)


class ClusterPlugin(BasePlugin):
    """
    Thin orchestration layer: owns one StreamingClusterTracker per layer_id
    and handles pub/sub + threading. All clustering, drift-gating,
    warm-starting, relabeling, and stabilization logic lives in the tracker.
    """

    @classmethod
    def Descriptor(cls, options: Dict[str, Any] = None) -> PluginDescriptor:
        return super().Descriptor(
            version="2.0.0",
            author="ModularNet Team",
            description=(
                "Plugin that performs adaptive, drift-gated, warm-started "
                "KMeans clustering per layer on a streaming position feed."
            ),
            options=options,
        )

    def __init__(
        self,
        worker: PluginWorker,
        plugin_id: str,
        options: Dict[str, Any],
    ):
        super().__init__(worker, plugin_id, options)

        # Cap FAISS's internal OpenMP thread pool so per-layer clustering
        # threads don't oversubscribe cores against DataLoader workers and
        # other plugin worker processes. Only applied if explicitly set.
        faiss_num_threads = options.get("faiss_num_threads")
        if faiss_num_threads is not None:
            faiss.omp_set_num_threads(faiss_num_threads)

        # Config shared by every per-layer StreamingClusterTracker.
        self._tracker_kwargs = dict(
            metric=options.get("metric", "cosine"),
            k_min=options.get("k_min"),
            k_max=options.get("k_max"),
            max_iter=options.get("max_iter", 100),
            tol=options.get("tol", 1e-5),
            n_init=options.get("n_init", 1),
            device=options.get("device"),
            verbose=options.get("verbose", False),
            precompute_distances=options.get("precompute_distances", True),
            ema_alpha=options.get("ema_alpha", 0.2),
            drift_threshold=options.get("drift_threshold", 0.02),
            k_search_radius=options.get("k_search_radius", 3),
            full_search_every=options.get("full_search_every", 20),
            warm_start_max_iter=options.get("warm_start_max_iter", 10),
            hysteresis_margin=options.get("hysteresis_margin", 0.01),
            debounce_steps=options.get("debounce_steps", 2),
        )
        self.verbose: bool = self._tracker_kwargs["verbose"]

        self.trackers: Dict[str, StreamingClusterTracker] = {}

        # Positions awaiting processing, protected by a lock so a mid-cycle
        # pub/sub update can't be read half-written by main() (the previous
        # single-dict version had no snapshot boundary between the two).
        self._positions_lock = threading.Lock()
        self.positions_pending: Dict[str, np.ndarray] = {}

        # Last published stats per layer_id: {"best_k", "best_score", "labels",
        # "centroids", "silhouette_scores", "ari_vs_previous", "drift"}
        self.cluster_stats: Dict[str, Dict[str, Any]] = {}

    def on_start(self) -> None:
        self.topic_subscribe("positions", self._on_positions_received)

    def _get_tracker(self, layer_id: str) -> StreamingClusterTracker:
        """Get or create the tracker dedicated to a given layer_id."""
        tracker = self.trackers.get(layer_id)
        if tracker is None:
            tracker = StreamingClusterTracker(layer_id=layer_id, **self._tracker_kwargs)
            self.trackers[layer_id] = tracker
        return tracker

    def main(self) -> None:
        with self._positions_lock:
            snapshot = dict(self.positions_pending)

        ths: List[threading.Thread] = []
        for layer_id, positions in snapshot.items():
            th = threading.Thread(target=self.process_positions, args=(layer_id, positions))
            ths.append(th)
            th.start()
        for th in ths:
            th.join()

        self.env_set("cluster_stats", self.cluster_stats)  # Make this plugin accessible to other plugins
        time.sleep(1.0)  # Allow other plugins to access this plugin's state

    def _on_positions_received(self, data: Dict[str, Any]):
        """
        Callback when new positions are received for a given layer.

        Args:
            data: dict containing "positions" (np.ndarray of shape (n_samples, n_features))
                  and "layer_id" (str)
        """
        positions: np.ndarray = data.get("positions")
        layer_id: str = data.get("layer_id")

        with self._positions_lock:
            self.positions_pending[layer_id] = positions

    def process_positions(self, layer_id: str, positions: np.ndarray):
        """
        Run one streaming update for a given layer's tracker and publish the
        result. If the tracker determines the frame is static (drift below
        threshold), it returns None and we republish the last known stats
        instead of doing any clustering work.
        """
        try:
            tracker = self._get_tracker(layer_id)
            stats = tracker.update(positions)

            if stats is None:
                stats = tracker.last_stats
                if stats is None:
                    return  # nothing has ever been published for this layer yet

            self.cluster_stats[layer_id] = stats

            # Publish results
            # NOTE: renamed from "labels" -> "clusters" to match UIPlugin's
            # topic_subscribe("clusters", self._on_clusters_received).
            self.topic_publish("clusters", {
                "labels": stats["labels"],
                "layer_id": layer_id,
                "best_k": stats["best_k"],
                "best_score": stats["best_score"],
                "centroids": stats["centroids"],
            })

            if self.verbose:
                print(
                    f"[ClusterPlugin] layer={layer_id} "
                    f"clustered {len(positions)} points, best K: {stats['best_k']}, "
                    f"score: {stats['best_score']}, drift: {stats.get('drift'):.5f}, "
                    f"ari_vs_prev: {stats.get('ari_vs_previous')}"
                )

        except Exception as e:
            error_msg = f"Error in clustering for layer {layer_id}: {str(e)}"
            logger.error(error_msg)

    def get_cluster_stats(self, layer_id: str) -> Optional[Dict[str, Any]]:
        """Convenience accessor for the last clustering stats of a given layer."""
        return self.cluster_stats.get(layer_id)
