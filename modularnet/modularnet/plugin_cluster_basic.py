import logging
import threading
import time
from typing import Any, Dict, List, Literal, Optional
import numpy as np
import torch

from modularnet.modularcluster.modularcluster_kmeans import ModularClusterKmeans
from modularnet.plugin.plugin_worker import PluginWorker
from modularnet.plugin import BasePlugin, PluginDescriptor

logger = logging.getLogger(__name__)


class ClusterPluginBasic(BasePlugin):

    @classmethod
    def Descriptor(cls, options: Dict[str, Any] = None) -> PluginDescriptor:
        return super().Descriptor(
            version="1.0.0",
            author="ModularNet Team",
            description="Plugin that performs adaptive KMeans clustering per layer on incoming positions.",
            options=options
        )

    def __init__(
        self,
        worker: PluginWorker,
        plugin_id: str,
        options: Dict[str, Any],
    ):
        super().__init__(worker, plugin_id, options)

        # Clustering config (shared across all per-layer clusterers)
        self.metric: Literal['euclidean', 'cosine'] = options.get("metric", "cosine")
        self.k_min: Optional[int] = options.get("k_min")
        self.k_max: Optional[int] = options.get("k_max")
        self.max_iter: int = options.get("max_iter", 100)
        self.tol: float = options.get("tol", 1e-5)
        self.n_init: int = options.get("n_init", 1)
        self.device: Optional[torch.device] = options.get("device")
        self.verbose: bool = options.get("verbose", False)
        self.precompute_distances: bool = options.get("precompute_distances", True)

        # One ModularClusterKmeans instance per layer_id
        self.positions_last: Dict[str, np.ndarray] = {}

        # One ModularClusterKmeans instance per layer_id
        self.clusterers: Dict[str, ModularClusterKmeans] = {}

        # Last results per layer_id: {"best_k", "best_score", "labels", "centroids", "silhouette_scores"}
        self.cluster_stats: Dict[str, Dict[str, Any]] = {}

    def on_start(self) -> None:
        self.topic_subscribe("positions", self._on_positions_received)

    def main(self) -> None:
        ths: List[threading.Thread] = []
        for layer_id, positions in self.positions_last.items():
            th = threading.Thread(target=self.process_positions, args=(layer_id, positions))
            ths.append(th)
            th.start()
        for th in ths:
            th.join()
        
        self.env_set("cluster_stats", self.cluster_stats)  # Make this plugin accessible to other plugins
        time.sleep(1.0)  # Allow other plugins to access this plugin's state
        

    def _get_clusterer(self, layer_id: str) -> ModularClusterKmeans:
        """Get or create the clusterer dedicated to a given layer_id."""
        clusterer = self.clusterers.get(layer_id)
        if clusterer is None:
            clusterer = ModularClusterKmeans(
                metric=self.metric,
                k_min=self.k_min,
                k_max=self.k_max,
                max_iter=self.max_iter,
                tol=self.tol,
                n_init=self.n_init,
                device=self.device,
                verbose=self.verbose,
                precompute_distances=self.precompute_distances,
            )
            self.clusterers[layer_id] = clusterer
        return clusterer

    def _on_positions_received(self, data: Dict[str, Any]):
        """
        Callback when new positions are received for a given layer.

        Args:
            data: dict containing "positions" (Tensor of shape (n_samples, n_features))
                  and "layer_id" (str)
        """
        positions: np.ndarray = data.get("positions")
        layer_id: str = data.get("layer_id")

        self.positions_last[layer_id] = positions  # Store last positions for this layer


    def process_positions(self, layer_id: str, positions: np.ndarray):
        """
        Process positions for a given layer.

        Args:
            layer_id: ID of the layer
            positions: NumPy array of shape (n_samples, n_features)
        """

        try:
            if not isinstance(positions, torch.Tensor):
                positions = torch.tensor(positions)

            clusterer = self._get_clusterer(layer_id)

            # Perform clustering (this updates clusterer.best_k/best_score/labels/etc.)
            labels = clusterer.fit_predict(positions)

            # Save last stats for this layer
            self.cluster_stats[layer_id] = {
                "best_k": clusterer.best_k,
                "best_score": clusterer.best_score,
                "labels": labels,
                "centroids": clusterer.centroids,
                "silhouette_scores": dict(clusterer.silhouette_scores),
            }

            # Publish results
            self.topic_publish("clusters", {
                "labels": labels,
                "layer_id": layer_id,
                "best_k": clusterer.best_k,
                "best_score": clusterer.best_score,
                "centroids": clusterer.centroids,
            })

            if self.verbose:
                print(
                    f"[ModularClusterPlugin] layer={layer_id} "
                    f"clustered {len(positions)} points, best K: {clusterer.best_k}, "
                    f"score: {clusterer.best_score}"
                )

        except Exception as e:
            error_msg = f"Error in clustering for layer {layer_id}: {str(e)}"
            logger.error(error_msg)

    def get_cluster_stats(self, layer_id: str) -> Optional[Dict[str, Any]]:
        """Convenience accessor for the last clustering stats of a given layer."""
        return self.cluster_stats.get(layer_id)