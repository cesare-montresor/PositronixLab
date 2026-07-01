
"""
PluginModularCluster: Pub/Sub-based clustering plugin.

Subscribes to "cu_positions" topic and publishes cluster labels
to "cu_labels" topic using the modular clustering system.
"""
from typing_extensions import override
import torch
from typing import Literal, Optional

from modularnet.modularcluster.modularcluster_kmeans import ModularClusterKmeans
from modularnet.plugin import BasePlugin


class ModularClusterPlugin(BasePlugin):
    """
    Pub/Sub-based clustering plugin.
    
    
    Args:
        metric: Distance metric ('euclidean' or 'cosine'). Default: 'cosine'
        k_min: Minimum K for adaptive search. Default: sqrt(n/2)
        k_max: Maximum K for adaptive search. Default: sqrt(2n)
        max_iter: Maximum KMeans iterations. Default: 100
        tol: Convergence tolerance. Default: 1e-5
        n_init: Number of random initializations. Default: 1
        device: torch device. Default: 'cpu'
        verbose: Print progress. Default: False
        precompute_distances: Precompute distance matrix. Default: True
        fatal_errors: Whether to raise errors or log them. Default: False
        name: Plugin name. Default: class name
    """
    
    INPUT_TOPIC = "cu_positions"
    OUTPUT_TOPIC = "cu_labels"
    
    def __init__(
        self,
        metric: Literal['euclidean', 'cosine'] = 'cosine',
        k_min: Optional[int] = None,
        k_max: Optional[int] = None,
        max_iter: int = 100,
        tol: float = 1e-5,
        n_init: int = 1,
        device: Optional[torch.device] = None,
        verbose: bool = False,
        precompute_distances: bool = True,
        fatal_errors: bool = False,
        name: Optional[str] = None
    ):
        super().__init__(fatal_errors=fatal_errors, name=name)
        
        # Initialize clusterer with provided options
        self.clusterer = ModularClusterKmeans(
            metric=metric,
            k_min=k_min,
            k_max=k_max,
            max_iter=max_iter,
            tol=tol,
            n_init=n_init,
            device=device,
            verbose=verbose,
            precompute_distances=precompute_distances
        )
        
        # Store configuration
        self.verbose = verbose

    def onStart(self):
        """Subscribe to input topic on plugin start."""
        self.topic_subscribe(self.INPUT_TOPIC, self._on_positions_received)


    def _on_positions_received(self, positions: torch.Tensor):
        """
        Callback when new positions are received.
        
        Args:
            positions: Tensor of shape (n_samples, n_features)
        """
        try:
            if not isinstance(positions, torch.Tensor):
                positions = torch.tensor(positions)
            
            # Perform clustering
            labels = self.clusterer.fit_predict(positions)
            
            # Publish results
            self.topic_publish(self.OUTPUT_TOPIC, labels)
            
            if self.verbose:
                print(f"[PluginModularCluster] Clustered {len(positions)} points, "
                      f"best K: {self.clusterer.best_k}")
                
        except Exception as e:
            error_msg = f"Error in clustering: {str(e)}"
            if self.verbose:
                print(f"[PluginModularCluster] {error_msg}")
            if self.fatal_errors:
                raise RuntimeError(error_msg) from e


    
