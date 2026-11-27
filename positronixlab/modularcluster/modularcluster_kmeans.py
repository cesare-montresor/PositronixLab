
from typing import Literal

#from fast_pytorch_kmeans import KMeans, MultiKMeans
import faiss
import numpy as np
import torch
import torch.nn.functional as F
from faiss import Kmeans as FAISS_KMeans
from sklearn.metrics import silhouette_score

#from faiss import SuperKMeans as FAISS_KMeans
from positronixlab.utils.utils import suppress_output


class ModularClusterKmeans:
    """
    Modular clustering with adaptive K selection via silhouette score.
    Supports both Euclidean and cosine (spherical) distance metrics.
    """
    EPS = 1e-8
    
    def __init__(
        self,
        metric: Literal['euclidean', 'cosine'] = 'cosine',
        k_min: int | None = None,
        k_max: int | None = None,
        max_iter: int = 100,
        tol: float = 1e-5,
        n_init: int = 1,
        device: torch.device | None = None,
        verbose: bool = False,
        precompute_distances: bool = True
    ):
        """
        Args:
            metric: Distance metric ('euclidean' or 'cosine')
            k_min: Minimum K to try (default: sqrt(n/2))
            k_max: Maximum K to try (default: sqrt(2n))
            max_iter: Maximum iterations per run
            tol: Convergence tolerance
            n_init: Number of random initializations
            device: torch device
            verbose: Print progress
            precompute_distances: Whether to precompute distances for efficiency
        """
        self.metric = metric
        self.k_min = k_min
        self.k_max = k_max
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.device = device or torch.device('cpu')
        self.verbose = verbose
        self.precompute_distances = precompute_distances
        
        
        self.centroids: torch.Tensor | None = None
        self.labels: torch.Tensor | None = None
        self.best_score: float | None = None
        self.best_k: int | None = None
        self.silhouette_scores: dict[int, float] = {}
        self.distance_cache: np.ndarray | None = None



    
    def _compute_k_range(self, n: int) -> tuple[int, int]:
        """Compute heuristic K range based on dataset size."""
        k_min = self.k_min or max(2, int(n ** 0.5 / 2))
        k_max = self.k_max or min(int((2 * n) ** 0.5), n // 2)
        return max(2, k_min), min(k_max, n - 1)
    
    def _distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distances between points and centroids."""
        if self.metric == 'euclidean':
            dist = torch.cdist(x, y, p=2)
        elif self.metric == 'cosine':
            x_norm = F.normalize(x, p=2, dim=1)
            y_norm = F.normalize(y, p=2, dim=1)
            dist = 1 - torch.mm(x_norm, y_norm.T)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

        if x is y:
            dist.fill_diagonal_(0.0) # added for eucledian
        dist.clamp_(min=0.0) # not alwys >=0 due to numerical errors
        return dist
    
    def _silhouette_score(self, x: torch.Tensor, labels: torch.Tensor, k: int) -> float:
        if k < 2:
            return -1.0  # Silhouette score undefined for k < 2
        
        labels_np = labels.cpu().numpy()
        if len(np.unique(labels_np)) < 2:
            return -1.0  # Silhouette score undefined when fewer than 2 clusters are populated

        if self.precompute_distances:
            score = silhouette_score(self.distance_cache, labels_np, metric='precomputed')
        else:
            score = silhouette_score(x.cpu().numpy(), labels_np, metric=self.metric)
        
        return score
    
    def _kmeans(self, x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Run KMeans clustering for a single K value."""
        spherical = (self.metric == 'cosine')

        if spherical:
            x = F.normalize(x, p=2, dim=1)
        
        x_np = x.detach().cpu().numpy().astype('float32', copy=False)
        use_gpu = self.device.type == 'cuda' and hasattr(faiss, 'StandardGpuResources')
        with suppress_output():
            kmeans = FAISS_KMeans(
                d=x_np.shape[1], 
                k=k, 
                niter=self.max_iter,  
                spherical=spherical, 
                verbose=self.verbose,
                nredo=self.n_init,
                gpu=use_gpu,
            )
            
            kmeans.train(x_np)

        _, labels_arr = kmeans.assign(x_np)
        labels = torch.as_tensor(labels_arr, dtype=torch.long, device=self.device)
        centroids = torch.from_numpy(kmeans.centroids).to(self.device)
        return centroids, labels
    
    def _evaluate_k(self, x: torch.Tensor, k: int) -> float:
        """Evaluate a single K value and update best if needed."""
        if k in self.silhouette_scores:
            return self.silhouette_scores[k]
        
        centroids, labels = self._kmeans(x, k)
        score = self._silhouette_score(x, labels, k)
        self.silhouette_scores[k] = score
        
        if self.verbose:
            print(f"K={k}: silhouette={score:.4f},")
        
        # Update best
        if self.best_k is None or score > self.silhouette_scores[self.best_k]:
            self.centroids = centroids
            self.labels = labels
            self.best_score = score
            self.best_k = k
        
        return score
    
    def _adaptive_search(self, x: torch.Tensor, k_min: int, k_max: int):
        """Adaptive K search: coarse sampling → peak detection → refinement."""
        k_range = k_max - k_min
        
        
        if k_range <= 10:
            # Small range: exhaustive search
            for k in range(k_min, k_max + 1):
                self._evaluate_k(x, k)
            return
        
        # Phase 1: Coarse sampling (~10 points)
        n_samples = min(10, k_range)
        sample_points = {k_min, k_max}
        
        for i in range(1, n_samples - 1):
            k = int(k_min + (i / (n_samples - 1)) * k_range)
            sample_points.add(k)
        
        scores = {k: self._evaluate_k(x, k) for k in sorted(sample_points)}
        
        # Phase 2: Find peaks
        sorted_ks = sorted(scores.keys())
        peaks = []
        
        for i in range(1, len(sorted_ks) - 1):
            k_prev, k_curr, k_next = sorted_ks[i-1:i+2]
            if scores[k_curr] > scores[k_prev] and scores[k_curr] > scores[k_next]:
                peaks.append(k_curr)
        
        if scores[sorted_ks[0]] >= scores[sorted_ks[1]]:
            peaks.append(sorted_ks[0])
        if scores[sorted_ks[-1]] >= scores[sorted_ks[-2]]:
            peaks.append(sorted_ks[-1])
        
        # Phase 3: Refine around peaks
        for peak in peaks:
            idx = sorted_ks.index(peak)
            left = sorted_ks[idx - 1] if idx > 0 else k_min
            right = sorted_ks[idx + 1] if idx < len(sorted_ks) - 1 else k_max
            
            for k in range(max(peak - 3, left), min(peak + 4, right + 1)):
                if k_min <= k <= k_max:
                    self._evaluate_k(x, k)
    
    def fit(self, x: torch.Tensor) -> 'ModularClusterKmeans':
        """
        Fit KMeans with adaptive K selection.
        
        Args:
            x: Input embeddings of shape (n_samples, n_features)
        """
        x = x.to(self.device)
        n = x.shape[0]
        
        if n < 2:
            raise ValueError("Need at least 2 samples for clustering")
        
        k_min, k_max = self._compute_k_range(n)
        
        if self.verbose:
            print(f"Adaptive search for K in [{k_min}, {k_max}]")
        
        self.silhouette_scores = {}
        self.best_k = None
        self.best_score = None
        self.centroids = None
        self.labels = None
        self.distance_cache = None

        if k_min > k_max:
            self.best_k = 1
            self.best_score = -1.0
            self.silhouette_scores[1] = -1.0
            self.labels = torch.zeros(n, dtype=torch.long, device=self.device)
            self.centroids = x.mean(dim=0, keepdim=True)
            return self

        if self.precompute_distances:
            dists = self._distance(x, x)
            self.distance_cache = dists.cpu().numpy()

        
        self._adaptive_search(x, k_min, k_max)
        
        if self.verbose:
            best_score = self.silhouette_scores[self.best_k] if self.best_k is not None else -1.0
            print(f"\nBest K: {self.best_k} (silhouette={best_score:.4f})")
            print(f"Evaluated {len(self.silhouette_scores)}/{k_max - k_min + 1} K values")
        
        return self
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict cluster labels for new points."""
        if self.centroids is None:
            raise ValueError("Must call fit() before predict()")
        
        x = x.to(self.device)
        dists = self._distance(x, self.centroids)
        return dists.argmin(dim=1)
    
    def fit_predict(self, x: torch.Tensor) -> torch.Tensor:
        """Fit and return cluster labels."""
        self.fit(x)
        if self.labels is None:
            raise RuntimeError("Clustering fit() did not produce labels.")
        return self.labels
    
    def print_stats(self):
        """Print clustering statistics."""
        if self.centroids is None or self.labels is None or self.best_k is None:
            print("Model not fitted yet.")
            return
        
        print(f"Best K: {self.best_k}")
        print(f"Best score: {self.best_score:.4f}")
        print(f"Silhouette Score: {self.silhouette_scores[self.best_k]:.4f}")
        print(f"Cluster sizes: {torch.bincount(self.labels).tolist()}")
        print(f"Evaluated K values: {sorted(self.silhouette_scores.keys())}")
        






