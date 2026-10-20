import time
from typing_extensions import override
import torch
import torch.nn.functional as F
from typing import Dict, Literal, Optional, Tuple
import os
import contextlib

from sklearn.metrics import silhouette_score

from faiss import Kmeans as FAISS_KMeans

from positronixlab.utils.utils import suppress_output


class ModularClusterKmeansStream:
    """
    Modular clustering with adaptive K selection via silhouette score.
    Supports both Euclidean and cosine (spherical) distance metrics.

    Supports warm-starting: callers (typically a streaming tracker) can pass
    warm_start_k / k_search_radius to search a small window around a known-good
    k instead of the full heuristic range, and max_iter_override to shorten
    FAISS training when continuing from a nearby previous solution. Centroids
    are cached per-k across fit() calls so repeated calls at the same k reuse
    the last converged centroids as FAISS's init_centroids, which is the core
    mechanism that makes warm-starting actually cheap.
    """
    EPS = 1e-8

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
        precompute_distances: bool = True
    ):
        """
        Args:
            metric: Distance metric ('euclidean' or 'cosine')
            k_min: Minimum K to try (default: sqrt(n/2))
            k_max: Maximum K to try (default: sqrt(2n))
            max_iter: Maximum iterations per run (used unless overridden per-call)
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

        self.centroids = None
        self.labels = None
        self.best_score = None
        self.best_k = None
        self.silhouette_scores = {}
        self.distance_cache = None

        # Effective max_iter for the current fit() call (may be overridden
        # per-call for warm-started continuations); reset after each fit().
        self._effective_max_iter = max_iter

        # Per-k centroid cache, persists ACROSS fit() calls (not reset in
        # fit()) so consecutive timesteps can warm-start FAISS training.
        # Keyed by k -> np.ndarray of shape (k, n_features).
        self._centroid_cache: Dict[int, "object"] = {}

    def _compute_k_range(self, n: int) -> Tuple[int, int]:
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

        dist.fill_diagonal_(0.0)  # added for eucledian
        dist.clamp_(min=0.0)  # not alwys >=0 due to numerical errors
        return dist

    def _silhouette_score(self, x: torch.Tensor, labels: torch.Tensor, k: int) -> float:
        if k < 2:
            return -1.0  # Silhouette score undefined for k < 2

        labels = labels.cpu().numpy()

        if self.precompute_distances:
            score = silhouette_score(self.distance_cache, labels, metric='precomputed')
        else:
            score = silhouette_score(x.cpu().numpy(), labels, metric=self.metric)

        return score

    def _kmeans(self, x: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run KMeans clustering for a single K value, warm-starting from a
        cached centroid set for this k if one exists and matches dimensions."""
        spherical = (self.metric == 'cosine')

        if spherical:
            x = F.normalize(x, p=2, dim=1)

        init_centroids = self._centroid_cache.get(k)
        if init_centroids is not None and init_centroids.shape[1] != x.shape[1]:
            init_centroids = None  # stale cache entry (feature dim changed)

        with suppress_output():
            kmeans = FAISS_KMeans(
                d=x.shape[1],
                k=k,
                niter=self._effective_max_iter,
                spherical=spherical,
                verbose=self.verbose,
                nredo=self.n_init
                # gpu=True
            )

            if init_centroids is not None:
                kmeans.train(x, init_centroids=init_centroids)
            else:
                kmeans.train(x)

        centroids = torch.from_numpy(kmeans.centroids).to(self.device)
        # Cache converged centroids for this k so the next fit() at the same
        # k (typically the following timestep) can warm-start from here.
        self._centroid_cache[k] = kmeans.centroids.copy()

        # Assign labels based on nearest centroid
        dists = self._distance(x, centroids)
        labels = dists.argmin(dim=1)
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

    def fit(
        self,
        x: torch.Tensor,
        warm_start_k: Optional[int] = None,
        k_search_radius: Optional[int] = None,
        max_iter_override: Optional[int] = None,
    ) -> 'ModularClusterKmeansStream':
        """
        Fit KMeans with adaptive K selection.

        Args:
            x: Input embeddings of shape (n_samples, n_features)
            warm_start_k: If given (together with k_search_radius), search only
                a narrow window [warm_start_k - radius, warm_start_k + radius]
                instead of the full heuristic range. Intended for streaming
                continuation steps where the previous best_k is a good prior.
            k_search_radius: Half-width of the narrow search window around
                warm_start_k. Ignored if warm_start_k is None.
            max_iter_override: If given, use this as FAISS's niter for this
                call instead of self.max_iter (typically a small number for
                warm-started continuations, since centroids are already close).
        """
        x = x.to(self.device)
        n = x.shape[0]

        if n < 2:
            raise ValueError("Need at least 2 samples for clustering")

        self.silhouette_scores = {}
        self.best_k = None
        self.best_score = None
        self.centroids = None
        self.labels = None
        self.distance_cache = None

        self._effective_max_iter = max_iter_override or self.max_iter

        if self.precompute_distances:
            dists = self._distance(x, x)
            self.distance_cache = dists.cpu().numpy()

        if warm_start_k is not None and k_search_radius is not None:
            k_min = max(2, warm_start_k - k_search_radius)
            k_max = min(warm_start_k + k_search_radius, n - 1)
            if self.verbose:
                print(f"Warm-started search for K in [{k_min}, {k_max}] (center={warm_start_k})")
            for k in range(k_min, k_max + 1):
                self._evaluate_k(x, k)
        else:
            k_min, k_max = self._compute_k_range(n)
            if self.verbose:
                print(f"Adaptive search for K in [{k_min}, {k_max}]")
            self._adaptive_search(x, k_min, k_max)

        # Reset to the configured default so a future cold-start fit() (no
        # overrides) doesn't inherit a leftover reduced iteration count.
        self._effective_max_iter = self.max_iter

        if self.verbose and self.best_k is not None:
            best_score = self.silhouette_scores[self.best_k]
            print(f"\nBest K: {self.best_k} (silhouette={best_score:.4f})")
            print(f"Evaluated {len(self.silhouette_scores)} K values")

        return self

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict cluster labels for new points."""
        if self.centroids is None:
            raise ValueError("Must call fit() before predict()")

        x = x.to(self.device)
        dists = self._distance(x, self.centroids)
        return dists.argmin(dim=1)

    def fit_predict(self, x: torch.Tensor, **fit_kwargs) -> torch.Tensor:
        """Fit and return cluster labels. Accepts the same optional
        warm-start kwargs as fit()."""
        self.fit(x, **fit_kwargs)
        return self.labels

    def print_stats(self):
        """Print clustering statistics."""
        if self.centroids is None or self.labels is None:
            print("Model not fitted yet.")
            return

        print(f"Best K: {self.best_k}")
        print(f"Best score: {self.best_score:.4f}")
        print(f"Silhouette Score: {self.silhouette_scores[self.best_k]:.4f}")
        print(f"Cluster sizes: {torch.bincount(self.labels).tolist()}")
        print(f"Evaluated K values: {sorted(self.silhouette_scores.keys())}")
