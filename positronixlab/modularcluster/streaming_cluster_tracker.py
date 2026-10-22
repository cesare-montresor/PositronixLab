
from typing import Any, Dict, Optional

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from positronixlab.modularcluster.modularcluster_kmeans_stream import ModularClusterKmeansStream

import logging
logger = logging.getLogger(__name__)


class StreamingClusterTracker:
    """
    Owns all per-layer streaming state for a continuously-evolving position
    stream (fixed point count, smooth motion, no teleportation between steps).

    Responsibilities, all scoped to a single layer_id:
      - EMA-based drift gating: skip clustering entirely on near-static frames.
      - Warm-started, narrowed-range clustering on continuation frames, with
        periodic full-range search to catch real splits/merges.
      - Hungarian-matching relabeling so cluster IDs stay consistent frame to
        frame (a physical cluster keeps its label id across timesteps).
      - ARI vs. the previous frame as a diagnostic/trigger signal.
      - Hysteresis + debounce on accepting a change in k, so the published
        cluster count doesn't flap on noisy near-tied silhouette scores.

    ClusterPlugin should hold one instance of this per layer_id and just call
    update(positions) once per cycle; it owns no clustering logic itself.
    """

    def __init__(
        self,
        layer_id: str,
        metric: str = "cosine",
        k_min: Optional[int] = None,
        k_max: Optional[int] = None,
        max_iter: int = 100,
        tol: float = 1e-5,
        n_init: int = 1,
        device: Optional[torch.device] = None,
        verbose: bool = False,
        precompute_distances: bool = True,
        # --- streaming-specific knobs ---
        ema_alpha: float = 0.2,
        drift_threshold: float = 0.02,
        k_search_radius: int = 3,
        full_search_every: int = 20,
        warm_start_max_iter: int = 10,
        hysteresis_margin: float = 0.01,
        debounce_steps: int = 2,
    ):
        """
        Args:
            layer_id: Identifier for the layer this tracker owns.
            metric/k_min/k_max/max_iter/tol/n_init/device/verbose/
                precompute_distances: passed straight through to the
                underlying ModularClusterKmeans.
            ema_alpha: EMA decay for the drift-gating baseline. Lower = smoother
                baseline, more sensitive to slow accumulated drift.
            drift_threshold: Mean abs deviation from the EMA baseline below
                which a frame is considered static and clustering is skipped.
            k_search_radius: Half-width of the narrow k window searched around
                the last accepted k on continuation (non-full-search) frames.
            full_search_every: Force a full-range k search at least this often,
                regardless of drift, to catch changes a narrow window would miss.
            warm_start_max_iter: FAISS niter used on warm-started continuation
                frames (kept low since centroids start close to converged).
            hysteresis_margin: A new k must beat the previously published k's
                silhouette score by more than this margin to be considered.
            debounce_steps: A new k must win by the hysteresis margin for this
                many consecutive evaluations before it is actually published.
        """
        self.layer_id = layer_id
        self.kmeans = ModularClusterKmeansStream(
            metric=metric,
            k_min=k_min,
            k_max=k_max,
            max_iter=max_iter,
            tol=tol,
            n_init=n_init,
            device=device,
            verbose=verbose,
            precompute_distances=precompute_distances,
        )

        self.ema_alpha = ema_alpha
        self.drift_threshold = drift_threshold
        self.k_search_radius = k_search_radius
        self.full_search_every = full_search_every
        self.warm_start_max_iter = warm_start_max_iter
        self.hysteresis_margin = hysteresis_margin
        self.debounce_steps = debounce_steps

        # Streaming state
        self.ema_positions: Optional[np.ndarray] = None
        self.last_published_labels: Optional[np.ndarray] = None
        self.last_published_centroids: Optional[np.ndarray] = None
        self.last_published_k: Optional[int] = None
        self.last_stats: Optional[Dict[str, Any]] = None

        self._steps_since_full_search = 0
        self._pending_k: Optional[int] = None
        self._pending_k_streak = 0

    def _drift(self, positions: np.ndarray) -> float:
        """Mean absolute deviation of new positions from the EMA baseline.
        Returns inf on the first frame (or a shape change) to force clustering."""
        if self.ema_positions is None or self.ema_positions.shape != positions.shape:
            return float("inf")
        return float(np.abs(positions - self.ema_positions).mean())

    def _update_ema(self, positions: np.ndarray) -> None:
        if self.ema_positions is None or self.ema_positions.shape != positions.shape:
            self.ema_positions = positions.copy()
        else:
            self.ema_positions = (
                self.ema_alpha * positions + (1 - self.ema_alpha) * self.ema_positions
            )

    def _relabel_to_match_previous(
        self, centroids: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Remap new cluster label IDs to best match the previous timestep's IDs
        via Hungarian assignment on centroid distances, so the same physical
        cluster keeps the same label id across steps. Only applies when k is
        unchanged from the previous published frame; a k change is a separate
        concern handled by the caller (this method just returns labels as-is
        in that case, since a clean 1:1 assignment doesn't exist).
        """
        if (
            self.last_published_centroids is None
            or self.last_published_centroids.shape[0] != centroids.shape[0]
        ):
            return labels

        prev = torch.as_tensor(
            self.last_published_centroids, device=centroids.device, dtype=centroids.dtype
        )
        cost = torch.cdist(prev, centroids).cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost)

        remap = np.empty(centroids.shape[0], dtype=np.int64)
        remap[col_ind] = row_ind  # new_label_id -> old_label_id
        remap_t = torch.as_tensor(remap, device=labels.device)
        return remap_t[labels]

    def _gate_k_change(self, new_k: int) -> int:
        """
        Only accept a change away from the last published k if the new k wins
        by more than hysteresis_margin AND does so for debounce_steps
        consecutive evaluations in a row. Otherwise keep reporting the last
        published k so downstream consumers don't see the cluster count flap
        on noisy near-tied silhouette scores.
        """
        if self.last_published_k is None or new_k == self.last_published_k:
            self._pending_k = None
            self._pending_k_streak = 0
            return new_k

        prev_score = self.kmeans.silhouette_scores.get(self.last_published_k)
        new_score = self.kmeans.silhouette_scores.get(new_k, self.kmeans.best_score)

        wins_by_margin = prev_score is None or (new_score - prev_score) > self.hysteresis_margin
        if not wins_by_margin:
            self._pending_k = None
            self._pending_k_streak = 0
            return self.last_published_k

        if self._pending_k == new_k:
            self._pending_k_streak += 1
        else:
            self._pending_k = new_k
            self._pending_k_streak = 1

        if self._pending_k_streak >= self.debounce_steps:
            return new_k
        return self.last_published_k

    def update(self, positions: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Process one timestep of positions for this layer.

        Returns a stats dict to publish, or None if this frame was skipped
        outright because drift was below threshold (caller should republish
        self.last_stats in that case, e.g. to keep downstream consumers'
        timestamps fresh without redoing any work).
        """
        drift = self._drift(positions)
        self._steps_since_full_search += 1
        force_full_search = self._steps_since_full_search >= self.full_search_every

        if drift < self.drift_threshold and not force_full_search and self.last_stats is not None:
            # Static frame: skip clustering entirely, just keep the EMA fresh.
            self._update_ema(positions)
            return None

        x = torch.as_tensor(positions)

        if force_full_search or self.last_published_k is None:
            self._steps_since_full_search = 0
            self.kmeans.fit(x)
        else:
            self.kmeans.fit(
                x,
                warm_start_k=self.last_published_k,
                k_search_radius=self.k_search_radius,
                max_iter_override=self.warm_start_max_iter,
            )

        self._update_ema(positions)

        new_k = self.kmeans.best_k
        accepted_k = self._gate_k_change(new_k)

        if accepted_k != new_k:
            # Held back by hysteresis/debounce: republish the previous
            # cluster assignment unchanged, just refresh diagnostics.
            stats = dict(self.last_stats)
            stats["drift"] = drift
            stats["ari_vs_previous"] = 1.0  # unchanged by definition
            self.last_stats = stats
            return stats

        labels = self._relabel_to_match_previous(self.kmeans.centroids, self.kmeans.labels)

        ari = None
        if (
            self.last_published_labels is not None
            and len(self.last_published_labels) == len(labels)
        ):
            ari = adjusted_rand_score(self.last_published_labels, labels.cpu().numpy())

        centroids_np = self.kmeans.centroids.cpu().numpy()
        labels_np = labels.cpu().numpy()

        self.last_published_labels = labels_np
        self.last_published_centroids = centroids_np
        self.last_published_k = accepted_k

        stats = {
            "best_k": accepted_k,
            "best_score": self.kmeans.best_score,
            "labels": labels_np,
            "centroids": centroids_np,
            "silhouette_scores": dict(self.kmeans.silhouette_scores),
            "ari_vs_previous": ari,
            "drift": drift,
        }
        self.last_stats = stats
        return stats
