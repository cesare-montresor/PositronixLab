
import logging
import math
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Processors: elaborazione a valle del reducer (gira nel loop di main)
# ---------------------------------------------------------
class Processor(ABC):
    @abstractmethod
    def process(self, items: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
        """items: chiave -> tensore CPU aggregato di una emissione.
        Ritorna messaggi {"key", "value", **extra}. Un campo "topic" instrada
        il messaggio su un altro topic (e lo esclude da storico ed env)."""


# Assuming UMAP is imported from either umap-learn or cuML
try:
    from cuml.manifold import UMAP
except ImportError:
    try:
        from umap import UMAP
    except ImportError:
        UMAP = None


class UmapProcessor(Processor):
    """
    UMAP projection per-layer. Collects `n_samples` snapshots to fit the manifold,
    then uses UMAP's warm-start fit_transform() to embed new points.

    Dynamic Scaling: n_neighbors scales automatically via a bounded sqrt(num_neurons)
    to balance local topology and compute budget across different layer sizes.

    Cluster centroids are NOT projected through UMAP.transform(): they are
    computed in 2D as the mean of the member neurons' 2D positions.
    """
    def __init__(
        self,
        n_components: int = 2,
        n_samples: int = 50,      # Lowered from 200 due to upstream DisplacementFilter
        min_dist: float = 0.1,
        min_neighbors: int = 10,  # Safety floor for small layers (prevents fracture)
        max_neighbors: int = 50,  # Safety ceiling for large layers (prevents compute lock)
        seed: int | None = 42
    ) -> None:
        self.n_components = n_components
        self.n_samples = n_samples
        self.min_dist = min_dist
        self.min_neighbors = min_neighbors
        self.max_neighbors = max_neighbors
        self.seed = seed

        self._lock = threading.Lock()
        self._mappers: dict[str, Any] = {}
        self._collect: dict[str, list[np.ndarray]] = {}
        self._steps: dict[str, int] = {}

        self._last_2d: dict[str, np.ndarray] = {}

    @staticmethod
    def _to_numpy(arr: Any) -> np.ndarray:
        if hasattr(arr, "get"):  # cupy (cuML) -> numpy
            arr = arr.get()
        return np.asarray(arr, dtype=np.float32)

    def _step(self, layer: str, pts2d: np.ndarray) -> dict[str, Any]:
        pts2d = self._to_numpy(pts2d)
        i = self._steps.get(layer, 0)
        self._steps[layer] = i + 1
        value = torch.from_numpy(np.ascontiguousarray(pts2d, dtype=np.float32))
        return {"key": layer, "value": value, "index": i}

    def _create_mapper(self, num_neurons: int, init: str | np.ndarray = "spectral") -> Any:
        """Factory to instantiate UMAP with dynamically scaled neighbors."""
        # 1. Calculate dynamic neighbors using Square Root scaling
        calculated_k = int(math.sqrt(num_neurons))

        # 2. Clamp between topological safety (min) and compute safety (max),
        #    while never exceeding num_neurons - 1 on tiny layers.
        dynamic_n_neighbors = max(self.min_neighbors, min(calculated_k, self.max_neighbors))
        if num_neurons > 1:
            dynamic_n_neighbors = min(dynamic_n_neighbors, num_neurons - 1)
        else:
            dynamic_n_neighbors = 1

        if UMAP is None:
            raise ImportError(
                "UmapProcessor requires `umap-learn` or `cuml`. "
                "Install with: pip install 'positronixlab[umap]'"
            )

        return UMAP(
            n_components=self.n_components,
            n_neighbors=dynamic_n_neighbors,
            min_dist=self.min_dist,
            random_state=self.seed,
            init=init
        )

    def process(self, items: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        with self._lock:
            for layer, t in items.items():
                pts = t.numpy()
                num_neurons = pts.shape[0]
                mapper = self._mappers.get(layer)

                if mapper is not None:
                    # WARM-START UPDATE:
                    # Create a new UMAP instance anchored to the previous frame
                    # (fall back to spectral init if neuron count changed)
                    prev_2d = self._to_numpy(self._last_2d[layer])
                    init_arg: str | np.ndarray = (
                        prev_2d if prev_2d.shape[0] == num_neurons else "spectral"
                    )
                    mapper = self._create_mapper(
                        num_neurons=num_neurons,
                        init=init_arg,
                    )

                    transformed = self._to_numpy(mapper.fit_transform(pts))

                    # Update states
                    self._mappers[layer] = mapper
                    self._last_2d[layer] = transformed

                    out.append(self._step(layer, transformed))
                    continue

                buf = self._collect.setdefault(layer, [])
                if buf and buf[0].shape[1:] != pts.shape[1:]:
                    buf.clear()

                buf.append(pts.copy())

                if len(buf) >= self.n_samples:
                    # Initialize based on the layer size (not the stacked buffer size)
                    # to maintain neighbor scaling consistency during the streaming phase
                    mapper = self._create_mapper(num_neurons=num_neurons, init="spectral")

                    stacked_buf = np.concatenate(buf, axis=0)
                    mapper.fit(stacked_buf)
                    self._mappers[layer] = mapper

                    transformed_buf = self._to_numpy(mapper.transform(stacked_buf))

                    # STORE THE INITIAL WARM-START ANCHOR:
                    # 2D coordinates of the very last snapshot in the buffer
                    last_snap_len = buf[-1].shape[0]
                    self._last_2d[layer] = transformed_buf[-last_snap_len:]

                    offset = 0
                    for snap in buf:
                        snap_len = snap.shape[0]
                        out.append(self._step(layer, transformed_buf[offset:offset + snap_len]))
                        offset += snap_len

                    del self._collect[layer]
        return out

    def ready(self, layer: str) -> bool:
        with self._lock:
            return layer in self._mappers

    def last_index(self, layer: str) -> int:
        with self._lock:
            return self._steps.get(layer, 0) - 1

    def centroids_2d(self, layer: str, labels: np.ndarray) -> np.ndarray | None:
        """2D centroids = mean of member neurons' latest 2D positions.

        Returns None if the layer has no 2D frame yet or if `labels` does not
        match the neuron count of the latest frame (e.g. a stale snapshot).
        Row i corresponds to the i-th unique label (sorted), as in np.unique.
        """
        with self._lock:
            pts = self._last_2d.get(layer)
            if pts is None:
                return None
            pts = self._to_numpy(pts)

        labels = np.asarray(labels).reshape(-1)
        if labels.shape[0] != pts.shape[0]:
            return None

        ks = np.unique(labels)
        if ks.size == 0:
            return np.empty((0, pts.shape[1]), dtype=np.float32)
        return np.stack([pts[labels == k].mean(axis=0) for k in ks]).astype(np.float32)

    def progress(self) -> dict[str, Any]:
        with self._lock:
            return {"required": self.n_samples,
                    "layers": {layer: len(b) for layer, b in self._collect.items()}}

    def collecting(self) -> dict[str, list[np.ndarray]]:
        with self._lock:
            return {layer: list(b) for layer, b in self._collect.items()}

class OverallProcessor(Processor):
    """Pass-through, piu' un messaggio extra su `topic` con la media tra le chiavi
    dell'ultimo valore noto di ciascuna (es. spaceloss medio su tutti i layer)."""

    def __init__(self, topic: str, key: str = "overall") -> None:
        self.topic = topic
        self.key = key
        self._latest: dict[str, torch.Tensor] = {}

    def process(self, items):
        self._latest.update(items)
        if not self._latest:
            return []
        msgs = [{"key": k, "value": v} for k, v in items.items()]
        overall = torch.stack(list(self._latest.values())).mean(0)
        msgs.append({"topic": self.topic, "key": self.key, "value": overall})
        return msgs


PROCESSORS: dict[str, Callable[..., Processor]] = {
    "umap": UmapProcessor, 
    "overall": OverallProcessor
}