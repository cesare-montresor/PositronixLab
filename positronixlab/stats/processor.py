
import torch
from abc import ABC, abstractmethod
from typing import Dict, Callable, List, Any, Optional
import numpy as np
from sklearn.decomposition import PCA
import threading

import logging

logger = logging.getLogger(__name__)


from cuml.manifold.umap import UMAP


# ---------------------------------------------------------
# Processors: elaborazione a valle del reducer (gira nel loop di main)
# ---------------------------------------------------------
class Processor(ABC):
    @abstractmethod
    def process(self, items: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
        """items: chiave -> tensore CPU aggregato di una emissione.
        Ritorna messaggi {"key", "value", **extra}. Un campo "topic" instrada
        il messaggio su un altro topic (e lo esclude da storico ed env)."""

class UmapProcessor(Processor):
    """
    UMAP projection per-layer. Collects `n_samples` snapshots to fit the manifold,
    then uses UMAP's transform() to embed new points.
    
    Warning: UMAP transform is computationally heavier than PCA.
    """
    def __init__(
        self, 
        n_components: int = 2, 
        n_samples: int = 200, 
        n_neighbors: int = 15, 
        min_dist: float = 0.1
    ) -> None:
        self.n_components = n_components
        self.n_samples = n_samples
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        
        self._lock = threading.Lock()
        self._mappers: Dict[str, Any] = {}  # Changed from umap.UMAP to Any
        self._collect: Dict[str, List[np.ndarray]] = {}
        self._steps: Dict[str, int] = {}

        self._last_2d: Dict[str, np.ndarray] = {}

    def _step(self, layer: str, pts2d: np.ndarray) -> Dict[str, Any]:
        if hasattr(pts2d, "get"):  # cupy (cuML) -> numpy
            pts2d = pts2d.get()
        i = self._steps.get(layer, 0)
        self._steps[layer] = i + 1
        value = torch.from_numpy(np.ascontiguousarray(pts2d, dtype=np.float32))
        return {"key": layer, "value": value, "index": i}

    def _create_mapper(self, init="spectral"):
        """Factory to instantiate the correct UMAP backend."""
        # cuML and umap-learn share this exact signature
        return UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            random_state=42,
            init=init
        )

    def process(self, items: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        with self._lock:
            for layer, t in items.items():
                pts = t.numpy()
                mapper = self._mappers.get(layer)
                
                if mapper is not None:
                    # WARM-START UPDATE: 
                    # Create a new UMAP instance anchored to the previous frame
                    mapper = self._create_mapper(init=self._last_2d[layer])
                    
                    # Use fit_transform instead of transform
                    transformed = mapper.fit_transform(pts)
                    
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
                    mapper = self._create_mapper(init="spectral")
                    
                    stacked_buf = np.concatenate(buf, axis=0)
                    mapper.fit(stacked_buf)
                    self._mappers[layer] = mapper
                    
                    transformed_buf = mapper.transform(stacked_buf)
                    
                    # STORE THE INITIAL WARM-START ANCHOR:
                    # Extract the 2D coordinates of the very last snapshot in the buffer
                    last_snap_len = buf[-1].shape[0]
                    self._last_2d[layer] = transformed_buf[-last_snap_len:]
                    
                    offset = 0
                    for snap in buf:
                        snap_len = snap.shape[0]
                        out.append(self._step(layer, transformed_buf[offset:offset+snap_len]))
                        offset += snap_len
                        
                    del self._collect[layer]
        return out

    def ready(self, layer: str) -> bool:
        with self._lock:
            return layer in self._mappers

    def last_index(self, layer: str) -> int:
        with self._lock:
            return max(self._steps.get(layer, 0) - 1, 0)

    def project(self, layer: str, arr: np.ndarray) -> Optional[np.ndarray]:
        with self._lock:
            mapper = self._mappers.get(layer)
            return None if mapper is None else mapper.transform(arr)

    def progress(self) -> Dict[str, Any]:
        with self._lock:
            return {"required": self.n_samples,
                    "layers": {l: len(b) for l, b in self._collect.items()}}

    def collecting(self) -> Dict[str, List[np.ndarray]]:
        with self._lock:
            return {l: list(b) for l, b in self._collect.items()}

class OverallProcessor(Processor):
    """Pass-through, piu' un messaggio extra su `topic` con la media tra le chiavi
    dell'ultimo valore noto di ciascuna (es. spaceloss medio su tutti i layer)."""

    def __init__(self, topic: str, key: str = "overall") -> None:
        self.topic = topic
        self.key = key
        self._latest: Dict[str, torch.Tensor] = {}

    def process(self, items):
        self._latest.update(items)
        msgs = [{"key": k, "value": v} for k, v in items.items()]
        overall = torch.stack(list(self._latest.values())).mean(0)
        msgs.append({"topic": self.topic, "key": self.key, "value": overall})
        return msgs


PROCESSORS: Dict[str, Callable[..., Processor]] = {
    "umap": UmapProcessor, 
    "overall": OverallProcessor
}