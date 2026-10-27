
import torch
from abc import ABC, abstractmethod
from typing import Dict, Callable, List, Any, Optional
import numpy as np
from sklearn.decomposition import PCA
import threading

import logging

logger = logging.getLogger(__name__)


from cuml.manifold.umap import UMAP


try:
    # 1. Try RAPIDS cuML (NVIDIA GPUs)
    
    UMAP_BACKEND = "cuml"

except ImportError:
    try:
        # 2. Fallback to fast parametric UMAP (Apple Silicon / AMD / Intel GPUs)
        from parametric_umap import ParametricUMAP as UMAP
        UMAP_BACKEND = "parametric"
        
    except ImportError:
        # 3. Final fallback to classical CPU UMAP
        from umap import UMAP
        UMAP_BACKEND = "umap-learn"

logger.info(f"Loaded UMAP backend: {UMAP_BACKEND}")


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

    def _step(self, layer: str, pts2d: np.ndarray) -> Dict[str, Any]:
        if hasattr(pts2d, "get"):  # cupy (cuML) -> numpy
            pts2d = pts2d.get()
        i = self._steps.get(layer, 0)
        self._steps[layer] = i + 1
        value = torch.from_numpy(np.ascontiguousarray(pts2d, dtype=np.float32))
        return {"key": layer, "value": value, "index": i}

    def _create_mapper(self):
        """Factory to instantiate the correct UMAP backend."""
        if UMAP_BACKEND == "parametric":
            # parametric_umap exposes a/b instead of the min_dist parameter.
            return UMAP(
                n_components=self.n_components,
                n_neighbors=self.n_neighbors,
                device="cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            )
        else:
            # cuML and umap-learn share this exact signature
            return UMAP(
                n_components=self.n_components,
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                random_state=42
            )

    def process(self, items: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        with self._lock:
            for layer, t in items.items():
                pts = t.numpy()
                mapper = self._mappers.get(layer)
                
                if mapper is not None:
                    # Note: UMAP transform is slower than PCA
                    transformed = mapper.transform(pts)
                    out.append(self._step(layer, transformed))
                    continue
                    
                buf = self._collect.setdefault(layer, [])
                if buf and buf[0].shape[1:] != pts.shape[1:]:
                    buf.clear()  # Layer resized: restart collection
                    
                buf.append(pts.copy())
                
                if len(buf) >= self.n_samples:
                    # Use the conditional factory method
                    mapper = self._create_mapper()
                    
                    stacked_buf = np.concatenate(buf, axis=0)
                    mapper.fit(stacked_buf)
                    self._mappers[layer] = mapper
                    
                    # Project and emit the initial collection
                    transformed_buf = mapper.transform(stacked_buf)
                    
                    # Split the transformed array back into steps to maintain timeline sync
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

class PcaProcessor(Processor):
    """PCA per-layer. Raccoglie `n_samples` snapshot, poi fitta e da quel momento
    emette posizioni gia' proiettate (n, n_components) con un indice progressivo
    (`index`) che e' la timeline di riferimento per i cluster.

    Al fit, ogni snapshot raccolto viene proiettato ed emesso come step separato
    (il vecchio codice li concatenava in un unico step da n_samples * n punti)."""

    def __init__(self, n_components: int = 2, n_samples: int = 100) -> None:
        self.n_components = n_components
        self.n_samples = n_samples
        self._lock = threading.Lock()
        self._pca: Dict[str, PCA] = {}
        self._collect: Dict[str, List[np.ndarray]] = {}
        self._steps: Dict[str, int] = {}

    def _step(self, layer: str, pts2d: np.ndarray) -> Dict[str, Any]:
        if hasattr(pts2d, "get"):  # cupy (cuML) -> numpy
            pts2d = pts2d.get()
        i = self._steps.get(layer, 0)
        self._steps[layer] = i + 1
        value = torch.from_numpy(np.ascontiguousarray(pts2d, dtype=np.float32))
        return {"key": layer, "value": value, "index": i}

    def process(self, items):
        out: List[Dict[str, Any]] = []
        with self._lock:
            for layer, t in items.items():
                pts = t.numpy()
                pca = self._pca.get(layer)
                if pca is not None:
                    out.append(self._step(layer, pca.transform(pts)))
                    continue
                buf = self._collect.setdefault(layer, [])
                if buf and buf[0].shape[1:] != pts.shape[1:]:
                    buf.clear()  # layer ridimensionato: ricomincia la raccolta
                buf.append(pts.copy())
                if len(buf) >= self.n_samples:
                    pca = PCA(n_components=self.n_components)
                    pca.fit(np.concatenate(buf, axis=0))
                    self._pca[layer] = pca
                    for snap in buf:
                        out.append(self._step(layer, pca.transform(snap)))
                    del self._collect[layer]
        return out

    def ready(self, layer: str) -> bool:
        with self._lock:
            return layer in self._pca

    def last_index(self, layer: str) -> int:
        with self._lock:
            return max(self._steps.get(layer, 0) - 1, 0)

    def project(self, layer: str, arr: np.ndarray) -> Optional[np.ndarray]:
        with self._lock:
            pca = self._pca.get(layer)
            return None if pca is None else pca.transform(arr)

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
    "pca": PcaProcessor, 
    "umap": UmapProcessor, 
    "overall": OverallProcessor
}