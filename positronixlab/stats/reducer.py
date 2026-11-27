import threading
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable

import torch

# ---------------------------------------------------------
# Reducers: stato per-chiave, drain() restituisce solo le chiavi "sporche"
# ---------------------------------------------------------


class Reducer(ABC):
    def __init__(self) -> None:
        self._dirty: set[str] = set()

    @abstractmethod
    def update(self, key: str, value: torch.Tensor) -> None: ...

    @abstractmethod
    def _snapshot(self, key: str) -> torch.Tensor:
        """Valore aggregato per `key` (tensore indipendente dallo stato)."""

    def _after_drain(self, key: str) -> None:  # noqa: B027
        """Hook per reset a blocchi (es. mean)."""

    def drain(self) -> dict[str, torch.Tensor]:
        out = {}
        for key in self._dirty:
            out[key] = self._snapshot(key)
            self._after_drain(key)
        self._dirty.clear()
        return out


class LastReducer(Reducer):
    """Tiene solo l'ultimo valore (decimazione pura)."""
    def __init__(self) -> None:
        super().__init__()
        self._s: dict[str, torch.Tensor] = {}

    def update(self, key, value):
        self._s[key] = value.clone()
        self._dirty.add(key)

    def _snapshot(self, key):
        return self._s[key].clone()


class EmaReducer(Reducer):
    """Media mobile esponenziale: filtra prima di decimare, evita aliasing."""
    def __init__(self, alpha: float = 0.3) -> None:
        super().__init__()
        self.alpha = alpha
        self._s: dict[str, torch.Tensor] = {}

    def update(self, key, value):
        s = self._s.get(key)
        if s is None or s.shape != value.shape:
            self._s[key] = value.clone()
        else:
            s.mul_(1.0 - self.alpha).add_(value.to(s.device), alpha=self.alpha)
        self._dirty.add(key)

    def _snapshot(self, key):
        return self._s[key].clone()


class MeanReducer(Reducer):
    """Media aritmetica dei messaggi arrivati dall'ultimo drain (blocco)."""
    def __init__(self) -> None:
        super().__init__()
        self._sum: dict[str, torch.Tensor] = {}
        self._n: dict[str, int] = {}
        self._lock = threading.Lock()

    def update(self, key, value):
        with self._lock:
            s = self._sum.get(key)
            if s is None or s.shape != value.shape:
                # .clone() qui previene anche il blocco della memoria CUDA IPC
                self._sum[key] = value.clone()
                self._n[key] = 1
            else:
                s.add_(value.to(s.device))
                self._n[key] += 1
            self._dirty.add(key)

    def drain(self):
        """Sincronizza il drain con il lock per prevenire l'eliminazione delle chiavi durante l'update."""
        with self._lock:
            return super().drain()

    def _snapshot(self, key):
        return self._sum[key] / self._n[key]

    def _after_drain(self, key):
        del self._sum[key], self._n[key]


class WindowReducer(Reducer):
    """Rolling window degli ultimi `maxlen` valori; emette l'intera finestra
    (shape [n, ...]). Adatta a metriche scalari/piccole, non a posizioni."""
    def __init__(self, maxlen: int = 200) -> None:
        super().__init__()
        self.maxlen = maxlen
        self._w: dict[str, deque] = {}

    def update(self, key, value):
        if self.maxlen <= 0:
            return
        dq = self._w.setdefault(key, deque(maxlen=self.maxlen))
        if dq and dq[-1].shape != value.shape:
            dq.clear()
        dq.append(value.clone())
        self._dirty.add(key)

    def _snapshot(self, key):
        return torch.stack(list(self._w[key]))

class DisplacementReducer(Reducer):
    """
    Decimazione basata su tolleranza spaziale (Data/Spatial Gating).
    Emette un nuovo tensore solo quando lo spostamento fisico (norma L2) 
    rispetto all'ultimo keyframe emesso supera la soglia specificata.
    """
    def __init__(self, threshold: float = 0.5, method: str = 'max') -> None:
        super().__init__()
        self.threshold = threshold
        if method not in ('max', 'rms'):
            raise ValueError("DisplacementReducer method must be 'max' or 'rms'")
        self.method = method
        
        # _keyframe stores the structural reference we measure against
        self._keyframe: dict[str, torch.Tensor] = {}
        # _pending stores the tensor waiting to be pulled by drain()
        self._pending: dict[str, torch.Tensor] = {}
        self._lock = threading.Lock()

    def update(self, key: str, value: torch.Tensor) -> None:
        with self._lock:
            kf = self._keyframe.get(key)
            
            # Inizializzazione o cambio di shape (es. layer re-init)
            if kf is None or kf.shape != value.shape:
                self._keyframe[key] = value.clone()
                self._pending[key] = value.clone()
                self._dirty.add(key)
                return

            if value.numel() == 0:
                return

            # Calcolo della differenza vettoriale, gestendo eventuali disallineamenti di device
            diff_tensor = value.to(kf.device) - kf
            
            if self.method == 'max':
                # Massima deviazione di un singolo neurone/punto (L2 norm sull'ultima dimensione)
                dists = torch.linalg.norm(diff_tensor, dim=-1)
                diff = dists.max().item()
            else: # 'rms'
                # Spostamento medio globale del sistema (Root Mean Square)
                dists_sq = (diff_tensor ** 2).sum(dim=-1)
                diff = torch.sqrt(dists_sq.mean()).item()

            # Se superiamo la soglia, aggiorniamo il reference frame per i tick successivi
            if diff > self.threshold:
                self._keyframe[key] = value.clone()
                self._pending[key] = value.clone()
                self._dirty.add(key)

    def drain(self) -> dict[str, torch.Tensor]:
        # Sincronizza il drain con il lock per prevenire l'eliminazione 
        # di chiavi pendenti durante un update ad alta frequenza.
        with self._lock:
            return super().drain()

    def _snapshot(self, key: str) -> torch.Tensor:
        return self._pending[key].clone()

    def _after_drain(self, key: str) -> None:
        # Libera la memoria del pending per non mantenere due copie 
        # dello stesso tensore in memoria GPU/CPU tra un threshold hit e l'altro.
        if key in self._pending:
            del self._pending[key]


REDUCERS: dict[str, Callable[..., Reducer]] = {
    "last": LastReducer, "ema": EmaReducer, "mean": MeanReducer, "window": WindowReducer, "displacement": DisplacementReducer
}