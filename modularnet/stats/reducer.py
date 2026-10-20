import torch
from abc import ABC, abstractmethod
from typing import Dict, Callable, Set
from collections import deque


# ---------------------------------------------------------
# Reducers: stato per-chiave, drain() restituisce solo le chiavi "sporche"
# ---------------------------------------------------------


class Reducer(ABC):
    def __init__(self) -> None:
        self._dirty: Set[str] = set()

    @abstractmethod
    def update(self, key: str, value: torch.Tensor) -> None: ...

    @abstractmethod
    def _snapshot(self, key: str) -> torch.Tensor:
        """Valore aggregato per `key` (tensore indipendente dallo stato)."""

    def _after_drain(self, key: str) -> None:
        """Hook per reset a blocchi (es. mean)."""

    def drain(self) -> Dict[str, torch.Tensor]:
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
        self._s: Dict[str, torch.Tensor] = {}

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
        self._s: Dict[str, torch.Tensor] = {}

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
        self._sum: Dict[str, torch.Tensor] = {}
        self._n: Dict[str, int] = {}

    def update(self, key, value):
        s = self._sum.get(key)
        if s is None or s.shape != value.shape:
            self._sum[key], self._n[key] = value.clone(), 1
        else:
            s.add_(value.to(s.device))
            self._n[key] += 1
        self._dirty.add(key)

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
        self._w: Dict[str, deque] = {}

    def update(self, key, value):
        self._w.setdefault(key, deque(maxlen=self.maxlen)).append(value.clone())
        self._dirty.add(key)

    def _snapshot(self, key):
        return torch.stack(list(self._w[key]))


REDUCERS: Dict[str, Callable[..., Reducer]] = {
    "last": LastReducer, "ema": EmaReducer, "mean": MeanReducer, "window": WindowReducer,
}