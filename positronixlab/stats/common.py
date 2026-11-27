# ---------------------------------------------------------
# Conversione
# ---------------------------------------------------------
from typing import Any

import numpy as np
import torch


def _as_tensor(value: Any) -> torch.Tensor:
    """Detached float32. NON garantisce una copia: chi conserva il valore
    deve fare .clone() (mai tenere riferimenti al buffer del messaggio ricevuto,
    puo' essere un buffer IPC/cudagraph riciclato)."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(torch.float32)
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], torch.Tensor):
        tensors = [_as_tensor(v) for v in value]
        if all(t.numel() == 1 for t in tensors):
            return torch.stack([t.reshape(()) for t in tensors])
        return torch.stack(tensors)
    return torch.as_tensor(np.asarray(value, dtype=np.float32))


def _to_np(value: Any) -> np.ndarray:
    """Copia numpy indipendente dal buffer del messaggio."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().copy()
    return np.array(value)

