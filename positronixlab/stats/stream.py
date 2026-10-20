
import copy
import time

import torch
from typing import Callable, Dict, Iterable, List, Any, Optional, Tuple

import threading

import logging

from positronixlab.stats.common import _as_tensor
from positronixlab.stats.processor import PROCESSORS, Processor
from positronixlab.stats.reducer import REDUCERS, Reducer

logger = logging.getLogger(__name__)

Extractor = Callable[[Dict[str, Any]], Iterable[Tuple[str, Any]]]

# ---------------------------------------------------------
# Stream
# ---------------------------------------------------------
class Stream:
    def __init__(self, name: str, cfg: Dict[str, Any], extractors: Dict[str, Extractor]):
        self.name = name
        self.topic: str = cfg["topic"]
        ext = cfg.get("extract", name)
        self.extract: Extractor = extractors[ext] if isinstance(ext, str) else ext
        self.reducer: Reducer = REDUCERS[cfg["reducer"]](**cfg.get("reducer_args", {}))
        pname = cfg.get("processor")
        self.processor: Optional[Processor] = (
            PROCESSORS[pname](**cfg.get("processor_args", {})) if pname else None
        )
        self.names: Optional[List[str]] = cfg.get("names")
        self.interval_s: Optional[float] = cfg.get("interval_s")
        self.every_n: Optional[int] = cfg.get("every_n")
        self.out_topic: Optional[str] = cfg.get("out_topic")
        self.out_env: Optional[str] = cfg.get("out_env")
        self.keep_history: bool = cfg.get("keep_history", True)

        self._lock = threading.Lock()
        self._count = 0
        self._last_emit = time.monotonic()
        self.seq = 0
        self.latest: Dict[str, torch.Tensor] = {}  # per out_env: stato completo

    def ingest(self, msg: Dict[str, Any]) -> None:
        """Hot path (thread di ricezione): solo fold nell'accumulatore."""
        items = self.extract(msg)
        if not items:
            return
        with self._lock:
            for key, value in items:
                self.reducer.update(key, _as_tensor(value))
            self._count += 1

    def due(self, now: float) -> bool:
        with self._lock:
            if self._count == 0:
                return False
            if self.every_n and self._count >= self.every_n:
                return True
            return bool(self.interval_s) and (now - self._last_emit) >= self.interval_s

    def drain(self) -> Tuple[Dict[str, torch.Tensor], int]:
        with self._lock:
            blocks = self.reducer.drain()
            n, self._count = self._count, 0
            self._last_emit = time.monotonic()
        return blocks, n



    @classmethod
    def merge_streams(cls, user: Optional[Dict[str, Any]], streams: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        cfg = copy.deepcopy(streams)
        for name, ov in (user or {}).items():
            if ov is None or ov.get("enabled") is False:
                cfg.pop(name, None)
                continue
            ov = {k: v for k, v in ov.items() if k != "enabled"}
            base = cfg.setdefault(name, {})
            for kind in ("reducer", "processor"):
                args = f"{kind}_args"
                changed = kind in ov and ov[kind] != base.get(kind)
                if changed and args not in ov:
                    base.pop(args, None)  # nuovo tipo: i vecchi args non valgono piu'
                elif not changed and args in ov:
                    ov[args] = {**base.get(args, {}), **ov[args]}  # stesso tipo: merge
            base.update(ov)
        return cfg
