"""StatsPlugin: sta tra i produttori raw (training) e i consumatori (UI, altri plugin).

Responsabilita' (tutte fuori dal processo della UI):
  * ingestione degli stream raw ad alta frequenza: ogni messaggio viene solo
    "ripiegato" in un accumulatore (O(1), lock per-stream tenuto pochi us);
  * riduzione a cadenza configurabile (ema / mean / window / last);
  * elaborazioni a valle del reducer (PCA/UMAP delle posizioni, media su layer);
  * sincronizzazione dei cluster con la timeline delle posizioni;
  * storico completo e persistenza (save_state);
  * pubblicazione dei dati aggregati su topic (blocchi ordinati) e/o env var (ultimo stato).

Tutto cio' che e' pesante (PCA/UMAP, conversione CPU, publish) gira nel loop di main(),
mai nel thread di ricezione.

Configurazione da `options["streams"][<nome>]`, esempio:

    "streams": {
        "positions": {"interval_s": 1.0, "processor_args": {"n_samples": 50}},
        "training":  {"reducer": "window", "reducer_args": {"maxlen": 500},
                      "out_topic": None, "out_env": "stats_training"},
        "spaceloss": None,                       # stream disattivato
    },
    "save_path": "/path/state.pkl",
"""

import copy
import logging
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import torch

from parallelazy import Plugin, PluginDescriptor, PluginWorker
from modularnet.stats.common import _to_np
from modularnet.stats.processor import Processor, PcaProcessor, UmapProcessor
from modularnet.stats.stream import Stream
from modularnet.utils.utils import save_data

logger = logging.getLogger(__name__)

SPACELOSS_NAMES = ["loss", "loss_on", "loss_off", "dict_center"]
PROGRESS_ENV = "stats_pca_progress"
_DEDICATED_ENV = {"space_stats", "observed_layers", "model_graph", "trial_config"}


# ---------------------------------------------------------
# Extractors: messaggio raw -> [(chiave, valore)]
# ---------------------------------------------------------
def _x_positions(msg: Dict[str, Any]) -> List[Tuple[str, Any]]:
    lid, pos = msg.get("layer_id"), msg.get("positions")
    return [] if lid is None or pos is None else [(lid, pos)]


def _x_spaceloss(msg: Dict[str, Any]) -> List[Tuple[str, Any]]:
    if msg.get("name") != "spaceloss" or msg.get("layer_id") is None:
        return []
    return [(msg["layer_id"], msg["metric"])]  # [loss, loss_on, loss_off, dict_center]


def _x_training(msg: Dict[str, Any]) -> List[Tuple[str, Any]]:
    if msg.get("name") != "training":
        return []
    return list((msg.get("metric") or {}).items())


EXTRACTORS: Dict[str, Callable[[Dict[str, Any]], Iterable[Tuple[str, Any]]]] = {
    "positions": _x_positions, "spaceloss": _x_spaceloss, "training": _x_training,
}


DEFAULT_STREAMS: Dict[str, Dict[str, Any]] = {
    "positions": {
        "topic": "positions", "reducer": "ema", "reducer_args": {"alpha": 0.3},
        "interval_s": 0.25,
        "processor": "umap", "processor_args": {"n_components": 2, "n_samples": 200, "n_neighbors": 15, "min_dist": 0.1},
        "out_topic": "stats_positions", "out_env": None, "keep_history": True,
    },
    "spaceloss": {
        "topic": "metrics", "reducer": "mean", "interval_s": 0.5,
        "names": SPACELOSS_NAMES,
        "processor": "overall", "processor_args": {"topic": "stats_spaceloss_overall"},
        "out_topic": "stats_spaceloss", "out_env": None, "keep_history": True,
    },
    "training": {
        "topic": "metrics", "reducer": "mean", "interval_s": 0.5,
        "out_topic": "stats_training", "out_env": None, "keep_history": True,
    },
}


class StatsPlugin(Plugin):

    @classmethod
    def default_options(cls) -> Dict[str, Any]:
        return {
            "tick_s": 0.02,
            "save_path": None,
            "streams": copy.deepcopy(DEFAULT_STREAMS),
        }

    @classmethod
    def Descriptor(cls, options: Dict[str, Any] = None) -> PluginDescriptor:
        return super().Descriptor(
            version="2.0.0",
            author="ModularNet Team",
            description="Aggregates raw streams (reduce, UMAP/PCA, cluster sync, history) and republishes them at a configurable rate.",
            options=options,
        )

    def __init__(self, worker: PluginWorker, plugin_id: str, options: Dict[str, Any]):
        super().__init__(worker, plugin_id, options)
        self.tick_s: float = options.get("tick_s", 0.02)
        self.save_path: Optional[str] = options.get("save_path")
        self.streams: Dict[str, Stream] = {
            name: Stream(name, c, EXTRACTORS)
            for name, c in Stream.merge_streams(options.get("streams"), DEFAULT_STREAMS).items()
        }
        
        self._projector: Optional[Processor] = next(
            (s.processor for s in self.streams.values() if isinstance(s.processor, (PcaProcessor, UmapProcessor))), None
        )

        self._history: Dict[str, Dict[str, List[Any]]] = {}
        self._history_lock = threading.Lock()

        self._cluster_lock = threading.Lock()
        self._pending_clusters: Dict[str, List[Dict[str, Any]]] = {}
        self._cluster_history: Dict[str, List[Dict[str, Any]]] = {}

        self._last_progress: Optional[Dict[str, Any]] = None

    # -----------------------------------------------------
    # lifecycle
    # -----------------------------------------------------
    def on_start(self) -> None:
        by_topic: Dict[str, List[Stream]] = {}
        for s in self.streams.values():
            by_topic.setdefault(s.topic, []).append(s)
        for topic, streams in by_topic.items():
            self.topic_subscribe(topic, self._make_dispatcher(streams))

        if self._projector is not None:
            self.topic_subscribe("clusters", self._on_clusters)
            self._publish_progress(force=True)

    def main(self) -> None:
        now = time.monotonic()
        for s in self.streams.values():
            if s.due(now):
                self._safe_emit(s)
        self._resolve_clusters()
        if self._projector is not None:
            self._publish_progress()
        time.sleep(self.tick_s)

    def on_stop(self) -> None:
        for s in self.streams.values():
            self._safe_emit(s)
        self._resolve_clusters()
        self.save_state()

    # -----------------------------------------------------
    # ingestione
    # -----------------------------------------------------
    def _make_dispatcher(self, streams: List[Stream]) -> Callable[[Dict[str, Any]], None]:
        def _dispatch(msg: Dict[str, Any]) -> None:
            for s in streams:
                s.ingest(msg)
        return _dispatch

    def _on_clusters(self, msg: Dict[str, Any]) -> None:
        layer = msg.get("layer_id")
        if layer is None:
            return
        item = {
            "labels": _to_np(msg.get("labels")),
            "centroids": _to_np(msg.get("centroids")),
            "best_k": msg.get("best_k"),
            "best_score": msg.get("best_score"),
        }
        with self._cluster_lock:
            self._pending_clusters.setdefault(layer, []).append(item)

    # -----------------------------------------------------
    # emissione
    # -----------------------------------------------------
    def _safe_emit(self, s: Stream) -> None:
        try:
            self._emit(s)
        except Exception:
            logger.exception(f"emit failed for stream '{s.name}'")

    def _emit(self, s: Stream) -> None:
        blocks, n = s.drain()
        if not blocks:
            return
        items = {k: t.detach().cpu() for k, t in blocks.items()}
        if s.processor is not None:
            msgs = s.processor.process(items)
        else:
            msgs = [{"key": k, "value": v} for k, v in items.items()]
        if not msgs:
            return

        s.seq += 1
        now = time.time()
        env_dirty = False
        for m in msgs:
            topic = m.pop("topic", None)
            m.update(stream=s.name, seq=s.seq, count=n, t=now)
            if s.names:
                m["names"] = s.names
            if topic:
                self.topic_publish(topic, m)
                continue
            if s.out_topic:
                self.topic_publish(s.out_topic, m)
            if s.keep_history:
                self._append_history(s.name, m["key"], m["value"])
            if s.out_env:
                s.latest[m["key"]] = m["value"].clone()
                env_dirty = True
        if env_dirty:
            self.env_set(s.out_env, dict(s.latest))

    def _publish_progress(self, force: bool = False) -> None:
        prog = self._projector.progress()
        if force or prog != self._last_progress:
            self._last_progress = prog
            self.env_set(PROGRESS_ENV, prog)

    def _resolve_clusters(self) -> None:
        if self._projector is None:
            return
        with self._cluster_lock:
            ready_layers = [l for l in self._pending_clusters if self._projector.ready(l)]
            ready = {l: self._pending_clusters.pop(l) for l in ready_layers}
        for layer, items in ready.items():
            index = self._projector.last_index(layer)
            for it in items:
                centroids_2d = self._projector.project(layer, it["centroids"]).astype(np.float32)
                snap = {
                    "position_index": index,
                    "labels": it["labels"],
                    "best_k": it["best_k"],
                    "best_score": it["best_score"],
                    "centroids": centroids_2d,
                }
                self._cluster_history.setdefault(layer, []).append(snap)
                
                self.topic_publish("stats_clusters", {
                    "layer_id": layer,
                    "position_index": index,
                    "labels": torch.as_tensor(it["labels"]),
                    "best_k": it["best_k"],
                    "best_score": it["best_score"],
                    "centroids": torch.as_tensor(centroids_2d),
                })

    # -----------------------------------------------------
    # storico e persistenza
    # -----------------------------------------------------
    def _append_history(self, stream: str, key: str, value: torch.Tensor) -> None:
        arr = value.numpy().copy()
        item = arr.item() if arr.ndim == 0 else arr
        with self._history_lock:
            self._history.setdefault(stream, {}).setdefault(key, []).append(item)

    def save_state(self) -> bool:
        if self.save_path is None:
            return False

        with self._history_lock:
            hist = {s: {k: list(v) for k, v in d.items()} for s, d in self._history.items()}

        names = (self.streams["spaceloss"].names if "spaceloss" in self.streams else None) or SPACELOSS_NAMES
        layer_spaceloss: Dict[str, Dict[str, List[float]]] = {}
        for layer, vecs in hist.get("spaceloss", {}).items():
            arr = np.stack(vecs)
            layer_spaceloss[layer] = {n: arr[:, i].astype(float).tolist() for i, n in enumerate(names)}
        training_metrics = {k: [float(x) for x in v] for k, v in hist.get("training", {}).items()}

        env = dict(self.env_get() or {})
        trial_config = env.get("trial_config")
        if hasattr(trial_config, "to_dict"):
            trial_config = trial_config.to_dict()
        dynamic_vars = {
            k: v for k, v in env.items()
            if k not in _DEDICATED_ENV and not k.startswith("stats_")
        }
        if trial_config:
            dynamic_vars["trial_config"] = trial_config

        data = {
            "dynamic_vars": dynamic_vars,
            "model_graph": env.get("model_graph") or {"nodes": {}, "edges": []},
            "stats": env.get("space_stats") or {},
            "observed_layers": env.get("observed_layers") or {},
            "training_metrics": training_metrics,
            "layer_spaceloss": layer_spaceloss,
            "positions_plot": hist.get("positions", {}),
            "pca_history": self._projector.collecting() if self._projector is not None else {}, 
            "cluster_history": {l: list(v) for l, v in self._cluster_history.items()},
        }
        save_data(self.save_path, {"header": {"timestamp": time.time()}, "data": data})
        logger.info(f"StatsPlugin state saved to {self.save_path}")
        return True