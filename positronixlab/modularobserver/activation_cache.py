"""Content-addressed disk cache for frozen-model layer activations and scores."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import torch
from torch import nn

from positronixlab.baselines.base import get_maskable_layer_names
from positronixlab.spacemath.scoring import activations_to_scores
from positronixlab.tasks.discoverer import unpack_batch

logger = logging.getLogger(__name__)

CACHE_FORMAT_VERSION = 3


def reduce_activation_2d(act: torch.Tensor, module: nn.Module | None = None) -> torch.Tensor:
    """Reduce a layer activation tensor to 2D shape ``(B, U)``."""
    if act.ndim == 2:
        return act
    if isinstance(module, nn.modules.conv._ConvNd):
        return act.mean(dim=tuple(range(2, act.ndim)))
    if act.ndim == 4:
        return act.mean(dim=(2, 3))
    if act.ndim == 3:
        return act.mean(dim=1)
    return act.reshape(act.shape[0], -1)


def infer_layer_info(name: str, module: nn.Module) -> dict[str, Any]:
    """Build an ``observed_layers`` metadata entry compatible with :class:`LayerSpace`."""
    if isinstance(module, nn.Linear):
        out_units = int(module.out_features)
        layer_type = "linear"
    elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        out_units = int(module.out_channels)
        layer_type = "conv"
    elif type(module).__name__ == "Conv1D" and getattr(module, "nf", None) is not None:
        out_units = int(getattr(module, "nf"))
        layer_type = "linear"
    else:
        w = getattr(module, "weight", None)
        out_units = int(w.shape[0]) if isinstance(w, torch.Tensor) and w.ndim >= 1 else 0
        layer_type = "linear"
    return {
        "name": name,
        "out_units": out_units,
        "layer_type": layer_type,
        "activation_type": None,
    }


class ActivationCache:
    """Disk-backed cache for frozen-model layer activations or ``[-1, 1]`` unit scores.

    Supports two modes:
    - ``"activations"``: stores raw detached 2D layer activations once per frozen
      model so a single cache serves multiple scoring/HPO configurations.
    - ``"scores"``: stores compact ``[-1, 1]`` unit scores directly.

    Files are written atomically via PID-scoped temp files and ``os.replace``.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        cache_mode: Literal["activations", "scores"] = "activations",
        cache_dtype: Literal["float16", "float32"] = "float32",
        scoring: str = "quantile",
        active_q: float = 0.80,
        reducer: str | None = None,
        apply_activation: bool = False,
        activation_fn: nn.Module | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_mode = cache_mode
        self.cache_dtype = cache_dtype
        self.scoring = scoring
        self.active_q = active_q
        self.reducer = reducer
        self.apply_activation = apply_activation
        self.activation_fn = activation_fn

    @classmethod
    def compute_cache_path(
        cls,
        cache_root: str | Path,
        *,
        signature_fields: dict[str, Any],
        cache_mode: Literal["activations", "scores"] = "activations",
        cache_dtype: Literal["float16", "float32"] = "float32",
        prefix: str = "cache",
        scoring: str = "quantile",
        active_q: float = 0.80,
        reducer: str | None = None,
        apply_activation: bool = False,
    ) -> Path:
        """Compute a deterministic content-addressed directory path under ``cache_root``."""
        payload = dict(signature_fields)
        payload["cache_mode"] = cache_mode
        payload["cache_dtype"] = cache_dtype
        if cache_mode == "scores":
            payload.update(
                {
                    "scoring": scoring,
                    "active_q": active_q,
                    "reducer": reducer,
                    "apply_activation": apply_activation,
                }
            )
        digest = hashlib.sha1(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8"),
            usedforsecurity=False,
        ).hexdigest()[:12]
        safe_prefix = quote(prefix, safe="._-")
        return Path(cache_root).resolve() / f"{safe_prefix}__{cache_mode}__{digest}"

    def try_load(self) -> tuple[dict[str, Any], dict[str, torch.Tensor]] | None:
        """Load cached ``(metadata, tensors)`` from disk if valid, else return ``None``."""
        meta_path = self.cache_dir / "metadata.pt"
        tensors_path = self.cache_dir / "tensors.pt"
        if not (meta_path.is_file() and tensors_path.is_file()):
            return None
        try:
            meta = torch.load(meta_path, map_location="cpu", weights_only=True)
            if (
                not isinstance(meta, dict)
                or meta.get("format_version") != CACHE_FORMAT_VERSION
                or meta.get("cache_mode") != self.cache_mode
                or not meta.get("observed_layers")
            ):
                return None
            tensors = torch.load(tensors_path, map_location="cpu", weights_only=True)
            if not isinstance(tensors, dict) or not tensors:
                return None
            return meta, tensors
        except Exception as exc:
            logger.warning("Ignoring unreadable activation cache at %s: %s", self.cache_dir, exc)
            return None

    def build(
        self,
        model: nn.Module,
        dataloader: Iterable[Any],
        *,
        device: torch.device | str | None = None,
        target_layers: Sequence[str] | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        """Run a single forward pass over ``dataloader``, persist cache atomically, and return it."""
        t0 = time.monotonic()
        was_training = model.training
        model.eval()
        first_param = next(model.parameters(), None)
        dev = (
            torch.device(device)
            if device is not None
            else (first_param.device if first_param is not None else torch.device("cpu"))
        )

        layers = list(target_layers) if target_layers is not None else get_maskable_layer_names(model)
        modules = dict(model.named_modules())
        observed_layers = {
            name: infer_layer_info(name, modules[name])
            for name in layers
            if name in modules
        }
        if not observed_layers:
            raise RuntimeError("No maskable layers found in model for ActivationCache")

        store_dtype = torch.float16 if self.cache_dtype == "float16" else torch.float32
        chunks: dict[str, list[torch.Tensor]] = {name: [] for name in observed_layers}
        handles = []

        for layer_name in observed_layers:
            mod = modules[layer_name]
            layer_type = observed_layers[layer_name]["layer_type"]

            def _make_hook(lname: str, ltype: str):
                def _hook(mod_ref: nn.Module, _inp: Any, output: Any) -> None:
                    out_t = output[0] if isinstance(output, (tuple, list)) else output
                    if not isinstance(out_t, torch.Tensor):
                        return
                    raw = out_t.detach()
                    if self.cache_mode == "scores":
                        tensor = activations_to_scores(
                            raw,
                            layer_type=ltype,
                            score_method=self.scoring,
                            reducer=self.reducer,
                            activation_fn=self.activation_fn,
                            apply_activation=self.apply_activation,
                            active_q=self.active_q,
                        )
                    else:
                        tensor = reduce_activation_2d(raw, mod_ref)
                    chunks[lname].append(tensor.to(device="cpu", dtype=store_dtype))

                return _hook

            handles.append(mod.register_forward_hook(_make_hook(layer_name, layer_type)))

        try:
            with torch.no_grad():
                for batch in dataloader:
                    x, _ = unpack_batch(batch)
                    model(x.to(dev))
        finally:
            for h in handles:
                h.remove()
            model.train(was_training)

        tensors: dict[str, torch.Tensor] = {
            name: torch.cat(parts, dim=0).contiguous()
            for name, parts in chunks.items()
            if parts
        }
        if not tensors:
            raise RuntimeError("ActivationCache collected zero batches from dataloader")

        datapoint_count = int(next(iter(tensors.values())).shape[0])
        meta: dict[str, Any] = {
            "format_version": CACHE_FORMAT_VERSION,
            "cache_mode": self.cache_mode,
            "cache_dtype": self.cache_dtype,
            "datapoint_count": datapoint_count,
            "observed_layers": observed_layers,
            "build_duration_sec": time.monotonic() - t0,
        }
        if extra_metadata:
            meta.update(extra_metadata)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp_tensors = self.cache_dir / f"tensors.{os.getpid()}.tmp"
        tmp_meta = self.cache_dir / f"metadata.{os.getpid()}.tmp"
        torch.save(tensors, tmp_tensors)
        os.replace(tmp_tensors, self.cache_dir / "tensors.pt")
        torch.save(meta, tmp_meta)
        os.replace(tmp_meta, self.cache_dir / "metadata.pt")
        return meta, tensors

    def load_or_build(
        self,
        model_and_loader_fn: Any,
        *,
        force_rebuild: bool = False,
        device: torch.device | str | None = None,
        target_layers: Sequence[str] | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor], bool]:
        """Return ``(meta, tensors, cache_hit)``; only invokes ``model_and_loader_fn()`` on a miss."""
        if not force_rebuild:
            cached = self.try_load()
            if cached is not None:
                return cached[0], cached[1], True
        model, dataloader = model_and_loader_fn()
        meta, tensors = self.build(
            model,
            dataloader,
            device=device,
            target_layers=target_layers,
            extra_metadata=extra_metadata,
        )
        return meta, tensors, False
