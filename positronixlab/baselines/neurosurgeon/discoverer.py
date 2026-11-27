"""NeuroSurgeon (Lepori et al., 2023/2024) subnetwork discoverer."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from positronixlab.baselines.base import get_maskable_layer_names
from positronixlab.baselines.lepori.discoverer import ContinuousSparsificationDiscoverer
from positronixlab.baselines.neurosurgeon.adapter import (
    is_neurosurgeon_available,
    wrap_with_neurosurgeon,
)
from positronixlab.tasks.discoverer import SubnetworkDiscoverer, unpack_batch
from positronixlab.tasks.masks import SubnetworkMask


class NeuroSurgeonDiscoverer(SubnetworkDiscoverer):
    """Lepori et al. subnetwork discovery backed by ``NeuroSurgeon.CircuitModel``
    or native PyTorch 2.x :class:`ContinuousSparsificationDiscoverer`.

    Supports:
    - ``mask_method="continuous_sparsification"`` (with exponential temperature annealing)
    - ``mask_method="hard_concrete"``
    - ``mask_method="magnitude_pruning"`` (zero training epochs required)
    and both ``mask_unit="weight"`` and ``mask_unit="neuron"``.

    Set ``backend="native"`` or ``backend="auto"`` to use pure-PyTorch 2.x
    Continuous Sparsification without requiring ``NeuroSurgeon``.
    """

    name = "neurosurgeon"

    def __init__(
        self,
        mask_method: str = "continuous_sparsification",
        mask_unit: str = "weight",
        l0_lambda: float = 1e-4,
        lr: float = 1e-2,
        epochs: int = 10,
        max_temp: float = 200.0,
        mask_hparams: dict[str, Any] | None = None,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        *,
        backend: Literal["neurosurgeon", "native", "auto"] = "neurosurgeon",
        init_value: float = 0.0,
        use_amp: bool = False,
        use_compile: bool = False,
    ) -> None:
        if backend not in ("neurosurgeon", "native", "auto"):
            raise ValueError(
                f"backend must be 'neurosurgeon', 'native', or 'auto', got {backend!r}"
            )
        if backend == "native" and mask_method != "continuous_sparsification":
            raise ValueError(
                f"backend='native' only supports mask_method='continuous_sparsification', "
                f"got {mask_method!r}"
            )
        self.mask_method = mask_method
        self.mask_unit = mask_unit
        self.l0_lambda = l0_lambda
        self.lr = lr
        self.epochs = epochs
        self.max_temp = max_temp
        self.mask_hparams = dict(mask_hparams) if mask_hparams else None
        self.loss_fn = loss_fn or F.cross_entropy
        self.backend = backend
        self.init_value = init_value
        self.use_amp = use_amp
        self.use_compile = use_compile
        self.last_stats: dict[str, Any] = {}

    def discover(
        self,
        model: nn.Module,
        dataloader: Iterable[Any],
        target_layers: list[str] | None = None,
        mask_name: str = "neurosurgeon",
        **kwargs: Any,
    ) -> SubnetworkMask:
        t0 = time.monotonic()
        use_native = self.mask_method == "continuous_sparsification" and self.backend in (
            "native",
            "auto",
        )
        if use_native:
            native = ContinuousSparsificationDiscoverer(
                mask_unit=self.mask_unit,  # type: ignore[arg-type]
                l0_lambda=self.l0_lambda,
                lr=self.lr,
                epochs=self.epochs,
                init_value=self.init_value,
                max_temp=self.max_temp,
                loss_fn=self.loss_fn,
                use_amp=self.use_amp,
                use_compile=self.use_compile,
            )
            mask = native.discover(
                model,
                dataloader,
                target_layers=target_layers,
                mask_name=mask_name,
                **kwargs,
            )
            self.last_stats = dict(native.last_stats)
            return mask

        if not is_neurosurgeon_available():
            raise ImportError(
                f"NeuroSurgeon is required for mask_method={self.mask_method!r} "
                f"with backend={self.backend!r}. Use backend='native' with "
                "mask_method='continuous_sparsification' for a dependency-free run."
            )

        layers = (
            list(target_layers)
            if target_layers is not None
            else get_maskable_layer_names(model)
        )
        wrapped = wrap_with_neurosurgeon(
            model=model,
            target_layers=layers,
            mask_method=self.mask_method,
            mask_unit=self.mask_unit,
            mask_hparams=self.mask_hparams,
            freeze_base=True,
            l0_lambda=self.l0_lambda,
            deepcopy_model=True,
        )

        if self.mask_method == "magnitude_pruning" or self.epochs <= 0:
            self.last_stats = {
                "epochs_run": 0,
                "duration_sec": time.monotonic() - t0,
                "backend": "neurosurgeon",
            }
            return wrapped.extract_subnetwork_mask(name=mask_name)

        first_param = next(wrapped.parameters(), None)
        device = first_param.device if first_param is not None else torch.device("cpu")
        mask_params = wrapped.mask_parameters()
        optimizer = torch.optim.Adam(mask_params, lr=self.lr)
        temp = 1.0
        temp_mult = (
            self.max_temp ** (1.0 / max(1, self.epochs))
            if self.mask_method == "continuous_sparsification"
            else 1.0
        )

        wrapped.train(True)
        for _ in range(self.epochs):
            if self.mask_method == "continuous_sparsification":
                wrapped.temperature = temp
            for batch in dataloader:
                x, y = unpack_batch(batch)
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = wrapped(x)
                task_loss = self.loss_fn(logits, y)
                l0 = wrapped.l0_loss()
                total_loss = task_loss + self.l0_lambda * l0
                total_loss.backward()
                optimizer.step()
            if self.mask_method == "continuous_sparsification":
                temp *= temp_mult

        self.last_stats = {
            "epochs_run": self.epochs,
            "duration_sec": time.monotonic() - t0,
            "backend": "neurosurgeon",
        }
        return wrapped.extract_subnetwork_mask(name=mask_name)
