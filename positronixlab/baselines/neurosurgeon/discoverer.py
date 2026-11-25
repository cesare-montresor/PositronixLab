"""NeuroSurgeon (Lepori et al., 2023/2024) subnetwork discoverer."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from positronixlab.baselines.base import get_maskable_layer_names
from positronixlab.baselines.neurosurgeon.adapter import wrap_with_neurosurgeon
from positronixlab.tasks.discoverer import SubnetworkDiscoverer, unpack_batch
from positronixlab.tasks.masks import SubnetworkMask


class NeuroSurgeonDiscoverer(SubnetworkDiscoverer):
    """Lepori et al. subnetwork discovery backed by ``NeuroSurgeon.CircuitModel``.

    Supports:
    - ``mask_method="continuous_sparsification"`` (with exponential temperature annealing)
    - ``mask_method="hard_concrete"``
    - ``mask_method="magnitude_pruning"`` (zero training epochs required)
    and both ``mask_unit="weight"`` and ``mask_unit="neuron"``.
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
        mask_hparams: Optional[Dict[str, Any]] = None,
        loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.mask_method = mask_method
        self.mask_unit = mask_unit
        self.l0_lambda = l0_lambda
        self.lr = lr
        self.epochs = epochs
        self.max_temp = max_temp
        self.mask_hparams = dict(mask_hparams) if mask_hparams else None
        self.loss_fn = loss_fn or F.cross_entropy

    def discover(
        self,
        model: nn.Module,
        dataloader: Iterable[Any],
        target_layers: Optional[List[str]] = None,
        mask_name: str = "neurosurgeon",
        **kwargs: Any,
    ) -> SubnetworkMask:
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
            return wrapped.extract_subnetwork_mask(name=mask_name)

        device = next(wrapped.parameters()).device
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

        return wrapped.extract_subnetwork_mask(name=mask_name)
