"""Csordas et al. (ICLR 2021) Gumbel-Sigmoid STE subnetwork discoverer."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from positronixlab.baselines.base import get_maskable_layer_names
from positronixlab.tasks.discoverer import SubnetworkDiscoverer, unpack_batch
from positronixlab.tasks.masks import SubnetworkMask


def _gumbel_sigmoid_ste(
    logits: torch.Tensor,
    tau: float = 1.0,
    hard: bool = True,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Exact Csordas et al. Gumbel-Sigmoid Straight-Through Estimator."""
    u0 = torch.rand_like(logits)
    u1 = torch.rand_like(logits)
    noise = -torch.log(torch.log(u1 + eps) / torch.log(u0 + eps) + eps)
    soft = torch.sigmoid((logits + noise) / tau)
    if not hard:
        return soft
    hard_mask = (soft > 0.5).to(soft.dtype)
    return (hard_mask - soft).detach() + soft


class _GumbelMaskParametrization(nn.Module):
    """PyTorch 2.x parametrization applying Csordas Gumbel-Sigmoid STE masks."""

    def __init__(
        self,
        shape: tuple[int, ...],
        init_logit: float = 2.0,
        tau: float = 1.0,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.logits = nn.Parameter(
            torch.full(shape, float(init_logit), device=device)
        )
        self.tau = tau

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if self.training:
            mask = _gumbel_sigmoid_ste(self.logits, tau=self.tau, hard=True)
        else:
            mask = (self.logits >= 0.0).to(weight.dtype)
        return weight * mask


class CsordasGumbelDiscoverer(SubnetworkDiscoverer):
    """Modern PyTorch 2.x implementation of Csordas et al. (ICLR 2021) weight masking.

    Uses ``torch.nn.utils.parametrize`` on a deep copy of *model* (zero mutation
    of caller's model, zero ``_parameters`` hacking), Gumbel-Sigmoid STE, and
    Csordas's exact linear logit sparsity penalty ``(mask_loss_weight / B) * sum(logits)``.
    """

    name = "csordas_gumbel"

    def __init__(
        self,
        mask_loss_weight: float = 1e-4,
        lr: float = 1e-2,
        epochs: int = 10,
        init_logit: float = 2.0,
        tau: float = 1.0,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.mask_loss_weight = mask_loss_weight
        self.lr = lr
        self.epochs = epochs
        self.init_logit = init_logit
        self.tau = tau
        self.loss_fn = loss_fn or F.cross_entropy

    def discover(
        self,
        model: nn.Module,
        dataloader: Iterable[Any],
        target_layers: list[str] | None = None,
        mask_name: str = "csordas_gumbel",
        **kwargs: Any,
    ) -> SubnetworkMask:
        import torch.nn.utils.parametrize as parametrize

        work_model = copy.deepcopy(model)
        work_model.eval()
        for p in work_model.parameters():
            p.requires_grad = False

        layers = (
            list(target_layers)
            if target_layers is not None
            else get_maskable_layer_names(work_model)
        )
        modules = dict(work_model.named_modules())
        mask_modules: dict[str, _GumbelMaskParametrization] = {}

        for layer_name in layers:
            mod = modules.get(layer_name)
            weight = getattr(mod, "weight", None) if mod is not None else None
            if mod is None or not isinstance(weight, torch.Tensor):
                continue
            param_mod = _GumbelMaskParametrization(
                tuple(weight.shape),
                init_logit=self.init_logit,
                tau=self.tau,
                device=weight.device,
            )
            parametrize.register_parametrization(mod, "weight", param_mod)
            param_mod.train(True)
            mask_modules[layer_name] = param_mod

        if not mask_modules:
            raise ValueError("No maskable target_layers with weights found.")

        device = next(iter(mask_modules.values())).logits.device
        optimizer = torch.optim.Adam(
            [m.logits for m in mask_modules.values()], lr=self.lr
        )

        for _ in range(self.epochs):
            for batch in dataloader:
                x, y = unpack_batch(batch)
                x, y = x.to(device), y.to(device)
                batch_size = max(1, x.shape[0])
                optimizer.zero_grad()
                logits = work_model(x)
                task_loss = self.loss_fn(logits, y)
                logit_penalty = sum(m.logits.sum() for m in mask_modules.values())
                total_loss = task_loss + (self.mask_loss_weight / batch_size) * logit_penalty
                total_loss.backward()
                optimizer.step()

        binary_masks = {
            k: (m.logits.detach() >= 0.0).clone() for k, m in mask_modules.items()
        }
        raw_scores = {k: m.logits.detach().clone() for k, m in mask_modules.items()}
        return SubnetworkMask(
            name=mask_name,
            granularity="weight",
            masks=binary_masks,
            scores=raw_scores,
            metadata={
                "method": self.name,
                "mask_loss_weight": self.mask_loss_weight,
                "init_logit": self.init_logit,
                "tau": self.tau,
            },
        )
