"""Unified SOTA subnetwork discovery backends.

Provides a common :class:`SubnetworkDiscoverer` interface producing canonical
:class:`SubnetworkMask` objects across:
- :class:`NeuroSurgeonDiscoverer` (Lepori et al. — Continuous Sparsification,
  Hard Concrete, Magnitude Pruning via ``libs/NeuroSurgeon``)
- :class:`CsordasGumbelDiscoverer` (Csordas et al. — Gumbel-Sigmoid STE +
  linear logit penalty, modernized for PyTorch 2.x without ``_parameters`` hacks)
- :class:`RandomMaskDiscoverer` (cardinality-matched control baseline)
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from positronixlab.metamodel.sota.base import get_maskable_layer_names
from positronixlab.metamodel.sota.neurosurgeon_adapter import wrap_with_neurosurgeon
from positronixlab.tasks.sota.masks import SubnetworkMask


def _unpack_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, (tuple, list)):
        return batch[0], batch[1]
    raise TypeError(f"Expected (x, y) or (x, y, idx) batch, got {type(batch).__name__}")


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


class SubnetworkDiscoverer(ABC):
    """Base interface for functional modularity / subnetwork discovery methods."""

    name: str = "base"

    @abstractmethod
    def discover(
        self,
        model: nn.Module,
        dataloader: Iterable[Any],
        target_layers: Optional[List[str]] = None,
        mask_name: str = "subnetwork",
        **kwargs: Any,
    ) -> SubnetworkMask:
        """Discover a subnetwork mask on *model* using *dataloader*."""


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
                x, y = _unpack_batch(batch)
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


class _GumbelMaskParametrization(nn.Module):
    """PyTorch 2.x parametrization applying Csordas Gumbel-Sigmoid STE masks."""

    def __init__(
        self,
        shape: Tuple[int, ...],
        init_logit: float = 2.0,
        tau: float = 1.0,
        device: Optional[torch.device] = None,
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
        n_mask_samples: int = 1,
        mask_bias: bool = False,
        grad_clip: Optional[float] = 1.0,
        loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        if n_mask_samples < 1:
            raise ValueError(f"n_mask_samples must be >= 1, got {n_mask_samples}")
        self.mask_loss_weight = mask_loss_weight
        self.lr = lr
        self.epochs = epochs
        self.init_logit = init_logit
        self.tau = tau
        self.n_mask_samples = n_mask_samples
        self.mask_bias = mask_bias
        self.grad_clip = grad_clip
        self.loss_fn = loss_fn or F.cross_entropy

    def discover(
        self,
        model: nn.Module,
        dataloader: Iterable[Any],
        target_layers: Optional[List[str]] = None,
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
        mask_modules: Dict[str, _GumbelMaskParametrization] = {}
        bias_mask_modules: Dict[str, _GumbelMaskParametrization] = {}

        for layer_name in layers:
            mod = modules.get(layer_name)
            if mod is None or not hasattr(mod, "weight") or mod.weight is None:
                continue
            param_mod = _GumbelMaskParametrization(
                tuple(mod.weight.shape),
                init_logit=self.init_logit,
                tau=self.tau,
                device=mod.weight.device,
            )
            parametrize.register_parametrization(mod, "weight", param_mod)
            param_mod.train(True)
            mask_modules[layer_name] = param_mod

            if self.mask_bias and getattr(mod, "bias", None) is not None:
                b_mod = _GumbelMaskParametrization(
                    tuple(mod.bias.shape),
                    init_logit=self.init_logit,
                    tau=self.tau,
                    device=mod.bias.device,
                )
                parametrize.register_parametrization(mod, "bias", b_mod)
                b_mod.train(True)
                bias_mask_modules[layer_name] = b_mod

        if not mask_modules:
            raise ValueError("No maskable target_layers with weights found.")

        all_mask_mods = list(mask_modules.values()) + list(bias_mask_modules.values())
        all_logit_params = [m.logits for m in all_mask_mods]
        device = all_logit_params[0].device
        optimizer = torch.optim.Adam(all_logit_params, lr=self.lr)

        for _ in range(self.epochs):
            for batch in dataloader:
                x, y = _unpack_batch(batch)
                x, y = x.to(device), y.to(device)
                batch_size = max(1, x.shape[0])
                optimizer.zero_grad()

                # Exact Csordas multi-mask gradient estimation: K independent mask samples
                # over K chunks of the minibatch, accumulating gradients before stepping.
                k = min(self.n_mask_samples, batch_size)
                if k == 1:
                    logits = work_model(x)
                    task_loss = self.loss_fn(logits, y)
                    task_loss.backward()
                else:
                    x_chunks = torch.chunk(x, k, dim=0)
                    y_chunks = torch.chunk(y, k, dim=0)
                    for xc, yc in zip(x_chunks, y_chunks):
                        logits = work_model(xc)
                        chunk_loss = self.loss_fn(logits, yc) * (xc.shape[0] / batch_size)
                        chunk_loss.backward()

                # Csordas et al. linear logit penalty scaled by batch size: (beta / B) * sum(l_i)
                logit_penalty = sum(m.logits.sum() for m in all_mask_mods)
                ((self.mask_loss_weight / batch_size) * logit_penalty).backward()

                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(all_logit_params, self.grad_clip)
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
                "n_mask_samples": self.n_mask_samples,
                "mask_bias": self.mask_bias,
            },
        )


class RandomMaskDiscoverer(SubnetworkDiscoverer):
    """Control baseline producing a random subnetwork mask of target density."""

    name = "random"

    def __init__(
        self,
        density: float = 0.5,
        granularity: str = "weight",
        seed: int = 0,
    ) -> None:
        if not (0.0 <= density <= 1.0):
            raise ValueError(f"density must be in [0, 1], got {density}")
        self.density = density
        self.granularity = granularity
        self.seed = seed

    def discover(
        self,
        model: nn.Module,
        dataloader: Iterable[Any] = (),
        target_layers: Optional[List[str]] = None,
        mask_name: str = "random",
        **kwargs: Any,
    ) -> SubnetworkMask:
        layers = (
            list(target_layers)
            if target_layers is not None
            else get_maskable_layer_names(model)
        )
        modules = dict(model.named_modules())
        gen = torch.Generator()
        gen.manual_seed(self.seed)

        masks: Dict[str, torch.Tensor] = {}
        for layer_name in layers:
            mod = modules.get(layer_name)
            if mod is None or not hasattr(mod, "weight") or mod.weight is None:
                continue
            if self.granularity == "unit":
                shape = (mod.weight.shape[0],)
            else:
                shape = tuple(mod.weight.shape)
            masks[layer_name] = torch.rand(shape, generator=gen) < self.density

        return SubnetworkMask(
            name=mask_name,
            granularity=self.granularity,  # type: ignore[arg-type]
            masks=masks,
            metadata={"method": self.name, "density": self.density, "seed": self.seed},
        )
