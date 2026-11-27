"""Csordas et al. (ICLR 2021) Gumbel-Sigmoid STE subnetwork discoverer."""

from __future__ import annotations

import copy
import time
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


def _binary_flip_fraction(
    prev_masks: dict[str, torch.Tensor],
    curr_masks: dict[str, torch.Tensor],
) -> float:
    flips = 0
    total = 0
    for k, curr in curr_masks.items():
        prev = prev_masks[k]
        flips += int((curr ^ prev).sum().item())
        total += int(curr.numel())
    return flips / max(1, total)


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
        *,
        use_amp: bool = False,
        use_compile: bool = False,
        early_stop_patience: int = 0,
        early_stop_tol: float = 1e-3,
    ) -> None:
        self.mask_loss_weight = mask_loss_weight
        self.lr = lr
        self.epochs = epochs
        self.init_logit = init_logit
        self.tau = tau
        self.loss_fn = loss_fn or F.cross_entropy
        self.use_amp = use_amp
        self.use_compile = use_compile
        self.early_stop_patience = early_stop_patience
        self.early_stop_tol = early_stop_tol
        self.last_stats: dict[str, Any] = {}

    def discover(
        self,
        model: nn.Module,
        dataloader: Iterable[Any],
        target_layers: list[str] | None = None,
        mask_name: str = "csordas_gumbel",
        **kwargs: Any,
    ) -> SubnetworkMask:
        import torch.nn.utils.parametrize as parametrize

        t0 = time.monotonic()
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
        run_model: Any = (
            torch.compile(work_model) if (self.use_compile and device.type == "cuda") else work_model
        )
        optimizer = torch.optim.Adam(
            [m.logits for m in mask_modules.values()], lr=self.lr
        )
        amp_enabled = bool(self.use_amp and device.type == "cuda")
        amp_dtype = (
            torch.bfloat16 if (amp_enabled and torch.cuda.is_bf16_supported()) else torch.float16
        )
        scaler = torch.amp.GradScaler(
            device.type, enabled=bool(amp_enabled and amp_dtype == torch.float16)
        )

        prev_binary: dict[str, torch.Tensor] | None = None
        prev_loss: float | None = None
        stable_epochs = 0
        epochs_run = 0
        last_task_loss = 0.0

        for epoch in range(max(0, self.epochs)):
            epochs_run = epoch + 1
            ep_loss = 0.0
            n_batches = 0
            for batch in dataloader:
                x, y = unpack_batch(batch)
                if x.device != device:
                    x = x.to(device, non_blocking=True)
                if y.device != device:
                    y = y.to(device, non_blocking=True)
                batch_size = max(1, int(x.shape[0]))
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
                ):
                    logits = run_model(x)
                    task_loss = self.loss_fn(logits, y)
                    logit_penalty = sum(m.logits.sum() for m in mask_modules.values())
                    total_loss = task_loss + (self.mask_loss_weight / batch_size) * logit_penalty
                if scaler.is_enabled():
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    optimizer.step()
                ep_loss += float(task_loss.detach().item())
                n_batches += 1

            last_task_loss = ep_loss / max(1, n_batches)

            if self.early_stop_patience > 0:
                curr_binary = {k: (m.logits.detach() >= 0.0) for k, m in mask_modules.items()}
                if prev_binary is not None and prev_loss is not None:
                    flip_frac = _binary_flip_fraction(prev_binary, curr_binary)
                    loss_delta = abs(last_task_loss - prev_loss)
                    if flip_frac <= self.early_stop_tol and loss_delta <= self.early_stop_tol:
                        stable_epochs += 1
                        if stable_epochs >= self.early_stop_patience:
                            break
                    else:
                        stable_epochs = 0
                prev_binary = curr_binary
                prev_loss = last_task_loss

        binary_masks = {
            k: (m.logits.detach() >= 0.0).clone() for k, m in mask_modules.items()
        }
        raw_scores = {k: m.logits.detach().clone() for k, m in mask_modules.items()}
        self.last_stats = {
            "epochs_run": epochs_run,
            "final_task_loss": last_task_loss,
            "duration_sec": time.monotonic() - t0,
            "backend": "native_fast",
        }
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
                **self.last_stats,
            },
        )
