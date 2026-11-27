"""Native PyTorch 2.x Continuous Sparsification discoverer (Lepori et al., NeurIPS 2023)."""

from __future__ import annotations

import copy
import time
from collections.abc import Callable, Iterable
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize

from positronixlab.baselines.base import get_maskable_layer_names
from positronixlab.tasks.discoverer import SubnetworkDiscoverer, unpack_batch
from positronixlab.tasks.masks import SubnetworkMask


class _ContSparseMaskParametrization(nn.Module):
    """Lepori (2023) / Savarese (2020) Continuous Sparsification parametrization.

    Supports both ``mask_unit="weight"`` (shape == weight.shape) and
    ``mask_unit="neuron"`` (shape == (out_units, 1, ...)).
    """

    def __init__(
        self,
        weight_shape: tuple[int, ...],
        mask_unit: str,
        init_value: float,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.mask_unit = mask_unit
        if mask_unit == "neuron":
            param_shape = (weight_shape[0],) + (1,) * (len(weight_shape) - 1)
        else:
            param_shape = weight_shape
        self.scores = nn.Parameter(
            torch.full(param_shape, float(init_value), device=device)
        )
        self.temperature: float = 1.0

    def soft_mask(self) -> torch.Tensor:
        return torch.sigmoid(self.temperature * self.scores)

    def binary_mask(self) -> torch.Tensor:
        return self.scores >= 0.0

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if self.training:
            mask = self.soft_mask().to(weight.dtype)
        else:
            mask = self.binary_mask().to(weight.dtype)
        return weight * mask


class ContinuousSparsificationDiscoverer(SubnetworkDiscoverer):
    """Pure-PyTorch 2.x Lepori (NeurIPS 2023) / Savarese (2020) Continuous Sparsification.

    Uses ``torch.nn.utils.parametrize`` on a deep copy of *model* with exponential
    temperature annealing ``1.0 -> max_temp`` and L0 soft-mask regularization,
    requiring zero external ``NeuroSurgeon`` or HuggingFace wrapper dependencies.
    """

    name = "lepori:continuous_sparsification"

    def __init__(
        self,
        mask_unit: Literal["weight", "neuron"] = "weight",
        l0_lambda: float = 1e-4,
        lr: float = 5e-2,
        epochs: int = 15,
        init_value: float = 0.0,
        max_temp: float = 200.0,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        *,
        use_amp: bool = False,
        use_compile: bool = False,
    ) -> None:
        if mask_unit not in ("weight", "neuron"):
            raise ValueError(f"mask_unit must be 'weight' or 'neuron', got {mask_unit!r}")
        self.mask_unit = mask_unit
        self.l0_lambda = l0_lambda
        self.lr = lr
        self.epochs = epochs
        self.init_value = init_value
        self.max_temp = max_temp
        self.loss_fn = loss_fn or F.cross_entropy
        self.use_amp = use_amp
        self.use_compile = use_compile
        self.last_stats: dict[str, Any] = {}

    def discover(
        self,
        model: nn.Module,
        dataloader: Iterable[Any],
        target_layers: list[str] | None = None,
        mask_name: str = "lepori:continuous_sparsification",
        **kwargs: Any,
    ) -> SubnetworkMask:
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
        mask_modules: dict[str, _ContSparseMaskParametrization] = {}

        for layer_name in layers:
            mod = modules.get(layer_name)
            weight = getattr(mod, "weight", None) if mod is not None else None
            if mod is None or not isinstance(weight, torch.Tensor):
                continue
            pm = _ContSparseMaskParametrization(
                tuple(weight.shape),
                mask_unit=self.mask_unit,
                init_value=self.init_value,
                device=weight.device,
            )
            parametrize.register_parametrization(mod, "weight", pm)
            pm.train(True)
            mask_modules[layer_name] = pm

        if not mask_modules:
            raise ValueError("No maskable target_layers with weights found.")

        device = next(iter(mask_modules.values())).scores.device
        run_model: Any = (
            torch.compile(work_model) if (self.use_compile and device.type == "cuda") else work_model
        )
        optimizer = torch.optim.Adam(
            [m.scores for m in mask_modules.values()], lr=self.lr
        )
        amp_enabled = bool(self.use_amp and device.type == "cuda")
        amp_dtype = (
            torch.bfloat16 if (amp_enabled and torch.cuda.is_bf16_supported()) else torch.float16
        )
        scaler = torch.amp.GradScaler(
            device.type, enabled=bool(amp_enabled and amp_dtype == torch.float16)
        )

        temp = 1.0
        temp_mult = self.max_temp ** (1.0 / max(1, self.epochs))
        epochs_run = 0
        last_task_loss = 0.0

        for epoch in range(max(0, self.epochs)):
            epochs_run = epoch + 1
            for pm in mask_modules.values():
                pm.temperature = temp

            ep_loss = 0.0
            n_batches = 0
            for batch in dataloader:
                x, y = unpack_batch(batch)
                if x.device != device:
                    x = x.to(device, non_blocking=True)
                if y.device != device:
                    y = y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
                ):
                    logits = run_model(x)
                    task_loss = self.loss_fn(logits, y)
                    l0_loss = sum(pm.soft_mask().sum() for pm in mask_modules.values())
                    total_loss = task_loss + self.l0_lambda * l0_loss
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
            temp *= temp_mult

        granularity = "unit" if self.mask_unit == "neuron" else "weight"
        if self.mask_unit == "neuron":
            binary_masks = {
                k: pm.binary_mask().detach().reshape(-1).clone()
                for k, pm in mask_modules.items()
            }
            raw_scores = {
                k: pm.scores.detach().reshape(-1).clone()
                for k, pm in mask_modules.items()
            }
        else:
            binary_masks = {
                k: pm.binary_mask().detach().clone() for k, pm in mask_modules.items()
            }
            raw_scores = {
                k: pm.scores.detach().clone() for k, pm in mask_modules.items()
            }

        self.last_stats = {
            "epochs_run": epochs_run,
            "final_task_loss": last_task_loss,
            "final_temp": temp,
            "duration_sec": time.monotonic() - t0,
            "backend": "native_fast",
        }
        return SubnetworkMask(
            name=mask_name,
            granularity=granularity,  # type: ignore[arg-type]
            masks=binary_masks,
            scores=raw_scores,
            metadata={
                "method": self.name,
                "mask_unit": self.mask_unit,
                "l0_lambda": self.l0_lambda,
                "max_temp": self.max_temp,
                **self.last_stats,
            },
        )
