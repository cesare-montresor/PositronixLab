"""Clean adapter around ``NeuroSurgeon`` (Lepori et al., 2023/2024).

Resolves the four HuggingFace-specific caveats of ``NeuroSurgeon.Models.CircuitModel``:
1. ``CircuitModel.forward`` passes ``return_dict=True`` -> bridged via :class:`HFForwardShim`.
2. ``CircuitModel.__call__`` accepts kwargs only -> :class:`NeuroSurgeonWrappedModel`
   supports standard positional ``model(x)`` and keyword ``model(x=x)`` calls.
3. ``CircuitModel._handle_model_freezing`` only freezes ``weight``/``bias``/
   ``position_embeddings``/``cls_token`` -> we freeze *all* non-mask parameters.
4. ``CircuitModel`` mutates the wrapped model in-place -> we ``copy.deepcopy``
   by default so the caller's base model remains untouched.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from positronixlab.metamodel.sota.base import get_maskable_layer_names


def _import_neurosurgeon():
    """Import ``NeuroSurgeon`` (Models + Masking), falling back to ``libs/NeuroSurgeon``.

    Handles ``NeuroSurgeon``'s undeclared ``torchmetrics`` dependency in
    ``NeuroSurgeon.Probing.circuit_probe`` so ``Models`` and ``Masking`` work
    even when ``torchmetrics`` is not installed.
    """
    import importlib
    import types

    repo_root = Path(__file__).resolve().parents[3]
    libs_ns = repo_root / "libs" / "NeuroSurgeon"
    if libs_ns.exists() and str(libs_ns) not in sys.path:
        sys.path.insert(0, str(libs_ns))

    try:
        return importlib.import_module("NeuroSurgeon")
    except ModuleNotFoundError as exc:
        if exc.name != "torchmetrics":
            raise ImportError(
                "NeuroSurgeon is not installed or accessible. "
                "Install it with `pip install -e libs/NeuroSurgeon`."
            ) from exc

        # Load NeuroSurgeon.Masking & NeuroSurgeon.Models without triggering Probing
        ns_pkg_dir = libs_ns / "NeuroSurgeon"
        if not ns_pkg_dir.is_dir():
            # Try locating via existing spec if installed in site-packages
            spec = importlib.util.find_spec("NeuroSurgeon")
            if spec and spec.submodule_search_locations:
                ns_pkg_dir = Path(list(spec.submodule_search_locations)[0])
        if not ns_pkg_dir.is_dir():
            raise ImportError(
                "NeuroSurgeon requires `torchmetrics` for `Probing`; install "
                "`torchmetrics` or ensure `libs/NeuroSurgeon` is present."
            ) from exc

        # Clean any half-initialized submodules before creating lightweight package stub
        for k in [k for k in sys.modules if k == "NeuroSurgeon" or k.startswith("NeuroSurgeon.")]:
            sys.modules.pop(k, None)

        pkg = types.ModuleType("NeuroSurgeon")
        pkg.__path__ = [str(ns_pkg_dir)]  # type: ignore[attr-defined]
        pkg.__package__ = "NeuroSurgeon"
        sys.modules["NeuroSurgeon"] = pkg
        pkg.Masking = importlib.import_module("NeuroSurgeon.Masking")  # type: ignore[attr-defined]
        pkg.Models = importlib.import_module("NeuroSurgeon.Models")  # type: ignore[attr-defined]
        return pkg
    except ImportError as exc:
        raise ImportError(
            "NeuroSurgeon is not installed or accessible. "
            "Install it with `pip install -e libs/NeuroSurgeon`."
        ) from exc


def is_neurosurgeon_available() -> bool:
    """Return ``True`` if ``NeuroSurgeon`` can be imported."""
    try:
        _import_neurosurgeon()
        return True
    except ImportError:
        return False


class HFForwardShim(nn.Module):
    """Adapts any standard PyTorch ``nn.Module`` to ``CircuitModel``'s HF interface.

    - Accepts ``return_dict=True`` (silently stripped for non-HF models).
    - Wraps plain tensor outputs in ``SimpleNamespace(logits=out, loss=None)``.
    - Preserves submodule attribute paths under ``self.model`` (with
      ``target_layers`` automatically prefixed by ``"model."``).
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, *args: Any, return_dict: bool = True, **kwargs: Any) -> Any:
        if "x" in kwargs and len(kwargs) == 1 and not args:
            out = self.model(kwargs["x"])
        else:
            out = self.model(*args, **kwargs)
        if hasattr(out, "logits") or hasattr(out, "loss"):
            return out
        return SimpleNamespace(logits=out, loss=None)


class NeuroSurgeonWrappedModel(nn.Module):
    """Ergonomic wrapper around ``NeuroSurgeon.Models.CircuitModel``."""

    def __init__(
        self,
        circuit_model: nn.Module,
        target_layers: List[str],
        mask_method: str,
        mask_unit: str,
        l0_lambda: float = 1e-8,
        shimmed: bool = True,
    ) -> None:
        super().__init__()
        self.circuit_model = circuit_model
        self.target_layers = list(target_layers)
        self.mask_method = mask_method
        self.mask_unit = mask_unit
        self.l0_lambda = l0_lambda
        self._shimmed = shimmed

    @property
    def temperature(self) -> float:
        return float(self.circuit_model.temperature)

    @temperature.setter
    def temperature(self, value: float) -> None:
        self.circuit_model.temperature = value

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        if args:
            if len(args) == 1 and not kwargs:
                out = self.circuit_model(x=args[0])
            else:
                # Invoke wrapped_model directly via CircuitModel.forward kwargs
                out = self.circuit_model.wrapped_model(*args, **kwargs)
        else:
            out = self.circuit_model(**kwargs)
        return out.logits if hasattr(out, "logits") else out

    def mask_parameters(self) -> List[nn.Parameter]:
        """Return all learnable mask parameters."""
        return [
            p
            for name, p in self.circuit_model.named_parameters()
            if p.requires_grad and "mask_params" in name
        ]

    def l0_loss(self) -> torch.Tensor:
        """Return unscaled L0 loss across all masked layers."""
        val = self.circuit_model._compute_l0_loss()
        if isinstance(val, torch.Tensor):
            return val
        device = next(self.parameters()).device
        return torch.tensor(float(val), device=device)

    def l0_statistics(self) -> Dict[str, Any]:
        """Return ``CircuitModel.compute_l0_statistics()``."""
        return self.circuit_model.compute_l0_statistics()

    def set_ablate_mode(self, ablation: str, force_resample: bool = False) -> None:
        self.circuit_model.set_ablate_mode(ablation, force_resample=force_resample)

    def use_masks(self, value: bool, name_list: Optional[List[str]] = None) -> None:
        if name_list is not None and self._shimmed:
            prefixed = [
                n if n.startswith("model.") else f"model.{n}" for n in name_list
            ]
        else:
            prefixed = name_list
        self.circuit_model.use_masks(value, name_list=prefixed)

    def extract_mask_tensors(
        self,
        binarize: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Extract canonical dotted-name masks and raw mask scores from ``CircuitModel``.

        Returns:
            ``(binary_masks, raw_scores)`` keyed by the original model's dotted
            layer names (with the ``"model."`` shim prefix stripped).
        """
        from NeuroSurgeon.Masking.mask_layer import MaskLayer  # type: ignore

        binary_masks: Dict[str, torch.Tensor] = {}
        raw_scores: Dict[str, torch.Tensor] = {}

        was_training = self.circuit_model.training
        self.circuit_model.eval()
        try:
            for name, mod in self.circuit_model.wrapped_model.named_modules():
                if not isinstance(mod, MaskLayer):
                    continue
                canonical = (
                    name[len("model.") :]
                    if (self._shimmed and name.startswith("model."))
                    else name
                )
                if hasattr(mod, "weight_mask_params"):
                    raw = mod.weight_mask_params.detach().clone()
                    raw_scores[canonical] = raw
                with torch.no_grad():
                    mask_t = mod._compute_mask("weight_mask_params").detach().clone()
                if self.mask_unit == "neuron":
                    # Collapse repeated in_features / in_channels dim to 1D unit mask
                    if mask_t.ndim >= 2:
                        mask_1d = mask_t.reshape(mask_t.shape[0], -1).any(dim=1)
                    else:
                        mask_1d = mask_t.bool()
                    binary_masks[canonical] = mask_1d if binarize else mask_1d.float()
                else:
                    binary_masks[canonical] = (mask_t > 0.5) if binarize else mask_t
        finally:
            self.circuit_model.train(was_training)

        return binary_masks, raw_scores

    def extract_subnetwork_mask(self, name: str = "neurosurgeon") -> Any:
        """Export discovered mask as a canonical :class:`SubnetworkMask`."""
        from positronixlab.tasks.sota.masks import SubnetworkMask

        granularity = "unit" if self.mask_unit == "neuron" else "weight"
        binary_masks, raw_scores = self.extract_mask_tensors(binarize=True)
        stats = self.l0_statistics()
        return SubnetworkMask(
            name=name,
            granularity=granularity,
            masks=binary_masks,
            scores=raw_scores if raw_scores else None,
            metadata={
                "method": f"neurosurgeon:{self.mask_method}",
                "mask_unit": self.mask_unit,
                "l0": stats.get("total_l0"),
                "max_l0": stats.get("max_l0"),
            },
        )


def _default_mask_hparams(mask_method: str, mask_unit: str) -> Dict[str, Any]:
    if mask_method == "continuous_sparsification":
        return {
            "ablation": "none",
            "mask_unit": mask_unit,
            "mask_bias": False,
            "mask_init_value": 0.0,
        }
    if mask_method == "hard_concrete":
        return {
            "ablation": "none",
            "mask_unit": mask_unit,
            "mask_bias": False,
            "mask_init_percentage": 0.9,
        }
    if mask_method == "magnitude_pruning":
        return {
            "ablation": "none",
            "mask_bias": False,
            "prune_percentage": 0.5,
        }
    raise ValueError(f"Unsupported NeuroSurgeon mask_method: {mask_method!r}")


def wrap_with_neurosurgeon(
    model: nn.Module,
    target_layers: Optional[List[str]] = None,
    mask_method: str = "continuous_sparsification",
    mask_unit: str = "weight",
    mask_hparams: Optional[Dict[str, Any]] = None,
    freeze_base: bool = True,
    l0_lambda: float = 1e-8,
    deepcopy_model: bool = True,
) -> NeuroSurgeonWrappedModel:
    """Wrap any PyTorch ``nn.Module`` in ``NeuroSurgeon.Models.CircuitModel``."""
    _import_neurosurgeon()
    from NeuroSurgeon.Masking.mask_layer import MaskLayer  # type: ignore
    from NeuroSurgeon.Models import CircuitConfig, CircuitModel  # type: ignore

    base = copy.deepcopy(model) if deepcopy_model else model
    layers = (
        list(target_layers)
        if target_layers is not None
        else get_maskable_layer_names(base)
    )
    if not layers:
        raise ValueError("No maskable target_layers found in model.")

    hparams = _default_mask_hparams(mask_method, mask_unit)
    if mask_hparams:
        hparams.update(mask_hparams)

    shimmed = HFForwardShim(base)
    prefixed_layers = [f"model.{name}" for name in layers]

    cfg = CircuitConfig(
        mask_method=mask_method,
        mask_hparams=hparams,
        target_layers=prefixed_layers,
        freeze_base=freeze_base,
        add_l0=False,  # Managed explicitly so non-HF models returning tensors work
        l0_lambda=l0_lambda,
    )
    circuit_model = CircuitModel(cfg, shimmed)

    # Harden base freezing: freeze ALL non-mask parameters (fixes NeuroSurgeon caveat #4)
    if freeze_base:
        for mod_name, mod in circuit_model.wrapped_model.named_modules():
            if isinstance(mod, MaskLayer):
                for p_name, p in mod.named_parameters(recurse=False):
                    p.requires_grad = "mask_params" in p_name
            else:
                for p in mod.parameters(recurse=False):
                    p.requires_grad = False

    return NeuroSurgeonWrappedModel(
        circuit_model=circuit_model,
        target_layers=layers,
        mask_method=mask_method,
        mask_unit=mask_unit,
        l0_lambda=l0_lambda,
        shimmed=True,
    )


def strip_neurosurgeon_mask_keys(
    state_dict: Dict[str, Any],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Separate base weights from a saved ``CircuitModel`` ``state_dict``."""
    clean_weights: Dict[str, torch.Tensor] = {}
    mask_params: Dict[str, torch.Tensor] = {}

    for key, val in state_dict.items():
        if not isinstance(val, torch.Tensor):
            continue
        k = key
        for prefix in ("circuit_model.wrapped_model.model.", "wrapped_model.model.", "wrapped_model."):
            if k.startswith(prefix):
                k = k[len(prefix) :]
                break
        if k.endswith(".weight_mask_params"):
            mask_params[k[: -len(".weight_mask_params")]] = val
            continue
        if k.endswith(".bias_mask_params"):
            continue
        clean_weights[k] = val

    return clean_weights, mask_params
