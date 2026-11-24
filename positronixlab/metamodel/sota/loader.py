"""Unified SOTA Model Loader for PositronixLab.

Provides a single, registry-driven entry point (:class:`ModelLoader`) to create
and load checkpoints for architectures across all three integrated SOTA repos:

- **Csordas et al.** (``are_neural_nets_modular``):
  ``csordas:mlp``, ``csordas:conv``, ``csordas:lstm``, ``csordas:resnet110``,
  ``csordas:transformer``
- **Lepori et al.** (``Compositional_Subnetworks``):
  ``lepori:cvr_resnet18``, ``lepori:cvr_resnet50``, ``lepori:cvr_wideresnet50``,
  ``lepori:cvr_vit``, ``lepori:syneval_bert``
- **NeuroSurgeon** (``libs/NeuroSurgeon``):
  ``ModelLoader.wrap_neurosurgeon(...)`` or ``load(..., wrap_neurosurgeon=True)``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from positronixlab.metamodel.sota.base import SOTAModel, SOTAModelSpec
from positronixlab.metamodel.sota.csordas_models import (
    CsordasConvModel,
    CsordasFeedforwardModel,
    CsordasLSTMModel,
    CsordasResNet110,
    CsordasTransformerEncDec,
    strip_csordas_mask_keys,
)
from positronixlab.metamodel.sota.lepori_models import (
    CVRResNetModel,
    CVRViTModel,
    SynEvalBertModel,
    strip_lepori_l0_keys,
)
from positronixlab.metamodel.sota.neurosurgeon_adapter import (
    NeuroSurgeonWrappedModel,
    strip_neurosurgeon_mask_keys,
    wrap_with_neurosurgeon,
)


def _detect_checkpoint_format(state_dict: Dict[str, Any]) -> str:
    """Heuristically detect whether a checkpoint came from Csordas, Lepori, NeuroSurgeon, or plain PyTorch."""
    keys = list(state_dict.keys())
    if "model" in state_dict and isinstance(state_dict["model"], dict):
        keys = list(state_dict["model"].keys())
    elif "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        keys = list(state_dict["state_dict"].keys())

    if any(k.endswith(".weight_mask_params") for k in keys):
        return "neurosurgeon"
    if any(k.endswith(".mask_weight") for k in keys):
        return "lepori"
    if any(k.startswith("masks.") for k in keys):
        return "csordas"
    return "plain"


class ModelLoader:
    """Unified factory & checkpoint loader for SOTA modularity architectures."""

    _REGISTRY: Dict[str, SOTAModelSpec] = {}

    @classmethod
    def register(
        cls,
        arch: str,
        source: str,
        task_type: str,
        builder: Callable[..., SOTAModel],
        description: str = "",
        default_exclude_patterns: Tuple[str, ...] = (),
    ) -> None:
        cls._REGISTRY[arch] = SOTAModelSpec(
            arch=arch,
            source=source,
            task_type=task_type,
            builder=builder,
            description=description,
            default_exclude_patterns=default_exclude_patterns,
        )

    @classmethod
    def list_models(cls, source: Optional[str] = None) -> List[str]:
        """Return sorted list of registered architecture identifiers."""
        if source is None:
            return sorted(cls._REGISTRY.keys())
        return sorted(k for k, v in cls._REGISTRY.items() if v.source == source)

    @classmethod
    def get_spec(cls, arch: str) -> SOTAModelSpec:
        if arch not in cls._REGISTRY:
            raise ValueError(
                f"Unknown architecture {arch!r}. Available: {cls.list_models()}"
            )
        return cls._REGISTRY[arch]

    @classmethod
    def create(
        cls,
        arch: str,
        device: Optional[torch.device | str] = None,
        **kwargs: Any,
    ) -> SOTAModel:
        """Instantiate a fresh standard-PyTorch SOTA model."""
        spec = cls.get_spec(arch)
        model = spec.builder(device=device, **kwargs)
        return model

    @classmethod
    def load(
        cls,
        arch: str,
        checkpoint: Optional[str | Path | Dict[str, Any]] = None,
        backbone_checkpoint: Optional[str | Path] = None,
        mlp_checkpoint: Optional[str | Path] = None,
        device: Optional[torch.device | str] = None,
        strict: bool = False,
        return_mask: bool = False,
        **model_kwargs: Any,
    ) -> Any:
        """Create a model and optionally load weights (auto-detecting SOTA checkpoint formats).

        Supports:
        - Plain PyTorch ``state_dict`` (``.pt`` / ``.pth`` or dict)
        - Csordas ``MaskedModel`` checkpoints (strips ``masks.*`` and ``model.`` prefix)
        - Lepori ``Compositional_Subnetworks`` checkpoints (single file or split
          ``backbone_checkpoint`` + ``mlp_checkpoint``, stripping ``*.mask_weight``)
        - NeuroSurgeon ``CircuitModel`` checkpoints (stripping ``*.weight_mask_params``)

        When ``return_mask=True``, returns ``(model, extracted_mask_or_None)`` where
        ``extracted_mask_or_None`` is a :class:`SubnetworkMask` if mask parameters
        were present in the checkpoint.
        """
        model = cls.create(arch, device=device, **model_kwargs)
        extracted_scores: Dict[str, torch.Tensor] = {}
        ckpt_format = "plain"

        # Lepori split-component loading (backbone.pt + mlp.pt)
        if (backbone_checkpoint is not None or mlp_checkpoint is not None) and isinstance(
            model, CVRResNetModel
        ):
            extracted_scores = model.load_lepori_components(
                backbone_path=backbone_checkpoint,
                mlp_path=mlp_checkpoint,
                strict=strict,
            )
            ckpt_format = "lepori"
        elif checkpoint is not None:
            raw = (
                checkpoint
                if isinstance(checkpoint, dict)
                else torch.load(checkpoint, map_location=model.device, weights_only=True)
            )
            ckpt_format = _detect_checkpoint_format(raw)
            if ckpt_format == "neurosurgeon":
                clean, extracted_scores = strip_neurosurgeon_mask_keys(raw)
                model.load_state_dict(clean, strict=strict)
            elif ckpt_format == "lepori":
                clean, extracted_scores = strip_lepori_l0_keys(raw)
                model.load_state_dict(clean, strict=strict)
            elif ckpt_format == "csordas":
                clean, stage_masks = strip_csordas_mask_keys(raw)
                model.load_state_dict(clean, strict=strict)
                if stage_masks:
                    first_stage = sorted(stage_masks.keys())[0]
                    extracted_scores = stage_masks[first_stage]
            else:
                model.load_state_dict(raw, strict=strict)

        if not return_mask:
            return model

        if not extracted_scores:
            return model, None

        from positronixlab.tasks.sota.masks import SubnetworkMask

        binary = {k: (v > 0).detach().clone() for k, v in extracted_scores.items()}
        mask = SubnetworkMask(
            name=f"{arch}_checkpoint_mask",
            granularity="weight",
            masks=binary,
            scores={k: v.detach().clone() for k, v in extracted_scores.items()},
            metadata={"source_format": ckpt_format, "arch": arch},
        )
        return model, mask

    @classmethod
    def from_config(
        cls,
        config: Any,
        arch: Optional[str] = None,
        name: Optional[str] = None,
        device: Optional[torch.device | str] = None,
        **kwargs: Any,
    ) -> nn.Module:
        """Unified model loader accepting either a ``PipelineConfig`` (``MetaModel``) or SOTA arch.

        - If ``arch`` is ``None`` or ``"positronix:metamodel"`` (and ``config`` has no
          ``sota_arch`` override), instantiates :class:`MetaModel`.
        - If ``arch == "csordas:mlp"``, automatically derives ``n_inputs``, ``n_outputs``,
          and ``layer_sizes`` from ``config`` unless explicitly overridden in ``kwargs``.
        - For other registered architectures (e.g. ``csordas:conv``, ``csordas:resnet110``,
          ``csordas:lstm``, ``csordas:transformer``), delegates to :meth:`create`
          (placeholder/non-FC architectures can be invoked explicitly once ready).
        """
        resolved_arch = arch or getattr(config, "sota_arch", None) or "positronix:metamodel"
        if resolved_arch == "positronix:metamodel":
            from positronixlab.metamodel.metamodel import MetaModel

            model_name = name or getattr(config, "name", None) or "MetaModel"
            return MetaModel(name=model_name, config=config, device=device)

        if resolved_arch == "csordas:mlp":
            n_inputs = kwargs.pop("n_inputs", getattr(config, "input_data_shape", None))
            n_outputs = kwargs.pop("n_outputs", getattr(config, "output_data_shape", None))
            if n_inputs is None or n_outputs is None:
                raise ValueError(
                    "csordas:mlp requires n_inputs and n_outputs (either in config or kwargs)."
                )
            if "layer_sizes" not in kwargs and hasattr(config, "input_hidden_dim"):
                n_hidden = max(1, int(getattr(config, "input_num_layers", 3)))
                kwargs["layer_sizes"] = [int(config.input_hidden_dim)] * n_hidden
            return cls.create(
                resolved_arch,
                device=device,
                n_inputs=n_inputs,
                n_outputs=n_outputs,
                name=name,
                **kwargs,
            )

        return cls.create(resolved_arch, device=device, name=name, **kwargs)

    @classmethod
    def wrap_neurosurgeon(
        cls,
        model: nn.Module,
        target_layers: Optional[List[str]] = None,
        mask_method: str = "continuous_sparsification",
        mask_unit: str = "weight",
        mask_hparams: Optional[Dict[str, Any]] = None,
        freeze_base: bool = True,
        l0_lambda: float = 1e-8,
        deepcopy_model: bool = True,
    ) -> NeuroSurgeonWrappedModel:
        """Wrap any ``nn.Module`` / :class:`SOTAModel` in ``NeuroSurgeon``."""
        return wrap_with_neurosurgeon(
            model=model,
            target_layers=target_layers,
            mask_method=mask_method,
            mask_unit=mask_unit,
            mask_hparams=mask_hparams,
            freeze_base=freeze_base,
            l0_lambda=l0_lambda,
            deepcopy_model=deepcopy_model,
        )


# ---------------------------------------------------------------------------
# Default Registrations
# ---------------------------------------------------------------------------

def _build_metamodel(config: Any, name: str = "MetaModel", device: Optional[torch.device | str] = None, **kwargs: Any):
    from positronixlab.metamodel.metamodel import MetaModel

    return MetaModel(name=name, config=config, device=device)


ModelLoader.register(
    arch="positronix:metamodel",
    source="positronix",
    task_type="classification",
    builder=_build_metamodel,
    description="PositronixLab config-driven MetaModel (nn_input -> nn_latent -> nn_output)",
)
ModelLoader.register(
    arch="csordas:mlp",
    source="csordas",
    task_type="classification",
    builder=CsordasFeedforwardModel,
    description="Csordas et al. FeedforwardModel (TupleArithmetic / AddMul / PermutedMNIST)",
)
ModelLoader.register(
    arch="csordas:conv",
    source="csordas",
    task_type="classification",
    builder=CsordasConvModel,
    description="Csordas et al. 4-layer ConvModel (CIFAR-10 class removal / stability)",
)
ModelLoader.register(
    arch="csordas:lstm",
    source="csordas",
    task_type="sequence",
    builder=CsordasLSTMModel,
    description="Csordas et al. LSTMModel (TupleArithmetic / AddMul)",
)
ModelLoader.register(
    arch="csordas:resnet110",
    source="csordas",
    task_type="classification",
    builder=CsordasResNet110,
    description="Csordas et al. CIFAR ResNet-110",
)
ModelLoader.register(
    arch="csordas:transformer",
    source="csordas",
    task_type="seq2seq",
    builder=CsordasTransformerEncDec,
    description="Csordas et al. Encoder-Decoder Transformer (SCAN / DeepMind Math)",
    default_exclude_patterns=("embedding", "pos_encoding"),
)

ModelLoader.register(
    arch="lepori:cvr_resnet18",
    source="lepori",
    task_type="odd_one_out",
    builder=lambda **kw: CVRResNetModel(backbone_name="resnet18", **kw),
    description="Lepori et al. CVR Odd-One-Out ResNet-18 (InstanceNorm2d)",
)
ModelLoader.register(
    arch="lepori:cvr_resnet50",
    source="lepori",
    task_type="odd_one_out",
    builder=lambda **kw: CVRResNetModel(backbone_name="resnet50", **kw),
    description="Lepori et al. CVR Odd-One-Out ResNet-50 (InstanceNorm2d)",
)
ModelLoader.register(
    arch="lepori:cvr_wideresnet50",
    source="lepori",
    task_type="odd_one_out",
    builder=lambda **kw: CVRResNetModel(backbone_name="wideresnet50", **kw),
    description="Lepori et al. CVR Odd-One-Out WideResNet-50 (InstanceNorm2d)",
)
ModelLoader.register(
    arch="lepori:cvr_vit",
    source="lepori",
    task_type="odd_one_out",
    builder=CVRViTModel,
    description="Lepori et al. CVR Odd-One-Out ViT (upstream transformers)",
    default_exclude_patterns=("embeddings",),
)
ModelLoader.register(
    arch="lepori:syneval_bert",
    source="lepori",
    task_type="odd_one_out",
    builder=SynEvalBertModel,
    description="Lepori et al. SynEval Odd-One-Out BERT-small (upstream transformers)",
    default_exclude_patterns=("embeddings",),
)
