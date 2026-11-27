"""Unified registries: :class:`ModelLoader` and :class:`DataProviderLoader`.

Provides a single entry point to create or load models and compositional
benchmarks across all baseline works (``csordas``, ``lepori``, ``neurosurgeon``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from positronixlab.baselines.base import (
    DataProvider,
    SOTADatasetSpec,
    SOTAModel,
    SOTAModelSpec,
)
from positronixlab.baselines.csordas.datasets import (
    AddMulDataProvider,
    ClassRemovalDataProvider,
    MathDataProvider,
    PermutedMnistDataProvider,
    SCANDataProvider,
    TupleArithmeticDataProvider,
)
from positronixlab.baselines.csordas.models import (
    CsordasConvModel,
    CsordasFeedforwardModel,
    CsordasLSTMModel,
    CsordasResNet110,
    CsordasTransformerEncDec,
    strip_csordas_mask_keys,
)
from positronixlab.baselines.lepori.datasets import (
    CVROddOneOutSOTAProvider,
    CVRSimCLRDataProvider,
    SynEvalOddOneOutSOTAProvider,
)
from positronixlab.baselines.lepori.models import (
    CVRResNetModel,
    CVRSimCLRModel,
    CVRViTModel,
    SynEvalBertModel,
    strip_lepori_l0_keys,
)
from positronixlab.baselines.neurosurgeon.adapter import (
    NeuroSurgeonWrappedModel,
    strip_neurosurgeon_mask_keys,
    wrap_with_neurosurgeon,
)
from positronixlab.utils.utils import load_data, save_data


def _detect_checkpoint_format(state_dict: dict[str, Any]) -> str:
    """Detect whether a checkpoint came from Csordas, Lepori, NeuroSurgeon, or plain PyTorch."""
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
    """Unified factory & checkpoint loader for baseline modularity architectures."""

    _REGISTRY: dict[str, SOTAModelSpec] = {}

    @classmethod
    def register(
        cls,
        arch: str,
        source: str,
        task_type: str,
        builder: Callable[..., SOTAModel],
        description: str = "",
        default_exclude_patterns: tuple[str, ...] = (),
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
    def list_models(cls, source: str | None = None) -> list[str]:
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
        device: torch.device | str | None = None,
        **kwargs: Any,
    ) -> SOTAModel:
        return cls.get_spec(arch).builder(device=device, **kwargs)

    @classmethod
    def save(
        cls,
        model: nn.Module,
        path: str | Path,
        *,
        arch: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
        dataset_name: str | None = None,
        metrics: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save a self-describing checkpoint bundle for later reload via :meth:`load_bundle`."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_arch = arch or getattr(model, "arch", None) or getattr(model, "sota_arch", None)
        bundle: dict[str, Any] = {
            "format": "positronix_sota_bundle_v1",
            "arch": resolved_arch,
            "model_kwargs": dict(model_kwargs or {}),
            "dataset_name": dataset_name,
            "metrics": dict(metrics or {}),
            "metadata": dict(metadata or {}),
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        }
        save_data(str(out_path), bundle)
        return out_path

    @classmethod
    def load_bundle(
        cls,
        path: str | Path,
        *,
        device: torch.device | str | None = None,
        strict: bool = True,
        **override_model_kwargs: Any,
    ) -> tuple[SOTAModel, dict[str, Any]]:
        """Load a self-describing bundle saved by :meth:`save`.

        Returns ``(model, info)`` where ``info`` contains ``arch``, ``model_kwargs``,
        ``dataset_name``, ``metrics``, and ``metadata``.
        """
        bundle = load_data(str(path), map_location="cpu", weights_only=False)
        if not isinstance(bundle, dict) or bundle.get("format") != "positronix_sota_bundle_v1":
            raise ValueError(
                f"File {path} is not a positronix_sota_bundle_v1 checkpoint. "
                "Use ModelLoader.load(arch, checkpoint=path) for raw state_dicts."
            )
        arch = bundle.get("arch")
        if not arch:
            raise ValueError(f"Bundle {path} is missing required 'arch' field.")
        kwargs = dict(bundle.get("model_kwargs") or {})
        kwargs.update(override_model_kwargs)
        model = cls.create(arch, device=device, **kwargs)
        model.load_state_dict(bundle["state_dict"], strict=strict)
        info = {
            "arch": arch,
            "model_kwargs": kwargs,
            "dataset_name": bundle.get("dataset_name"),
            "metrics": dict(bundle.get("metrics") or {}),
            "metadata": dict(bundle.get("metadata") or {}),
        }
        return model, info

    @classmethod
    def load(
        cls,
        arch: str,
        checkpoint: str | Path | dict[str, Any] | None = None,
        backbone_checkpoint: str | Path | None = None,
        mlp_checkpoint: str | Path | None = None,
        device: torch.device | str | None = None,
        strict: bool = False,
        return_mask: bool = False,
        **model_kwargs: Any,
    ) -> Any:
        """Create a model and optionally load weights (auto-detecting checkpoint format)."""
        model = cls.create(arch, device=device, **model_kwargs)
        extracted_scores: dict[str, torch.Tensor] = {}
        ckpt_format = "plain"

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
                else load_data(str(checkpoint), map_location=model.device, weights_only=True)
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

        from positronixlab.tasks.masks import SubnetworkMask

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
    def infer_kwargs(
        cls,
        arch: str,
        dp: Any,
        model_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Infer architecture I/O dimensions from ``dp`` without overriding explicit kwargs.

        Currently populates defaults for ``csordas:mlp``, ``csordas:conv``,
        ``csordas:resnet110``, and ``csordas:lstm``; other architectures pass
        ``model_kwargs`` through unchanged.
        """
        kwargs = dict(model_kwargs or {})
        if arch == "csordas:mlp":
            kwargs.setdefault("n_inputs", dp.num_features)
            kwargs.setdefault("n_outputs", dp.num_classes)
            kwargs.setdefault("layer_sizes", [128, 64])
        elif arch == "csordas:conv":
            kwargs.setdefault("out_channels", dp.num_classes)
        elif arch == "csordas:resnet110":
            kwargs.setdefault("num_classes", dp.num_classes)
        elif arch == "csordas:lstm":
            kwargs.setdefault("n_inputs", dp.num_features)
            kwargs.setdefault("n_outputs", dp.num_classes)
        return kwargs

    @classmethod
    def wrap_neurosurgeon(
        cls,
        model: nn.Module,
        target_layers: list[str] | None = None,
        mask_method: str = "continuous_sparsification",
        mask_unit: str = "weight",
        mask_hparams: dict[str, Any] | None = None,
        freeze_base: bool = True,
        l0_lambda: float = 1e-8,
        deepcopy_model: bool = True,
    ) -> NeuroSurgeonWrappedModel:
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


class DataProviderLoader:
    """Unified factory for baseline compositional benchmark datasets."""

    _REGISTRY: dict[str, SOTADatasetSpec] = {}

    @classmethod
    def register(
        cls,
        name: str,
        source: str,
        task_type: str,
        builder: Callable[..., DataProvider],
        description: str = "",
    ) -> None:
        cls._REGISTRY[name] = SOTADatasetSpec(
            name=name,
            source=source,
            task_type=task_type,
            builder=builder,
            description=description,
        )

    @classmethod
    def list_datasets(cls, source: str | None = None) -> list[str]:
        if source is None:
            return sorted(cls._REGISTRY.keys())
        return sorted(k for k, v in cls._REGISTRY.items() if v.source == source)

    @classmethod
    def create(
        cls,
        name: str,
        *,
        base_dataset_name: str | None = None,
        base_dataset_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> DataProvider:
        if name == "csordas:class_removal" and base_dataset_name:
            from positronixlab.dataprovider.dataprovider_factory import DataProviderFactory

            base_kw = dict(base_dataset_kwargs or {})
            if "batch_size" in kwargs:
                base_kw.setdefault("batch_size", kwargs["batch_size"])
            base_dp = DataProviderFactory.create(base_dataset_name, **base_kw)
            wrap_kw = {k: v for k, v in kwargs.items() if k != "seed"}
            wrap_kw.setdefault("name", f"class_removal:{base_dataset_name}")
            return ClassRemovalDataProvider.from_provider(base_dp, **wrap_kw)

        if name not in cls._REGISTRY:
            raise ValueError(
                f"Unknown baseline dataset {name!r}. Available: {cls.list_datasets()}"
            )
        return cls._REGISTRY[name].builder(**kwargs)

    @classmethod
    def resolve_subtasks(
        cls,
        dp: Any,
        subtasks: Sequence[str] | None = None,
    ) -> list[str]:
        """Return explicit ``subtasks`` if given, else non-full subtasks from ``dp``."""
        if subtasks:
            return list(subtasks)
        names = [name for name in dp.list_subtasks() if name not in ("full", "all")]
        return names if names else list(dp.list_subtasks())


# ---------------------------------------------------------------------------
# Default Model Registrations
# ---------------------------------------------------------------------------

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
    arch="lepori:cvr_simclr_resnet18",
    source="lepori",
    task_type="contrastive",
    builder=lambda **kw: CVRSimCLRModel(backbone_name="resnet18", **kw),
    description="Lepori et al. CVR SimCLR Pretraining ResNet-18 (InstanceNorm2d + NT-Xent)",
)
ModelLoader.register(
    arch="lepori:cvr_simclr_resnet50",
    source="lepori",
    task_type="contrastive",
    builder=lambda **kw: CVRSimCLRModel(backbone_name="resnet50", **kw),
    description="Lepori et al. CVR SimCLR Pretraining ResNet-50 (InstanceNorm2d + NT-Xent)",
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

# ---------------------------------------------------------------------------
# Default Dataset Registrations
# ---------------------------------------------------------------------------

DataProviderLoader.register(
    name="csordas:tuple_arithmetic",
    source="csordas",
    task_type="arithmetic",
    builder=TupleArithmeticDataProvider,
    description="Csordas et al. Double Addition / Multiplication (synthetic, zero download)",
)
DataProviderLoader.register(
    name="csordas:add_mul",
    source="csordas",
    task_type="arithmetic",
    builder=AddMulDataProvider,
    description="Csordas et al. Add vs. Mul (synthetic, zero download)",
)
DataProviderLoader.register(
    name="csordas:class_removal",
    source="csordas",
    task_type="classification",
    builder=ClassRemovalDataProvider,
    description="Csordas et al. Leave-One-Class-Out wrapper",
)
DataProviderLoader.register(
    name="csordas:deepmind_math",
    source="csordas",
    task_type="seq2seq",
    builder=MathDataProvider,
    description="Csordas et al. DeepMind Mathematics v1.0 (HF auto-download + per-difficulty/module subtasks)",
)
DataProviderLoader.register(
    name="csordas:scan",
    source="csordas",
    task_type="seq2seq",
    builder=SCANDataProvider,
    description="Csordas et al. SCAN compositional sequence-to-sequence benchmark (GitHub auto-download)",
)
DataProviderLoader.register(
    name="csordas:permuted_mnist",
    source="csordas",
    task_type="classification",
    builder=PermutedMnistDataProvider,
    description="Csordas et al. Permuted MNIST transfer/reuse benchmark (permutations 0..10)",
)

for _fam in ("inside_contact", "inside_count", "contact_count"):
    DataProviderLoader.register(
        name=f"lepori:cvr_{_fam}",
        source="lepori",
        task_type="odd_one_out",
        builder=lambda _f=_fam, **kw: CVROddOneOutSOTAProvider(family=_f, **kw),
        description=f"Lepori et al. CVR Odd-One-Out compositional family ({_fam})",
    )

DataProviderLoader.register(
    name="lepori:cvr_simclr",
    source="lepori",
    task_type="contrastive",
    builder=CVRSimCLRDataProvider,
    description="Lepori et al. CVR SimCLR self-supervised contrastive pretraining (base tasks 103, 109, 115)",
)

for _fam in ("sv_singular", "sv_plural", "pro_ant_singular", "pro_ant_plural"):
    DataProviderLoader.register(
        name=f"lepori:syneval_{_fam}",
        source="lepori",
        task_type="odd_one_out",
        builder=lambda _f=_fam, **kw: SynEvalOddOneOutSOTAProvider(family=_f, **kw),
        description=f"Lepori et al. SynEval syntactic Odd-One-Out family ({_fam})",
    )
