"""Unified SOTA model architectures & loader (Csordas, Lepori, NeuroSurgeon)."""

from positronixlab.metamodel.sota.base import (
    MASKABLE_LAYER_TYPES,
    OddOneOutHead,
    SOTAModel,
    SOTAModelSpec,
    get_maskable_layer_names,
)
from positronixlab.metamodel.sota.csordas_models import (
    CsordasConvModel,
    CsordasFeedforwardModel,
    CsordasLSTMModel,
    CsordasResNet110,
    CsordasTransformerEncDec,
    strip_csordas_mask_keys,
)
from positronixlab.metamodel.sota.lepori_models import (
    CVRDecisionMLP,
    CVRResNetModel,
    CVRViTModel,
    SynEvalBertModel,
    strip_lepori_l0_keys,
)
from positronixlab.metamodel.sota.loader import ModelLoader
from positronixlab.metamodel.sota.neurosurgeon_adapter import (
    HFForwardShim,
    NeuroSurgeonWrappedModel,
    is_neurosurgeon_available,
    strip_neurosurgeon_mask_keys,
    wrap_with_neurosurgeon,
)

__all__ = [
    "CVRDecisionMLP",
    "CVRResNetModel",
    "CVRViTModel",
    "CsordasConvModel",
    "CsordasFeedforwardModel",
    "CsordasLSTMModel",
    "CsordasResNet110",
    "CsordasTransformerEncDec",
    "HFForwardShim",
    "MASKABLE_LAYER_TYPES",
    "ModelLoader",
    "NeuroSurgeonWrappedModel",
    "OddOneOutHead",
    "SOTAModel",
    "SOTAModelSpec",
    "SynEvalBertModel",
    "get_maskable_layer_names",
    "is_neurosurgeon_available",
    "strip_csordas_mask_keys",
    "strip_lepori_l0_keys",
    "strip_neurosurgeon_mask_keys",
    "wrap_with_neurosurgeon",
]
