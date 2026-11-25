"""Lepori et al. (NeurIPS 2023) — Compositional_Subnetworks baseline."""

from positronixlab.baselines.lepori.datasets import (
    CVR_FAMILIES,
    SYNEVAL_FAMILIES,
    CVROddOneOutDataset,
    CVROddOneOutSOTAProvider,
    SynEvalOddOneOutDataset,
    SynEvalOddOneOutSOTAProvider,
)
from positronixlab.baselines.lepori.models import (
    CVRDecisionMLP,
    CVRResNetModel,
    CVRViTModel,
    OddOneOutHead,
    SynEvalBertModel,
    strip_lepori_l0_keys,
)

__all__ = [
    "CVRDecisionMLP",
    "CVRResNetModel",
    "CVRViTModel",
    "CVR_FAMILIES",
    "CVROddOneOutDataset",
    "CVROddOneOutSOTAProvider",
    "OddOneOutHead",
    "SYNEVAL_FAMILIES",
    "SynEvalBertModel",
    "SynEvalOddOneOutDataset",
    "SynEvalOddOneOutSOTAProvider",
    "strip_lepori_l0_keys",
]
