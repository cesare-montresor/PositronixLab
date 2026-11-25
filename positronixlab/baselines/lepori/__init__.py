"""Lepori et al. (NeurIPS 2023) — Compositional_Subnetworks baseline."""

from positronixlab.baselines.lepori.datasets import (
    CVR_FAMILIES,
    PAPER_CONFIGS,
    SYNEVAL_FAMILIES,
    CVROddOneOutDataset,
    CVROddOneOutSOTAProvider,
    SynEvalOddOneOutDataset,
    SynEvalOddOneOutSOTAProvider,
    cvr_contact_count,
    cvr_inside_contact,
    cvr_inside_count,
    syneval_pro_ant_plural,
    syneval_pro_ant_singular,
    syneval_sv_plural,
    syneval_sv_singular,
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
    "PAPER_CONFIGS",
    "SYNEVAL_FAMILIES",
    "SynEvalBertModel",
    "SynEvalOddOneOutDataset",
    "SynEvalOddOneOutSOTAProvider",
    "cvr_contact_count",
    "cvr_inside_contact",
    "cvr_inside_count",
    "strip_lepori_l0_keys",
    "syneval_pro_ant_plural",
    "syneval_pro_ant_singular",
    "syneval_sv_plural",
    "syneval_sv_singular",
]
