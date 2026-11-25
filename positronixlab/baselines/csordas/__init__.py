"""Csordas et al. (ICLR 2021) — are_neural_nets_modular baseline."""

from positronixlab.baselines.csordas.datasets import (
    AddMulDataProvider,
    AddMulDataset,
    ClassRemovalDataProvider,
    TupleArithmeticDataProvider,
    TupleArithmeticDataset,
)
from positronixlab.baselines.csordas.discoverer import CsordasGumbelDiscoverer
from positronixlab.baselines.csordas.models import (
    CsordasConvModel,
    CsordasFeedforwardModel,
    CsordasLSTMModel,
    CsordasResNet110,
    CsordasTransformerEncDec,
    strip_csordas_mask_keys,
)

__all__ = [
    "AddMulDataProvider",
    "AddMulDataset",
    "ClassRemovalDataProvider",
    "CsordasConvModel",
    "CsordasFeedforwardModel",
    "CsordasGumbelDiscoverer",
    "CsordasLSTMModel",
    "CsordasResNet110",
    "CsordasTransformerEncDec",
    "TupleArithmeticDataProvider",
    "TupleArithmeticDataset",
    "strip_csordas_mask_keys",
]
