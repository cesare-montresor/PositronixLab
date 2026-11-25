"""Compositional Odd-One-Out datasets from Lepori et al. (NeurIPS 2023).

Re-exports the canonical implementations from ``positronixlab.dataprovider``.
"""

from __future__ import annotations

from positronixlab.dataprovider.cvr_oddoneout_dataprovider import (
    CVR_FAMILIES,
    CVROddOneOutDataProvider as CVROddOneOutSOTAProvider,
    CVROddOneOutDataset,
)
from positronixlab.dataprovider.syneval_oddoneout_dataprovider import (
    SYNEVAL_FAMILIES,
    SynevalOddOneOutDataProvider as SynEvalOddOneOutSOTAProvider,
    SynevalOddOneOutDataset as SynEvalOddOneOutDataset,
)

__all__ = [
    "CVR_FAMILIES",
    "CVROddOneOutDataset",
    "CVROddOneOutSOTAProvider",
    "SYNEVAL_FAMILIES",
    "SynEvalOddOneOutDataset",
    "SynEvalOddOneOutSOTAProvider",
]
