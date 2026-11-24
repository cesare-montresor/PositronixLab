"""Unified SOTA compositional DataProviders (Csordas & Lepori benchmarks)."""

from positronixlab.dataprovider.sota.base import (
    CompositionalDataProvider,
    SOTADatasetSpec,
    SubtaskSplit,
)
from positronixlab.dataprovider.sota.csordas_providers import (
    AddMulDataProvider,
    AddMulDataset,
    ClassRemovalDataProvider,
    TupleArithmeticDataProvider,
    TupleArithmeticDataset,
)
from positronixlab.dataprovider.sota.lepori_providers import (
    CVR_FAMILIES,
    SYNEVAL_FAMILIES,
    CVROddOneOutDataset,
    CVROddOneOutSOTAProvider,
    SynEvalOddOneOutDataset,
    SynEvalOddOneOutSOTAProvider,
)
from positronixlab.dataprovider.sota.loader import DataProviderLoader

__all__ = [
    "AddMulDataProvider",
    "AddMulDataset",
    "CVR_FAMILIES",
    "CVROddOneOutDataset",
    "CVROddOneOutSOTAProvider",
    "ClassRemovalDataProvider",
    "CompositionalDataProvider",
    "DataProviderLoader",
    "SOTADatasetSpec",
    "SYNEVAL_FAMILIES",
    "SubtaskSplit",
    "SynEvalOddOneOutDataset",
    "SynEvalOddOneOutSOTAProvider",
    "TupleArithmeticDataProvider",
    "TupleArithmeticDataset",
]
