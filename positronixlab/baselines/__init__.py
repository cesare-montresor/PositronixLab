"""External SOTA baselines organized by work (csordas, lepori, neurosurgeon)."""

from positronixlab.baselines import csordas, lepori, neurosurgeon
from positronixlab.baselines.artifacts import (
    BaselineArtifact,
    BaselineArtifactLoader,
    CheckpointStore,
)
from positronixlab.baselines.base import (
    MASKABLE_LAYER_TYPES,
    CompositionalDataProvider,
    SOTADatasetSpec,
    SOTAModel,
    SOTAModelSpec,
    SubtaskSplit,
    get_maskable_layer_names,
)
from positronixlab.baselines.csordas import CsordasGumbelDiscoverer
from positronixlab.baselines.lepori import (
    ContinuousSparsificationDiscoverer,
    OddOneOutHead,
)
from positronixlab.baselines.loader import DataProviderLoader, ModelLoader
from positronixlab.baselines.neurosurgeon import (
    NeuroSurgeonDiscoverer,
    is_neurosurgeon_available,
    wrap_with_neurosurgeon,
)

__all__ = [
    "BaselineArtifact",
    "BaselineArtifactLoader",
    "CheckpointStore",
    "CompositionalDataProvider",
    "ContinuousSparsificationDiscoverer",
    "CsordasGumbelDiscoverer",
    "DataProviderLoader",
    "MASKABLE_LAYER_TYPES",
    "ModelLoader",
    "NeuroSurgeonDiscoverer",
    "OddOneOutHead",
    "SOTADatasetSpec",
    "SOTAModel",
    "SOTAModelSpec",
    "SubtaskSplit",
    "csordas",
    "get_maskable_layer_names",
    "is_neurosurgeon_available",
    "lepori",
    "neurosurgeon",
    "wrap_with_neurosurgeon",
]
