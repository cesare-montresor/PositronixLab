"""Unified SOTA subnetwork mask contract, ablation engine, and discoverers."""

from positronixlab.tasks.sota.atlas_discoverer import AtlasDiscoverer
from positronixlab.tasks.sota.benchmarks import (
    BenchmarkReport,
    ModularityBenchmark,
    SubnetworkEvalResult,
    default_eval_accuracy,
)
from positronixlab.tasks.sota.discoverers import (
    CsordasGumbelDiscoverer,
    NeuroSurgeonDiscoverer,
    RandomMaskDiscoverer,
    SubnetworkDiscoverer,
)
from positronixlab.tasks.sota.masks import (
    AblationMode,
    Granularity,
    MaskApplier,
    SubnetworkMask,
)

__all__ = [
    "AblationMode",
    "AtlasDiscoverer",
    "BenchmarkReport",
    "CsordasGumbelDiscoverer",
    "Granularity",
    "MaskApplier",
    "ModularityBenchmark",
    "NeuroSurgeonDiscoverer",
    "RandomMaskDiscoverer",
    "SubnetworkDiscoverer",
    "SubnetworkEvalResult",
    "SubnetworkMask",
    "default_eval_accuracy",
]
