"""Unified registry-driven SOTA DataProviderLoader (symmetric with ModelLoader)."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from positronixlab.dataprovider.sota.base import CompositionalDataProvider, SOTADatasetSpec
from positronixlab.dataprovider.sota.csordas_providers import (
    AddMulDataProvider,
    ClassRemovalDataProvider,
    TupleArithmeticDataProvider,
)
from positronixlab.dataprovider.sota.lepori_providers import (
    CVROddOneOutSOTAProvider,
    SynEvalOddOneOutSOTAProvider,
)


class DataProviderLoader:
    """Unified factory for SOTA modularity benchmark datasets (Csordas & Lepori)."""

    _REGISTRY: Dict[str, SOTADatasetSpec] = {}

    @classmethod
    def register(
        cls,
        name: str,
        source: str,
        task_type: str,
        builder: Callable[..., CompositionalDataProvider],
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
    def list_datasets(cls, source: Optional[str] = None) -> List[str]:
        if source is None:
            return sorted(cls._REGISTRY.keys())
        return sorted(k for k, v in cls._REGISTRY.items() if v.source == source)

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> CompositionalDataProvider:
        if name not in cls._REGISTRY:
            raise ValueError(
                f"Unknown SOTA dataset {name!r}. Available: {cls.list_datasets()}"
            )
        return cls._REGISTRY[name].builder(**kwargs)


# ---------------------------------------------------------------------------
# Default Registrations
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

for _fam in ("inside_contact", "inside_count", "contact_count"):
    DataProviderLoader.register(
        name=f"lepori:cvr_{_fam}",
        source="lepori",
        task_type="odd_one_out",
        builder=lambda _f=_fam, **kw: CVROddOneOutSOTAProvider(family=_f, **kw),
        description=f"Lepori et al. CVR Odd-One-Out compositional family ({_fam})",
    )

for _fam in ("sv_singular", "sv_plural", "pro_ant_singular", "pro_ant_plural"):
    DataProviderLoader.register(
        name=f"lepori:syneval_{_fam}",
        source="lepori",
        task_type="odd_one_out",
        builder=lambda _f=_fam, **kw: SynEvalOddOneOutSOTAProvider(family=_f, **kw),
        description=f"Lepori et al. SynEval syntactic Odd-One-Out family ({_fam})",
    )
