"""Paper-calibrated dataset references for Lepori et al. (NeurIPS 2023).

Reference: ``libs/Compositional_Subnetworks/``

Every compositional family from the paper is exposed here as a typed builder
pointing to the canonical ``positronixlab.dataprovider`` class with the
original paper defaults, alongside :data:`PAPER_CONFIGS`:

Vision (CVR Odd-One-Out, tasks 103–120, ``128x128`` real renderer):
- :func:`cvr_inside_contact`
- :func:`cvr_inside_count`
- :func:`cvr_contact_count`

Language (SynEval Syntactic Odd-One-Out, tasks 0–19, ``prajjwal1/bert-small``, ``max_length=50``):
- :func:`syneval_sv_singular`
- :func:`syneval_sv_plural`
- :func:`syneval_pro_ant_singular`
- :func:`syneval_pro_ant_plural`
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union, Unpack

from positronixlab.dataprovider.config import DataConfig
from positronixlab.dataprovider.cvr_oddoneout_dataprovider import (
    CVR_FAMILIES,
    CVROddOneOutDataProvider as CVROddOneOutSOTAProvider,
    CVROddOneOutDataset,
)
from positronixlab.dataprovider.dataprovider import DataProviderOptions
from positronixlab.dataprovider.syneval_oddoneout_dataprovider import (
    SYNEVAL_FAMILIES,
    SynevalOddOneOutDataProvider as SynEvalOddOneOutSOTAProvider,
    SynevalOddOneOutDataset as SynEvalOddOneOutDataset,
)


# ---------------------------------------------------------------------------
# Vision: CVR Odd-One-Out paper builders (Lepori et al., tasks 103..120)
# ---------------------------------------------------------------------------


def cvr_inside_contact(
    *,
    n_train: int = 10_000,
    n_val: int = 1_000,
    n_test: int = 1_000,
    image_size: int = 128,
    batch_size: int = 64,
    use_real_generator: bool = True,
    root: Optional[Union[str, Path]] = None,
    seed: int = 0,
    **kwargs: Unpack[DataProviderOptions],
) -> CVROddOneOutSOTAProvider:
    """Lepori et al. CVR ``inside_contact`` family (tasks 103–108, ``128x128``)."""
    return CVROddOneOutSOTAProvider(
        family="inside_contact",
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        image_size=image_size,
        batch_size=batch_size,
        use_real_generator=use_real_generator,
        root=root,
        seed=seed,
        **kwargs,
    )


def cvr_inside_count(
    *,
    n_train: int = 10_000,
    n_val: int = 1_000,
    n_test: int = 1_000,
    image_size: int = 128,
    batch_size: int = 64,
    use_real_generator: bool = True,
    root: Optional[Union[str, Path]] = None,
    seed: int = 0,
    **kwargs: Unpack[DataProviderOptions],
) -> CVROddOneOutSOTAProvider:
    """Lepori et al. CVR ``inside_count`` family (tasks 109–114, ``128x128``)."""
    return CVROddOneOutSOTAProvider(
        family="inside_count",
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        image_size=image_size,
        batch_size=batch_size,
        use_real_generator=use_real_generator,
        root=root,
        seed=seed,
        **kwargs,
    )


def cvr_contact_count(
    *,
    n_train: int = 10_000,
    n_val: int = 1_000,
    n_test: int = 1_000,
    image_size: int = 128,
    batch_size: int = 64,
    use_real_generator: bool = True,
    root: Optional[Union[str, Path]] = None,
    seed: int = 0,
    **kwargs: Unpack[DataProviderOptions],
) -> CVROddOneOutSOTAProvider:
    """Lepori et al. CVR ``contact_count`` family (tasks 115–120, ``128x128``)."""
    return CVROddOneOutSOTAProvider(
        family="contact_count",
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        image_size=image_size,
        batch_size=batch_size,
        use_real_generator=use_real_generator,
        root=root,
        seed=seed,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Language: SynEval Syntactic Odd-One-Out paper builders (tasks 0..19)
# ---------------------------------------------------------------------------


def syneval_sv_singular(
    *,
    n_train: int = 10_000,
    n_val: int = 1_000,
    n_test: int = 1_000,
    max_length: int = 50,
    batch_size: int = 32,
    tokenizer_name: str = "prajjwal1/bert-small",
    root: Optional[Union[str, Path]] = None,
    seed: int = 0,
    **kwargs: Unpack[DataProviderOptions],
) -> SynEvalOddOneOutSOTAProvider:
    """Lepori et al. SynEval ``sv_singular`` family (``max_length=50``, ``bert-small``)."""
    return SynEvalOddOneOutSOTAProvider(
        family="sv_singular",
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        max_length=max_length,
        batch_size=batch_size,
        tokenizer_name=tokenizer_name,
        root=root,
        seed=seed,
        **kwargs,
    )


def syneval_sv_plural(
    *,
    n_train: int = 10_000,
    n_val: int = 1_000,
    n_test: int = 1_000,
    max_length: int = 50,
    batch_size: int = 32,
    tokenizer_name: str = "prajjwal1/bert-small",
    root: Optional[Union[str, Path]] = None,
    seed: int = 0,
    **kwargs: Unpack[DataProviderOptions],
) -> SynEvalOddOneOutSOTAProvider:
    """Lepori et al. SynEval ``sv_plural`` family (``max_length=50``, ``bert-small``)."""
    return SynEvalOddOneOutSOTAProvider(
        family="sv_plural",
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        max_length=max_length,
        batch_size=batch_size,
        tokenizer_name=tokenizer_name,
        root=root,
        seed=seed,
        **kwargs,
    )


def syneval_pro_ant_singular(
    *,
    n_train: int = 10_000,
    n_val: int = 1_000,
    n_test: int = 1_000,
    max_length: int = 50,
    batch_size: int = 32,
    tokenizer_name: str = "prajjwal1/bert-small",
    root: Optional[Union[str, Path]] = None,
    seed: int = 0,
    **kwargs: Unpack[DataProviderOptions],
) -> SynEvalOddOneOutSOTAProvider:
    """Lepori et al. SynEval ``pro_ant_singular`` family (``max_length=50``, ``bert-small``)."""
    return SynEvalOddOneOutSOTAProvider(
        family="pro_ant_singular",
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        max_length=max_length,
        batch_size=batch_size,
        tokenizer_name=tokenizer_name,
        root=root,
        seed=seed,
        **kwargs,
    )


def syneval_pro_ant_plural(
    *,
    n_train: int = 10_000,
    n_val: int = 1_000,
    n_test: int = 1_000,
    max_length: int = 50,
    batch_size: int = 32,
    tokenizer_name: str = "prajjwal1/bert-small",
    root: Optional[Union[str, Path]] = None,
    seed: int = 0,
    **kwargs: Unpack[DataProviderOptions],
) -> SynEvalOddOneOutSOTAProvider:
    """Lepori et al. SynEval ``pro_ant_plural`` family (``max_length=50``, ``bert-small``)."""
    return SynEvalOddOneOutSOTAProvider(
        family="pro_ant_plural",
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        max_length=max_length,
        batch_size=batch_size,
        tokenizer_name=tokenizer_name,
        root=root,
        seed=seed,
        **kwargs,
    )


PAPER_CONFIGS: Dict[str, DataConfig] = {
    **{
        f"cvr_{fam}": DataConfig(
            name=f"lepori_cvr_{fam}",
            dataset_name=f"lepori:cvr_{fam}",
            batch_size=64,
            seed=0,
            dataset_kwargs={
                "n_train": 10_000,
                "n_val": 1_000,
                "n_test": 1_000,
                "image_size": 128,
                "use_real_generator": True,
            },
        )
        for fam in ("inside_contact", "inside_count", "contact_count")
    },
    **{
        f"syneval_{fam}": DataConfig(
            name=f"lepori_syneval_{fam}",
            dataset_name=f"lepori:syneval_{fam}",
            batch_size=32,
            seed=0,
            dataset_kwargs={
                "n_train": 10_000,
                "n_val": 1_000,
                "n_test": 1_000,
                "max_length": 50,
                "tokenizer_name": "prajjwal1/bert-small",
            },
        )
        for fam in ("sv_singular", "sv_plural", "pro_ant_singular", "pro_ant_plural")
    },
}


__all__ = [
    "CVR_FAMILIES",
    "CVROddOneOutDataset",
    "CVROddOneOutSOTAProvider",
    "PAPER_CONFIGS",
    "SYNEVAL_FAMILIES",
    "SynEvalOddOneOutDataset",
    "SynEvalOddOneOutSOTAProvider",
    "cvr_contact_count",
    "cvr_inside_contact",
    "cvr_inside_count",
    "syneval_pro_ant_plural",
    "syneval_pro_ant_singular",
    "syneval_sv_plural",
    "syneval_sv_singular",
]
