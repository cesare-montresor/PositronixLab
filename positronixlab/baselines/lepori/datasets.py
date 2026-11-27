"""Paper-calibrated dataset references for Lepori et al. (NeurIPS 2023).

Reference: ``libs/Compositional_Subnetworks/``

Every compositional family and pretraining dataset from the paper is exposed
here as a typed builder function pointing to the canonical
``positronixlab.dataprovider`` class with the original paper defaults,
alongside :data:`PAPER_CONFIGS`:

Vision (CVR Odd-One-Out, tasks 103–120, ``128x128`` real renderer, ``10k/500/1k``):
- :func:`cvr_inside_contact`
- :func:`cvr_inside_count`
- :func:`cvr_contact_count`
- :func:`cvr_simclr` (SimCLR self-supervised pretraining on base tasks 103, 109, 115)

Language (SynEval Syntactic Odd-One-Out, tasks 0–19, ``prajjwal1/bert-small``, ``max_length=50``):
- :func:`syneval_sv_singular` (``9500/500/1000``)
- :func:`syneval_sv_plural` (``9500/500/1000``)
- :func:`syneval_pro_ant_singular` (``2500/200/200``)
- :func:`syneval_pro_ant_plural` (``2500/200/200``)
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from positronixlab.baselines.lepori._syneval_gen import (
    SYNEVAL_PAPER_SIZES,
    ensure_syneval_family_splits,
    generate_syneval_family_splits,
)
from positronixlab.dataprovider.config import DataConfig
from positronixlab.dataprovider.cvr_oddoneout_dataprovider import (
    CVR_FAMILIES,
    CVROddOneOutDataset,
    load_cvr_png_strip,
)
from positronixlab.dataprovider.cvr_oddoneout_dataprovider import (
    CVROddOneOutDataProvider as CVROddOneOutSOTAProvider,
)
from positronixlab.dataprovider.cvr_simclr_dataprovider import (
    CVR_SIMCLR_BASE_TASKS,
    ContrastiveTransformations,
    CVRSimCLRDataProvider,
    CVRSimCLRDataset,
    build_simclr_transforms,
)
from positronixlab.dataprovider.syneval_oddoneout_dataprovider import (
    SYNEVAL_FAMILIES,
)
from positronixlab.dataprovider.syneval_oddoneout_dataprovider import (
    SynevalOddOneOutDataProvider as SynEvalOddOneOutSOTAProvider,
)
from positronixlab.dataprovider.syneval_oddoneout_dataprovider import (
    SynevalOddOneOutDataset as SynEvalOddOneOutDataset,
)

# ---------------------------------------------------------------------------
# Vision: CVR Odd-One-Out & SimCLR paper builders (Lepori et al., tasks 103..120)
# ---------------------------------------------------------------------------


def cvr_inside_contact(
    *,
    n_train: int = 10_000,
    n_val: int = 500,
    n_test: int = 1_000,
    image_size: int = 128,
    batch_size: int = 64,
    use_real_generator: bool = True,
    root: str | Path | None = None,
    seed: int = 0,
    **kwargs: Any,
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
    n_val: int = 500,
    n_test: int = 1_000,
    image_size: int = 128,
    batch_size: int = 64,
    use_real_generator: bool = True,
    root: str | Path | None = None,
    seed: int = 0,
    **kwargs: Any,
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
    n_val: int = 500,
    n_test: int = 1_000,
    image_size: int = 128,
    batch_size: int = 64,
    use_real_generator: bool = True,
    root: str | Path | None = None,
    seed: int = 0,
    **kwargs: Any,
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


def cvr_simclr(
    *,
    n_train_per_task: int = 10_000,
    n_val_per_task: int = 500,
    image_size: int = 128,
    batch_size: int = 128,
    n_views: int = 2,
    as_list: bool = False,
    use_real_generator: bool = True,
    root: str | Path | None = None,
    seed: int = 0,
    **kwargs: Any,
) -> CVRSimCLRDataProvider:
    """Lepori et al. CVR SimCLR pretraining dataset (base tasks 103, 109, 115, ``128x128``)."""
    return CVRSimCLRDataProvider(
        n_train_per_task=n_train_per_task,
        n_val_per_task=n_val_per_task,
        image_size=image_size,
        batch_size=batch_size,
        n_views=n_views,
        as_list=as_list,
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
    n_train: int = 9_500,
    n_val: int = 500,
    n_test: int = 1_000,
    max_length: int = 50,
    batch_size: int = 32,
    tokenizer_name: str = "prajjwal1/bert-small",
    tokenizer: Callable[..., dict[str, Any]] | None = None,
    root: str | Path | None = None,
    raw_data_path: str | Path | None = None,
    download: bool = True,
    seed: int = 0,
    **kwargs: Any,
) -> SynEvalOddOneOutSOTAProvider:
    """Lepori et al. SynEval ``sv_singular`` family (``9500/500/1000``, ``max_length=50``, ``bert-small``)."""
    return SynEvalOddOneOutSOTAProvider(
        family="sv_singular",
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        max_length=max_length,
        batch_size=batch_size,
        tokenizer_name=tokenizer_name,
        tokenizer=tokenizer,
        root=root,
        raw_data_path=raw_data_path,
        download=download,
        seed=seed,
        **kwargs,
    )


def syneval_sv_plural(
    *,
    n_train: int = 9_500,
    n_val: int = 500,
    n_test: int = 1_000,
    max_length: int = 50,
    batch_size: int = 32,
    tokenizer_name: str = "prajjwal1/bert-small",
    tokenizer: Callable[..., dict[str, Any]] | None = None,
    root: str | Path | None = None,
    raw_data_path: str | Path | None = None,
    download: bool = True,
    seed: int = 0,
    **kwargs: Any,
) -> SynEvalOddOneOutSOTAProvider:
    """Lepori et al. SynEval ``sv_plural`` family (``9500/500/1000``, ``max_length=50``, ``bert-small``)."""
    return SynEvalOddOneOutSOTAProvider(
        family="sv_plural",
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        max_length=max_length,
        batch_size=batch_size,
        tokenizer_name=tokenizer_name,
        tokenizer=tokenizer,
        root=root,
        raw_data_path=raw_data_path,
        download=download,
        seed=seed,
        **kwargs,
    )


def syneval_pro_ant_singular(
    *,
    n_train: int = 2_500,
    n_val: int = 200,
    n_test: int = 200,
    max_length: int = 50,
    batch_size: int = 32,
    tokenizer_name: str = "prajjwal1/bert-small",
    tokenizer: Callable[..., dict[str, Any]] | None = None,
    root: str | Path | None = None,
    raw_data_path: str | Path | None = None,
    download: bool = True,
    seed: int = 0,
    **kwargs: Any,
) -> SynEvalOddOneOutSOTAProvider:
    """Lepori et al. SynEval ``pro_ant_singular`` family (``2500/200/200``, ``max_length=50``, ``bert-small``)."""
    return SynEvalOddOneOutSOTAProvider(
        family="pro_ant_singular",
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        max_length=max_length,
        batch_size=batch_size,
        tokenizer_name=tokenizer_name,
        tokenizer=tokenizer,
        root=root,
        raw_data_path=raw_data_path,
        download=download,
        seed=seed,
        **kwargs,
    )


def syneval_pro_ant_plural(
    *,
    n_train: int = 2_500,
    n_val: int = 200,
    n_test: int = 200,
    max_length: int = 50,
    batch_size: int = 32,
    tokenizer_name: str = "prajjwal1/bert-small",
    tokenizer: Callable[..., dict[str, Any]] | None = None,
    root: str | Path | None = None,
    raw_data_path: str | Path | None = None,
    download: bool = True,
    seed: int = 0,
    **kwargs: Any,
) -> SynEvalOddOneOutSOTAProvider:
    """Lepori et al. SynEval ``pro_ant_plural`` family (``2500/200/200``, ``max_length=50``, ``bert-small``)."""
    return SynEvalOddOneOutSOTAProvider(
        family="pro_ant_plural",
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        max_length=max_length,
        batch_size=batch_size,
        tokenizer_name=tokenizer_name,
        tokenizer=tokenizer,
        root=root,
        raw_data_path=raw_data_path,
        download=download,
        seed=seed,
        **kwargs,
    )


PAPER_CONFIGS: dict[str, DataConfig] = {
    **{
        f"cvr_{fam}": DataConfig(
            name=f"lepori_cvr_{fam}",
            dataset_name=f"lepori:cvr_{fam}",
            batch_size=64,
            seed=0,
            dataset_kwargs={
                "n_train": 10_000,
                "n_val": 500,
                "n_test": 1_000,
                "image_size": 128,
                "use_real_generator": True,
            },
        )
        for fam in ("inside_contact", "inside_count", "contact_count")
    },
    "cvr_simclr": DataConfig(
        name="lepori_cvr_simclr",
        dataset_name="lepori:cvr_simclr",
        batch_size=128,
        seed=0,
        dataset_kwargs={
            "n_train_per_task": 10_000,
            "n_val_per_task": 500,
            "image_size": 128,
            "n_views": 2,
            "use_real_generator": True,
        },
    ),
    **{
        f"syneval_{fam}": DataConfig(
            name=f"lepori_syneval_{fam}",
            dataset_name=f"lepori:syneval_{fam}",
            batch_size=32,
            seed=0,
            dataset_kwargs={
                "n_train": SYNEVAL_PAPER_SIZES[fam]["train"],
                "n_val": SYNEVAL_PAPER_SIZES[fam]["val"],
                "n_test": SYNEVAL_PAPER_SIZES[fam]["test"],
                "max_length": 50,
                "tokenizer_name": "prajjwal1/bert-small",
                "download": True,
            },
        )
        for fam in ("sv_singular", "sv_plural", "pro_ant_singular", "pro_ant_plural")
    },
}


__all__ = [
    "CVR_FAMILIES",
    "CVR_SIMCLR_BASE_TASKS",
    "CVROddOneOutDataset",
    "CVROddOneOutSOTAProvider",
    "CVRSimCLRDataProvider",
    "CVRSimCLRDataset",
    "ContrastiveTransformations",
    "PAPER_CONFIGS",
    "SYNEVAL_FAMILIES",
    "SYNEVAL_PAPER_SIZES",
    "SynEvalOddOneOutDataset",
    "SynEvalOddOneOutSOTAProvider",
    "build_simclr_transforms",
    "cvr_contact_count",
    "cvr_inside_contact",
    "cvr_inside_count",
    "cvr_simclr",
    "ensure_syneval_family_splits",
    "generate_syneval_family_splits",
    "load_cvr_png_strip",
    "syneval_pro_ant_plural",
    "syneval_pro_ant_singular",
    "syneval_sv_plural",
    "syneval_sv_singular",
]
