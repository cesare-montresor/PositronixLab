"""Registry-driven DataProviderFactory decoupled from ``parallelazy``."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple, Union

from positronixlab.dataprovider.config import DataConfig, DatasetName
from positronixlab.dataprovider.dataprovider import DataProvider

if TYPE_CHECKING:
    from positronixlab.pipeline.pipeline_config import PipelineConfig

# Lazy module map so built-in providers register on demand without importing
# heavyweight dependencies (sklearn, torchvision, pandas) upfront.
_BUILTIN_MODULES: Dict[str, str] = {
    "iris": "positronixlab.dataprovider.iris_dataprovider",
    "mnist": "positronixlab.dataprovider.mnist_dataprovider",
    "fashion_mnist": "positronixlab.dataprovider.fmnist_dataprovider",
    "fmnist": "positronixlab.dataprovider.fmnist_dataprovider",
    "cifar10": "positronixlab.dataprovider.cifar10_dataprovider",
    "cifar100": "positronixlab.dataprovider.cifar100_dataprovider",
    "covertype": "positronixlab.dataprovider.covertype_dataprovider",
    "higgs": "positronixlab.dataprovider.higgs_dataprovider",
    "scan": "positronixlab.dataprovider.scan_dataprovider",
    "deepmind_math": "positronixlab.dataprovider.math_dataprovider",
    "odd_one_out": "positronixlab.dataprovider.oddoneout_dataprovider",
    "cvr_odd_one_out": "positronixlab.dataprovider.cvr_oddoneout_dataprovider",
    "syneval_odd_one_out": "positronixlab.dataprovider.syneval_oddoneout_dataprovider",
    "addmul": "positronixlab.dataprovider.addmul_dataprovider",
    "csordas:add_mul": "positronixlab.dataprovider.addmul_dataprovider",
    "tuple_arithmetic": "positronixlab.dataprovider.tuple_arithmetic_dataprovider",
    "csordas:tuple_arithmetic": "positronixlab.dataprovider.tuple_arithmetic_dataprovider",
    "permuted_mnist": "positronixlab.dataprovider.permuted_mnist_dataprovider",
    "csordas:permuted_mnist": "positronixlab.dataprovider.permuted_mnist_dataprovider",
    "class_removal": "positronixlab.dataprovider.class_removal_dataprovider",
    "csordas:class_removal": "positronixlab.dataprovider.class_removal_dataprovider",
}

# Family-parameterized baseline presets routed to canonical dataprovider classes.
_FAMILY_PRESETS: Dict[str, Tuple[str, Dict[str, Any]]] = {
    "lepori:cvr_inside_contact": ("cvr_odd_one_out", {"family": "inside_contact"}),
    "lepori:cvr_inside_count": ("cvr_odd_one_out", {"family": "inside_count"}),
    "lepori:cvr_contact_count": ("cvr_odd_one_out", {"family": "contact_count"}),
    "lepori:syneval_sv_singular": ("syneval_odd_one_out", {"family": "sv_singular"}),
    "lepori:syneval_sv_plural": ("syneval_odd_one_out", {"family": "sv_plural"}),
    "lepori:syneval_pro_ant_singular": ("syneval_odd_one_out", {"family": "pro_ant_singular"}),
    "lepori:syneval_pro_ant_plural": ("syneval_odd_one_out", {"family": "pro_ant_plural"}),
}


def _resolve_provider_cls(name: str) -> type[DataProvider]:
    module_path = _BUILTIN_MODULES.get(name)
    if (
        module_path is not None
        and name not in DataProvider._registry
        and name not in DataProvider._aliases
    ):
        importlib.import_module(module_path)
    return DataProvider.get(name)


class DataProviderFactory:
    """Factory that resolves registered ``DataProvider`` subclasses from config or name."""

    @staticmethod
    def list_available() -> list[str]:
        """Return all built-in, preset, and dynamically registered dataset names."""
        names = (
            set(_BUILTIN_MODULES.keys())
            | set(_FAMILY_PRESETS.keys())
            | set(DataProvider.list_registered())
        )
        return sorted(names)

    @staticmethod
    def create(
        dataset_name: Union[DatasetName, str],
        *,
        flatten: bool = False,
        **kwargs: Any,
    ) -> DataProvider:
        """Instantiate a registered provider by *dataset_name*."""
        target_name = dataset_name
        if dataset_name in _FAMILY_PRESETS:
            target_name, preset_kwargs = _FAMILY_PRESETS[dataset_name]
            for k, v in preset_kwargs.items():
                kwargs.setdefault(k, v)

        provider_cls = _resolve_provider_cls(target_name)

        if target_name in ("permuted_mnist", "csordas:permuted_mnist"):
            kwargs.setdefault("flatten", flatten)
        elif (
            flatten
            and getattr(provider_cls, "supports_flatten", False)
            and kwargs.get("transforms") is None
        ):
            from positronixlab.metamodel.metatransform import MetaTransform

            kwargs["transforms"] = MetaTransform(["flatten"])

        return provider_cls(**kwargs)

    @staticmethod
    def from_config(
        config: Union[DataConfig, "PipelineConfig", Any],
        split: Optional[Sequence[float]] = None,
        num_workers: Optional[int] = None,
        indexed_dataset: Optional[bool] = None,
        **extra_kwargs: Any,
    ) -> DataProvider:
        """Build a ``DataProvider`` from ``DataConfig``, ``PipelineConfig``, or duck-typed config."""
        name = config.dataset_name
        batch_size = config.batch_size
        flatten = bool(
            getattr(config, "flatten", False)
            or getattr(config, "model_type", None) == "fc"
        )

        use_split = split if split is not None else getattr(config, "split", None)
        use_workers = (
            num_workers if num_workers is not None else getattr(config, "num_workers", None)
        )
        use_indexed = (
            indexed_dataset
            if indexed_dataset is not None
            else getattr(config, "indexed_dataset", False)
        )

        kwargs: Dict[str, Any] = {
            "batch_size": batch_size,
            "num_workers": use_workers,
            "indexed_dataset": use_indexed,
        }
        if use_split is not None:
            kwargs["split"] = use_split

        for opt in ("root", "seed", "shuffle_train", "drop_last", "persistent_workers", "pin_memory"):
            val = getattr(config, opt, None)
            if val is not None:
                kwargs[opt] = val

        dataset_kwargs = getattr(config, "dataset_kwargs", None)
        if isinstance(dataset_kwargs, dict):
            kwargs.update(dataset_kwargs)
        kwargs.update(extra_kwargs)

        return DataProviderFactory.create(name, flatten=flatten, **kwargs)
