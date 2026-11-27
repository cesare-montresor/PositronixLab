"""Leave-one-class-out compositional wrapper (Csordas et al., ICLR 2021).

Wraps any classification ``DataProvider`` or ``(dataset_train, dataset_val, dataset_test)``
and creates subtasks ``"all"`` and ``"without_0" .. "without_{C-1}"`` plus the
corresponding evaluation matrix specification.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import Dataset, Subset

from positronixlab.dataprovider.dataprovider import (
    CompositionalDataProvider,
    DataProvider,
    SubtaskSplit,
)


def _extract_labels(ds: Dataset) -> list[int]:
    if hasattr(ds, "targets"):
        t = ds.targets  # type: ignore[attr-defined]
        return [int(v) for v in (t.tolist() if isinstance(t, torch.Tensor) else t)]
    if hasattr(ds, "y"):
        y = ds.y  # type: ignore[attr-defined]
        return [int(v) for v in (y.tolist() if isinstance(y, torch.Tensor) else y)]
    return [int(ds[i][1]) for i in range(len(ds))]  # type: ignore[arg-type]


class ClassRemovalDataProvider(
    CompositionalDataProvider,
    plugin_name="class_removal",
    aliases=("csordas:class_removal",),
):
    """Leave-one-class-out wrapper matching ``Cifar10ClassRemovalTask`` (Csordas et al.)."""

    source = "csordas"
    task_type = "classification"

    def __init__(
        self,
        dataset_train: Dataset | None = None,
        dataset_val: Dataset | None = None,
        dataset_test: Dataset | None = None,
        *,
        provider: DataProvider | None = None,
        base_dataset: str | None = None,
        num_classes: int | None = None,
        name: str = "csordas:class_removal",
        batch_size: int = 64,
        **kwargs: Any,
    ) -> None:
        if provider is None and base_dataset is not None:
            from positronixlab.dataprovider.dataprovider_factory import (
                DataProviderFactory,
            )

            provider = DataProviderFactory.create(base_dataset, batch_size=batch_size, **kwargs)
        if provider is not None:
            dataset_train = dataset_train or provider.dataset_train
            dataset_val = dataset_val or provider.dataset_val
            dataset_test = dataset_test or provider.dataset_test
            if num_classes is None:
                num_classes = provider.num_classes

        if dataset_train is None or dataset_val is None:
            raise ValueError(
                "ClassRemovalDataProvider requires `provider`, `base_dataset`, or both "
                "`dataset_train` and `dataset_val`."
            )

        super().__init__(name=name, batch_size=batch_size, **kwargs)

        train_labels = _extract_labels(dataset_train)
        val_labels = _extract_labels(dataset_val)
        test_labels = _extract_labels(dataset_test) if dataset_test is not None else None

        if num_classes is None:
            num_classes = (max(train_labels) + 1) if train_labels else 0

        self.num_classes = num_classes
        self.num_features = getattr(provider, "num_features", None)
        self.x_shape = getattr(provider, "x_shape", None)
        self.y_shape = ()
        self.criterion = getattr(provider, "criterion", None) or nn.CrossEntropyLoss()

        split_kw: dict[str, Any] = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            indexed_dataset=self.indexed_dataset,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )

        self.subtasks["all"] = SubtaskSplit(
            name="all",
            dataset_train=dataset_train,
            dataset_val=dataset_val,
            dataset_test=dataset_test or dataset_val,
            **split_kw,
        )

        for cls_id in range(self.num_classes):
            tr_idx = [i for i, y in enumerate(train_labels) if y != cls_id]
            va_idx = [i for i, y in enumerate(val_labels) if y != cls_id]
            te_ds = (
                Subset(dataset_test, [i for i, y in enumerate(test_labels) if y != cls_id])
                if (dataset_test is not None and test_labels is not None)
                else Subset(dataset_val, va_idx)
            )
            self.subtasks[f"without_{cls_id}"] = SubtaskSplit(
                name=f"without_{cls_id}",
                dataset_train=Subset(dataset_train, tr_idx),
                dataset_val=Subset(dataset_val, va_idx),
                dataset_test=te_ds,
                **split_kw,
            )

        self._eval_matrix = {
            f"without_{c}": ["all", f"without_{c}"] for c in range(self.num_classes)
        }
        self._sync_base_split()

    @classmethod
    def from_provider(
        cls,
        provider: DataProvider,
        *,
        name: str | None = None,
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> ClassRemovalDataProvider:
        """Wrap an existing classification ``DataProvider`` (e.g. ``CIFAR10DataProvider``)."""
        return cls(
            provider=provider,
            name=name or f"class_removal:{provider.name}",
            batch_size=provider.batch_size if batch_size is None else batch_size,
            **kwargs,
        )
