"""Iris tabular classification dataset provider."""

from __future__ import annotations

from typing import Any, Unpack

import sklearn.datasets as sklearn_datasets
import torch
from torch import nn

from positronixlab.dataprovider.dataprovider import (
    BaseTensorDataset,
    DataProvider,
    DataProviderOptions,
    TransformLike,
)


class IrisDataset(BaseTensorDataset):
    """Offline scikit-learn Iris dataset (150 samples, 4 features, 3 classes)."""

    def __init__(
        self,
        *,
        indices=None,
        transforms: TransformLike | None = None,
    ) -> None:
        iris: Any = sklearn_datasets.load_iris()
        self.ds = iris
        x = torch.as_tensor(iris["data"], dtype=torch.float32)
        y = torch.as_tensor(iris["target"], dtype=torch.long)
        super().__init__(
            x,
            y,
            indices=indices,
            transforms=transforms,
            name_class=list(iris["target_names"]),
        )


class IrisDataProvider(DataProvider, plugin_name="iris"):
    """DataProvider for the Iris dataset with proper train/val(/test) splitting."""

    def __init__(self, **kwargs: Unpack[DataProviderOptions]) -> None:
        super().__init__(**kwargs)
        self.name = "iris"
        self.num_features = 4
        self.num_classes = 3
        self.x_shape = (4,)
        self.y_shape = ()
        self.criterion = nn.CrossEntropyLoss()

        self.dataset = IrisDataset(transforms=self.transforms)
        self._split_dataset(self.dataset)
