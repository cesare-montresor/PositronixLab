"""CIFAR-10 dataset provider."""

from __future__ import annotations

from pathlib import Path
from typing import Unpack

import torch
from torch import nn
from torchvision import datasets

from positronixlab.dataprovider.dataprovider import (
    BaseTensorDataset,
    DataProvider,
    DataProviderOptions,
    TransformLike,
    resolve_data_root,
)


class CIFAR10Dataset(BaseTensorDataset):
    """CIFAR-10 dataset returning ``(3, 32, 32)`` float32 tensors in ``[0, 1]``."""

    def __init__(
        self,
        train: bool = True,
        indices=None,
        transforms: TransformLike | None = None,
        root: str | Path | None = None,
    ) -> None:
        self.train = train
        self.root = resolve_data_root(root)
        self.ds = datasets.CIFAR10(str(self.root), train=train, download=True)
        x = torch.as_tensor(self.ds.data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        y = torch.as_tensor(self.ds.targets, dtype=torch.long)
        super().__init__(
            x,
            y,
            indices=indices,
            transforms=transforms,
            name_class=self.ds.classes,
        )


class CIFAR10DataProvider(DataProvider, plugin_name="cifar10"):
    """DataProvider for CIFAR-10 (``(3, 32, 32)``, 10 classes)."""

    supports_flatten = True

    def __init__(self, **kwargs: Unpack[DataProviderOptions]) -> None:
        super().__init__(**kwargs)
        self.name = "cifar10"
        self.num_features = 3072
        self.num_classes = 10
        self.x_shape = (3, 32, 32)
        self.y_shape = ()
        self.criterion = nn.CrossEntropyLoss()

        self.dataset = CIFAR10Dataset(train=True, transforms=self.transforms, root=self.root)
        test_ds = (
            CIFAR10Dataset(train=False, transforms=self.transforms, root=self.root)
            if len(self.dataset_split) == 2
            else None
        )
        self._split_dataset(self.dataset, test_dataset=test_ds)
