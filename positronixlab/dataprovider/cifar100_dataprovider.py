"""CIFAR-100 dataset provider."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, Unpack

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


class CIFAR100Dataset(BaseTensorDataset):
    """CIFAR-100 dataset returning ``(3, 32, 32)`` float32 tensors in ``[0, 1]``."""

    def __init__(
        self,
        train: bool = True,
        indices=None,
        transforms: Optional[TransformLike] = None,
        root: Optional[Union[str, Path]] = None,
    ) -> None:
        self.train = train
        self.root = resolve_data_root(root)
        self.ds = datasets.CIFAR100(str(self.root), train=train, download=True)
        x = torch.as_tensor(self.ds.data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        y = torch.as_tensor(self.ds.targets, dtype=torch.long)
        super().__init__(
            x,
            y,
            indices=indices,
            transforms=transforms,
            name_class=self.ds.classes,
        )


class CIFAR100DataProvider(DataProvider, plugin_name="cifar100"):
    """DataProvider for CIFAR-100 (``(3, 32, 32)``, 100 classes)."""

    supports_flatten = True

    def __init__(self, **kwargs: Unpack[DataProviderOptions]) -> None:
        super().__init__(**kwargs)
        self.name = "cifar100"
        self.num_features = 3072
        self.num_classes = 100
        self.x_shape = (3, 32, 32)
        self.y_shape = ()
        self.criterion = nn.CrossEntropyLoss()

        self.dataset = CIFAR100Dataset(train=True, transforms=self.transforms, root=self.root)
        test_ds = (
            CIFAR100Dataset(train=False, transforms=self.transforms, root=self.root)
            if len(self.dataset_split) == 2
            else None
        )
        self._split_dataset(self.dataset, test_dataset=test_ds)
