"""MNIST handwritten digits dataset provider."""

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


class MnistDataset(BaseTensorDataset):
    """MNIST dataset returning ``(1, 28, 28)`` float32 tensors in ``[0, 1]``."""

    def __init__(
        self,
        train: bool = True,
        indices=None,
        transforms: Optional[TransformLike] = None,
        root: Optional[Union[str, Path]] = None,
    ) -> None:
        self.train = train
        self.root = resolve_data_root(root)
        self.ds = datasets.MNIST(str(self.root), train=train, download=True)
        x = (self.ds.data.to(dtype=torch.float32) / 255.0).unsqueeze(1)
        y = self.ds.targets.long()
        super().__init__(
            x,
            y,
            indices=indices,
            transforms=transforms,
            name_class=self.ds.classes,
        )


class MnistDataProvider(DataProvider, plugin_name="mnist"):
    """DataProvider for MNIST (``(1, 28, 28)``, 10 classes)."""

    supports_flatten = True

    def __init__(self, **kwargs: Unpack[DataProviderOptions]) -> None:
        super().__init__(**kwargs)
        self.name = "mnist"
        self.num_features = 784
        self.num_classes = 10
        self.x_shape = (1, 28, 28)
        self.y_shape = ()
        self.criterion = nn.CrossEntropyLoss()

        self.dataset = MnistDataset(train=True, transforms=self.transforms, root=self.root)
        test_ds = (
            MnistDataset(train=False, transforms=self.transforms, root=self.root)
            if len(self.dataset_split) == 2
            else None
        )
        self._split_dataset(self.dataset, test_dataset=test_ds)
