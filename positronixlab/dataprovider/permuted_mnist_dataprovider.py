"""Permuted MNIST dataset (Csordas et al., ICLR 2021 — Transfer Learning / P_reuse).

Applies a deterministic pixel permutation ``np.random.RandomState(permutation).permutation(784)``
when ``permutation > 0`` (and leaves pixels unpermuted when ``permutation == 0``),
matching ``are_neural_nets_modular/dataset/image/permuted_mnist.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, Unpack

import numpy as np
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


class PermutedMnistDataset(BaseTensorDataset):
    """MNIST dataset with a fixed deterministic pixel permutation and zero-copy splits."""

    def __init__(
        self,
        permutation: int = 0,
        train: bool = True,
        indices=None,
        flatten: bool = True,
        normalize: bool = True,
        transforms: Optional[TransformLike] = None,
        data_root: Optional[Union[str, Path]] = None,
    ) -> None:
        self.permutation = int(permutation)
        self.train = train
        self.flatten = flatten
        self.data_root = str(resolve_data_root(data_root))

        self.ds = datasets.MNIST(self.data_root, train=train, download=True)
        x = self.ds.data.to(dtype=torch.float32) / 255.0  # (N, 28, 28)
        y = self.ds.targets.long()

        if normalize:
            x = (x - 0.1307) / 0.3081

        flat = x.reshape(x.shape[0], 28 * 28)
        if self.permutation > 0:
            rng = np.random.RandomState(self.permutation)
            perm = torch.from_numpy(rng.permutation(28 * 28)).long()
            flat = flat[:, perm]

        tensor_x = flat if self.flatten else flat.reshape(-1, 1, 28, 28)
        super().__init__(
            tensor_x,
            y,
            indices=indices,
            transforms=transforms,
            normalize=False,  # already applied above
            name_class=self.ds.classes,
        )
        self.normalize = normalize


class PermutedMnistDataProvider(
    DataProvider,
    plugin_name="permuted_mnist",
    aliases=("csordas:permuted_mnist",),
):
    """Unified PositronixLab ``DataProvider`` for Csordas et al.'s Permuted MNIST."""

    supports_flatten = True

    def __init__(
        self,
        permutation: int = 0,
        flatten: bool = True,
        normalize: bool = True,
        data_root: Optional[Union[str, Path]] = None,
        **kwargs: Unpack[DataProviderOptions],
    ) -> None:
        if data_root is not None and "root" not in kwargs:
            kwargs["root"] = data_root
        super().__init__(**kwargs)
        self.name = "permuted_mnist"
        self.permutation = int(permutation)
        self.flatten = flatten
        self.normalize = normalize
        self.data_root = str(self.root)

        self.num_features = 784
        self.num_classes = 10
        self.x_shape = (784,) if flatten else (1, 28, 28)
        self.y_shape = (10,)
        self.criterion = nn.CrossEntropyLoss()

        self.dataset = PermutedMnistDataset(
            permutation=self.permutation,
            train=True,
            flatten=flatten,
            normalize=normalize,
            transforms=self.transforms,
            data_root=self.root,
        )
        test_ds = (
            PermutedMnistDataset(
                permutation=self.permutation,
                train=False,
                flatten=flatten,
                normalize=normalize,
                transforms=self.transforms,
                data_root=self.root,
            )
            if len(self.dataset_split) == 2
            else None
        )
        self._split_dataset(self.dataset, test_dataset=test_ds)

    def with_permutation(self, permutation: int) -> "PermutedMnistDataProvider":
        """Return a provider for a different permutation index (0..10)."""
        return PermutedMnistDataProvider(
            permutation=permutation,
            flatten=self.flatten,
            normalize=self.normalize,
            root=self.root,
            batch_size=self.batch_size,
            split=self.dataset_split,
            num_workers=self.num_workers,
            transforms=self.transforms,
            indexed_dataset=self.indexed_dataset,
            seed=self.seed,
            shuffle_train=self.shuffle_train,
            drop_last=self.drop_last,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )
