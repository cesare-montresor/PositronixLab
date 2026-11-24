"""Permuted MNIST dataset (Csordas et al., ICLR 2021 — Transfer Learning / P_reuse).

Applies a deterministic pixel permutation ``np.random.RandomState(permutation).permutation(784)``
when ``permutation > 0`` (and leaves pixels unpermuted when ``permutation == 0``),
matching ``are_neural_nets_modular/dataset/image/permuted_mnist.py``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, random_split
from torchvision import datasets

from positronixlab.dataprovider.dataprovider import DataProvider


class PermutedMnistDataset(Dataset):
    """MNIST dataset with a fixed deterministic pixel permutation."""

    def __init__(
        self,
        permutation: int = 0,
        train: bool = True,
        indices=None,
        flatten: bool = True,
        normalize: bool = True,
        transforms=None,
        data_root: str = "./data",
    ) -> None:
        super().__init__()
        self.permutation = int(permutation)
        self.train = train
        self.indices = indices
        self.flatten = flatten
        self.normalize = normalize
        self.transforms = transforms
        self.data_root = data_root

        self.ds = datasets.MNIST(data_root, train=train, download=True)
        self.name_class = self.ds.classes
        self.num_class = len(self.name_class)

        x = self.ds.data.to(dtype=torch.float32) / 255.0  # (N, 28, 28)
        y = self.ds.targets.long()

        if self.normalize:
            # Standard MNIST channel mean/std on [0, 1] scale
            x = (x - 0.1307) / 0.3081

        flat = x.reshape(x.shape[0], 28 * 28)
        if self.permutation > 0:
            rng = np.random.RandomState(self.permutation)
            perm = torch.from_numpy(rng.permutation(28 * 28)).long()
            flat = flat[:, perm]

        self.x = flat if self.flatten else flat.reshape(-1, 1, 28, 28)
        self.y = y

        if indices is not None:
            self.x = self.x[indices]
            self.y = self.y[indices]

    def copy(self, indices=None) -> "PermutedMnistDataset":
        return PermutedMnistDataset(
            permutation=self.permutation,
            train=self.train,
            indices=indices,
            flatten=self.flatten,
            normalize=self.normalize,
            transforms=self.transforms,
            data_root=self.data_root,
        )

    def random_split(self, split):
        subsets = random_split(self, split)
        return [self.copy(subset.indices) for subset in subsets]

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.x[idx]
        if self.transforms is not None:
            x = self.transforms.apply(x)
        return x, self.y[idx]


class PermutedMnistDataProvider(DataProvider):
    """Unified PositronixLab ``DataProvider`` for Csordas et al.'s Permuted MNIST."""

    def __init__(
        self,
        permutation: int = 0,
        flatten: bool = True,
        normalize: bool = True,
        data_root: str = "./data",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.name = "permuted_mnist"
        self.permutation = int(permutation)
        self.flatten = flatten
        self.normalize = normalize
        self.data_root = data_root

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
            data_root=data_root,
        )
        self.dataset_train, self.dataset_val = self.dataset.random_split(self.dataset_split)
        self.dataset_test = PermutedMnistDataset(
            permutation=self.permutation,
            train=False,
            flatten=flatten,
            normalize=normalize,
            transforms=self.transforms,
            data_root=data_root,
        )

    def with_permutation(self, permutation: int) -> "PermutedMnistDataProvider":
        """Return a provider for a different permutation index (0..10)."""
        return PermutedMnistDataProvider(
            permutation=permutation,
            flatten=self.flatten,
            normalize=self.normalize,
            data_root=self.data_root,
            batch_size=self.batch_size,
            split=self.dataset_split,
            num_workers=self.num_workers,
            transforms=self.transforms,
            indexed_dataset=self.indexed_dataset,
        )
