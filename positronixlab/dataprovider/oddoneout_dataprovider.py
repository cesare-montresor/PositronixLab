"""Generic Odd-One-Out dataset provider (.pt preprocessed or offline dummy fallback)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, Unpack
import warnings

import torch
from torch import nn

from positronixlab.dataprovider.dataprovider import (
    BaseTensorDataset,
    DataProvider,
    DataProviderOptions,
    TransformLike,
    resolve_data_root,
)


class OddOneOutDataset(BaseTensorDataset):
    """Preprocessed 4-way Odd-One-Out image dataset with per-image transforms."""

    def __init__(
        self,
        train: bool = True,
        indices=None,
        transforms: Optional[TransformLike] = None,
        root: Optional[Union[str, Path]] = None,
    ) -> None:
        self.train = train
        self.root = resolve_data_root(root)
        file_path = self.root / "odd_one_out" / ("train.pt" if train else "test.pt")

        if file_path.exists():
            data = torch.load(file_path, weights_only=True)
            x = data["inputs"]
            y = data["targets"]
        else:
            warnings.warn(
                f"OddOneOut data file not found at {file_path}; using dummy tensors.",
                UserWarning,
                stacklevel=2,
            )
            g = torch.Generator().manual_seed(0 if train else 1)
            x = torch.zeros((100, 4, 3, 64, 64), dtype=torch.float32)
            y = torch.randint(0, 4, (100,), dtype=torch.long, generator=g)

        super().__init__(
            x,
            y,
            indices=indices,
            transforms=transforms,
            transform_per_element=True,
        )


class OddOneOutDataProvider(DataProvider, plugin_name="odd_one_out"):
    """DataProvider for 4-way visual Odd-One-Out."""

    supports_flatten = True

    def __init__(self, **kwargs: Unpack[DataProviderOptions]) -> None:
        super().__init__(**kwargs)
        self.name = "odd_one_out"
        self.num_features = None
        self.num_classes = 4
        self.x_shape = (4, 3, 64, 64)
        self.y_shape = ()
        self.criterion = nn.CrossEntropyLoss()

        self.dataset = OddOneOutDataset(train=True, transforms=self.transforms, root=self.root)
        test_ds = (
            OddOneOutDataset(train=False, transforms=self.transforms, root=self.root)
            if len(self.dataset_split) == 2
            else None
        )
        self._split_dataset(self.dataset, test_dataset=test_ds)
