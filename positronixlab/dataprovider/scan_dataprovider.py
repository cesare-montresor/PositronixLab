"""SCAN sequence-to-sequence dataset provider (.pt preprocessed or offline dummy fallback)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, Unpack
import warnings

import torch

from positronixlab.dataprovider.dataprovider import (
    BaseTensorDataset,
    DataProvider,
    DataProviderOptions,
    TransformLike,
    resolve_data_root,
)


class SCANDataset(BaseTensorDataset):
    """Preprocessed SCAN tensor dataset with zero-copy split views."""

    def __init__(
        self,
        train: bool = True,
        indices=None,
        transforms: Optional[TransformLike] = None,
        root: Optional[Union[str, Path]] = None,
    ) -> None:
        self.train = train
        self.root = resolve_data_root(root)
        file_path = self.root / "scan" / ("train.pt" if train else "test.pt")

        if file_path.exists():
            data = torch.load(file_path, weights_only=True)
            x = data["inputs"]
            y = data["targets"]
        else:
            warnings.warn(
                f"SCAN data file not found at {file_path}; using dummy tensors.",
                UserWarning,
                stacklevel=2,
            )
            x = torch.zeros((100, 20), dtype=torch.long)
            y = torch.zeros((100, 30), dtype=torch.long)

        super().__init__(x, y, indices=indices, transforms=transforms)


class SCANDataProvider(DataProvider, plugin_name="scan"):
    """DataProvider for preprocessed SCAN tensors."""

    def __init__(self, **kwargs: Unpack[DataProviderOptions]) -> None:
        super().__init__(**kwargs)
        self.name = "scan"
        self.num_features = None
        self.num_classes = None

        self.dataset = SCANDataset(train=True, transforms=self.transforms, root=self.root)
        test_ds = (
            SCANDataset(train=False, transforms=self.transforms, root=self.root)
            if len(self.dataset_split) == 2
            else None
        )
        self._split_dataset(self.dataset, test_dataset=test_ds)
