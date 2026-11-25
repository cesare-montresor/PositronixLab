"""DeepMind Math dataset provider (.pt preprocessed or offline dummy fallback)."""

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


class MathDataset(BaseTensorDataset):
    """Preprocessed DeepMind Math tensor dataset with zero-copy split views."""

    def __init__(
        self,
        train: bool = True,
        indices=None,
        transforms: Optional[TransformLike] = None,
        root: Optional[Union[str, Path]] = None,
    ) -> None:
        self.train = train
        self.root = resolve_data_root(root)
        file_path = self.root / "math" / ("train.pt" if train else "test.pt")

        if file_path.exists():
            data = torch.load(file_path, weights_only=True)
            x = data["inputs"]
            y = data["targets"]
        else:
            warnings.warn(
                f"Math data file not found at {file_path}; using dummy tensors.",
                UserWarning,
                stacklevel=2,
            )
            x = torch.zeros((100, 50), dtype=torch.long)
            y = torch.zeros((100, 10), dtype=torch.long)

        super().__init__(x, y, indices=indices, transforms=transforms)


class MathDataProvider(DataProvider, plugin_name="deepmind_math"):
    """DataProvider for preprocessed DeepMind Math tensors."""

    def __init__(self, **kwargs: Unpack[DataProviderOptions]) -> None:
        super().__init__(**kwargs)
        self.name = "deepmind_math"
        self.num_features = None
        self.num_classes = None

        self.dataset = MathDataset(train=True, transforms=self.transforms, root=self.root)
        test_ds = (
            MathDataset(train=False, transforms=self.transforms, root=self.root)
            if len(self.dataset_split) == 2
            else None
        )
        self._split_dataset(self.dataset, test_dataset=test_ds)
