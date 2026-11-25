"""Covertype (Forest Cover Type) tabular classification dataset provider."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, Unpack

import numpy as np
import torch
from torch import nn

from positronixlab.dataprovider.dataprovider import (
    BaseTensorDataset,
    DataProvider,
    DataProviderOptions,
    TransformLike,
    resolve_data_root,
)


class CovertypeDataset(BaseTensorDataset):
    """Covertype dataset with zero-copy split views and lazy continuous-feature normalization."""

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        normalize: bool = True,
        indices=None,
        transforms: Optional[TransformLike] = None,
    ) -> None:
        from sklearn.datasets import fetch_covtype

        self.root = resolve_data_root(root)
        self.root.mkdir(parents=True, exist_ok=True)
        data = fetch_covtype(data_home=str(self.root), download_if_missing=True)
        x = torch.from_numpy(data.data.astype(np.float32))
        y = torch.from_numpy(data.target.astype(np.int64)) - 1  # 1..7 -> 0..6

        super().__init__(
            x,
            y,
            indices=indices,
            transforms=transforms,
            normalize=normalize,
            norm_cols=slice(0, 10),
        )


class CovertypeDataProvider(DataProvider, plugin_name="covertype"):
    """Forest Covertype provider (54 features: 10 continuous + 44 binary; 7 classes)."""

    default_split = (0.7, 0.15, 0.15)

    def __init__(
        self,
        *,
        normalize: bool = True,
        **kwargs: Unpack[DataProviderOptions],
    ) -> None:
        super().__init__(**kwargs)
        self.name = "covertype"
        self.num_features = 54
        self.num_classes = 7
        self.x_shape = (54,)
        self.y_shape = ()
        self.criterion = nn.CrossEntropyLoss()
        self.normalize = normalize

        # Single load shared across all splits via zero-copy views
        self.dataset = CovertypeDataset(
            root=self.root,
            normalize=normalize,
            transforms=self.transforms,
        )
        self._split_dataset(self.dataset)

        if normalize and isinstance(self.dataset_train, BaseTensorDataset):
            mean, std = self.dataset_train.compute_normalization()
            if isinstance(self.dataset_val, BaseTensorDataset):
                self.dataset_val.apply_normalization(mean, std)
            if isinstance(self.dataset_test, BaseTensorDataset):
                self.dataset_test.apply_normalization(mean, std)

    def get_feature_groups(self) -> dict[str, list[int]]:
        """Return semantic feature groups."""
        return {
            "elevation": [0],
            "aspect": [1],
            "slope": [2],
            "hydrology_distance": [3, 4],
            "roadways_distance": [5],
            "hillshade": [6, 7, 8],
            "firepoints_distance": [9],
            "wilderness_area": list(range(10, 14)),
            "soil_type": list(range(14, 54)),
        }
