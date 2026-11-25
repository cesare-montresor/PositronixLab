"""HIGGS physics binary classification dataset provider (memmap + per-epoch subsampling)."""

from __future__ import annotations

import gzip
from pathlib import Path
import shutil
from typing import Any, Optional, Sequence, Tuple, Union, Unpack
import warnings

import numpy as np
import pandas as pd
import requests
import torch
from torch import nn
from torch.utils.data import Dataset, Sampler, random_split

from positronixlab.dataprovider.dataprovider import (
    DataProvider,
    DataProviderOptions,
    DataReader,
    EpochSubsampleSampler,
    SplitName,
    TransformLike,
    apply_transform,
    resolve_data_root,
)


class _HIGGSPoolDataset(Dataset):
    """Non-mutating view over a split's full index pool, driven by ``EpochSubsampleSampler``."""

    def __init__(self, dataset: "HIGGSDataset") -> None:
        self._ds = dataset

    def __len__(self) -> int:
        return self._ds.split_size

    def __getitem__(self, pool_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        real_idx = int(self._ds.indices[pool_idx])
        return self._ds._get_raw_item(real_idx)


class HIGGSDataProvider(DataProvider, plugin_name="higgs"):
    """HIGGS Dataset Provider for binary classification — optimized for large datasets.

    Features:
    - 21 low-level features + 7 high-level features = 28 features
    - 11M samples (subsampled via ``epoch_size`` per epoch)
    - Binary classification: signal (1) vs background (0)

    Key properties:
    - Loads/memmaps ``HIGGS.npy`` once and shares it across splits & worker processes.
    - Uses :class:`EpochSubsampleSampler` for DataLoader workers so calling
      :meth:`set_epoch` resamples cleanly even when ``persistent_workers=True``.
    - Three independent seeds: ``test_seed`` (carves test first), ``split_seed``
      (train/val partition), and ``epoch_seed`` (per-epoch resampling).
    """

    default_split = (0.7, 0.15, 0.15)

    def __init__(
        self,
        *,
        epoch_size: Union[float, int] = 0.1,
        normalize: bool = True,
        split_seed: int = 42,
        test_seed: int = 1234,
        epoch_seed: int = 42,
        **kwargs: Unpack[DataProviderOptions],
    ) -> None:
        super().__init__(**kwargs)

        self.name = "higgs"
        self.num_features = 28
        self.num_classes = 2
        self.x_shape = (28,)
        self.y_shape = ()
        self.criterion = nn.CrossEntropyLoss()
        self.normalize = normalize
        self.epoch_size = epoch_size
        self.split_seed = split_seed
        self.test_seed = test_seed
        self.epoch_seed = epoch_seed

        # Load (or memmap) the raw array once. Shared by every split.
        full_x, full_y = _load_or_build_npy(self.root)
        self.full_x = full_x
        self.full_y = full_y
        true_dataset_size = len(full_x)

        # ---- split into index arrays only (no data copies) ----
        if len(self.dataset_split) == 3:
            train_frac, val_frac, test_frac = self.dataset_split
            test_size = int(test_frac * true_dataset_size)
            remainder_size = true_dataset_size - test_size

            test_split = random_split(
                np.arange(true_dataset_size),
                [remainder_size, test_size],
                generator=torch.Generator().manual_seed(self.test_seed),
            )
            remainder_indices = np.asarray(test_split[0].indices)
            self.test_indices: Optional[np.ndarray] = np.asarray(test_split[1].indices)

            train_ratio = train_frac / (train_frac + val_frac)
            train_size = int(train_ratio * remainder_size)
            val_size = remainder_size - train_size

            trainval_split = random_split(
                remainder_indices,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.split_seed),
            )
            self.train_indices = remainder_indices[np.asarray(trainval_split[0].indices)]
            self.val_indices = remainder_indices[np.asarray(trainval_split[1].indices)]
        elif len(self.dataset_split) == 2:
            train_size = int(self.dataset_split[0] * true_dataset_size)
            val_size = true_dataset_size - train_size
            test_size = 0

            trainval_split = random_split(
                np.arange(true_dataset_size),
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.split_seed),
            )
            self.train_indices = np.asarray(trainval_split[0].indices)
            self.val_indices = np.asarray(trainval_split[1].indices)
            self.test_indices = None
        else:
            raise ValueError(f"split must have 2 or 3 entries, got {self.dataset_split!r}")

        # ---- effective epoch_size per split ----
        if isinstance(epoch_size, int) and not isinstance(epoch_size, bool):
            percentage = epoch_size / max(train_size, 1)
            epoch_size_train: Union[int, float] = epoch_size
            epoch_size_val: Union[int, float] = max(1, int(percentage * val_size))
            epoch_size_test: Union[int, float] = (
                max(1, int(percentage * test_size)) if test_size > 0 else 0
            )
        else:
            epoch_size_train = epoch_size
            epoch_size_val = epoch_size
            epoch_size_test = epoch_size

        # ---- build split datasets as thin views over the shared array ----
        self.dataset_train = HIGGSDataset(
            full_x,
            full_y,
            indices=self.train_indices,
            epoch_size=epoch_size_train,
            normalize=normalize,
            split_name="train",
            transforms=self.transforms,
            base_seed=self.epoch_seed,
        )
        self.dataset_val = HIGGSDataset(
            full_x,
            full_y,
            indices=self.val_indices,
            epoch_size=epoch_size_val,
            normalize=normalize,
            split_name="val",
            transforms=self.transforms,
            base_seed=self.epoch_seed + 10_000,
        )
        self.dataset_test = None
        if self.test_indices is not None and len(self.test_indices) > 0:
            self.dataset_test = HIGGSDataset(
                full_x,
                full_y,
                indices=self.test_indices,
                epoch_size=epoch_size_test,
                normalize=normalize,
                split_name="test",
                transforms=self.transforms,
                base_seed=self.test_seed,
            )
            self.dataset_test.set_epoch(0)

        # ---- worker-safe per-epoch samplers for persistent DataLoader workers ----
        self._train_sampler = EpochSubsampleSampler(
            pool_size=self.dataset_train.split_size,
            epoch_size=self.dataset_train.epoch_size,
            base_seed=self.epoch_seed,
            shuffle=self.shuffle_train,
        )
        self._val_sampler = EpochSubsampleSampler(
            pool_size=self.dataset_val.split_size,
            epoch_size=self.dataset_val.epoch_size,
            base_seed=self.epoch_seed + 10_000,
            shuffle=False,
            sort_for_locality=True,
        )

        # ---- normalization: computed on train, applied to val/test ----
        if normalize:
            self.dataset_train.compute_normalization()
            self.dataset_val.apply_normalization(self.dataset_train.mean, self.dataset_train.std)
            if self.dataset_test is not None:
                self.dataset_test.apply_normalization(
                    self.dataset_train.mean, self.dataset_train.std
                )

    def get_loader(
        self,
        split: SplitName,
        *,
        batch_size: Optional[int] = None,
        shuffle: Optional[bool] = None,
        drop_last: Optional[bool] = None,
        num_workers: Optional[int] = None,
        indexed_dataset: Optional[bool] = None,
        persistent_workers: Optional[bool] = None,
        pin_memory: Optional[bool] = None,
        sampler: Optional[Sampler] = None,
        **kwargs: Any,
    ) -> Optional[DataReader]:
        """Build a worker-safe ``DataReader`` using ``EpochSubsampleSampler`` for train/val."""
        if split == "train" and sampler is None and self.dataset_train is not None:
            return DataReader(
                _HIGGSPoolDataset(self.dataset_train),
                batch_size=self.batch_size if batch_size is None else batch_size,
                num_workers=self.num_workers if num_workers is None else num_workers,
                persistent_workers=(
                    self.persistent_workers if persistent_workers is None else persistent_workers
                ),
                drop_last=self.drop_last if drop_last is None else drop_last,
                indexed_dataset=(
                    self.indexed_dataset if indexed_dataset is None else indexed_dataset
                ),
                shuffle=False,
                pin_memory=self.pin_memory if pin_memory is None else pin_memory,
                sampler=self._train_sampler,
                **kwargs,
            )
        if split == "val" and sampler is None and self.dataset_val is not None:
            return DataReader(
                _HIGGSPoolDataset(self.dataset_val),
                batch_size=self.batch_size if batch_size is None else batch_size,
                num_workers=self.num_workers if num_workers is None else num_workers,
                persistent_workers=(
                    self.persistent_workers if persistent_workers is None else persistent_workers
                ),
                drop_last=self.drop_last if drop_last is None else drop_last,
                indexed_dataset=(
                    self.indexed_dataset if indexed_dataset is None else indexed_dataset
                ),
                shuffle=False,
                pin_memory=self.pin_memory if pin_memory is None else pin_memory,
                sampler=self._val_sampler,
                **kwargs,
            )
        return super().get_loader(
            split,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            indexed_dataset=indexed_dataset,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            sampler=sampler,
            **kwargs,
        )

    def set_epoch(self, epoch: int) -> None:
        """Advance epoch for both persistent DataLoader samplers and direct dataset views.

        ``dataset_test`` is intentionally excluded so test membership stays fixed.
        """
        super().set_epoch(epoch)
        self._train_sampler.set_epoch(epoch)
        self._val_sampler.set_epoch(epoch)
        self.dataset_train.set_epoch(epoch)
        self.dataset_val.set_epoch(epoch)

    def get_feature_groups(self) -> dict[str, list[int]]:
        """Return feature groups: low-level (0-20) and high-level (21-27)."""
        return {
            "low_level": list(range(21)),
            "high_level": list(range(21, 28)),
        }


def _load_or_build_npy(root: Union[str, Path]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ensure ``HIGGS.npy`` exists, then open it read-only memmap'd (zero-copy).

    Returns ``(full_x, full_y)`` backed directly by the memmap; labels are cast
    to ``torch.long`` lazily per item in ``__getitem__`` to avoid copying ~88 MB.
    """
    root_path = resolve_data_root(root)
    root_path.mkdir(parents=True, exist_ok=True)

    raw_file = root_path / "HIGGS.csv.gz"
    processed_file = root_path / "HIGGS.csv"
    npy_file = root_path / "HIGGS.npy"

    if not npy_file.exists():
        if not processed_file.exists():
            _download(raw_file, processed_file)

        print(f"Converting {processed_file} -> {npy_file} (one-time)...")
        columns = ["label"] + [f"feature_{i}" for i in range(28)]
        df = pd.read_csv(processed_file, names=columns)
        data = df.values.astype(np.float32)
        # Keep .npy suffix so np.save does not append a second .npy before rename()
        tmp_npy = npy_file.with_name(f"{npy_file.stem}.tmp.npy")
        np.save(tmp_npy, data)
        tmp_npy.rename(npy_file)
        del df, data

    data = np.load(npy_file, mmap_mode="r")
    # Intentional read-only memmap view: never mutated in-place.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The given NumPy array is not writable",
            category=UserWarning,
        )
        full_x = torch.from_numpy(data[:, 1:])
        full_y = torch.from_numpy(data[:, 0])
    return full_x, full_y


def _download(raw_file: Path, processed_file: Path) -> None:
    """Download + decompress HIGGS with atomic temp files."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
    tmp_file = raw_file.with_suffix(".csv.gz.tmp")

    print(f"Downloading HIGGS dataset to {raw_file}...")
    print("Warning: this is ~2.8GB compressed, ~7.5GB uncompressed.")

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(tmp_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    tmp_file.rename(raw_file)

    print("Decompressing...")
    tmp_csv = processed_file.with_suffix(".csv.tmp")
    with gzip.open(raw_file, "rb") as f_in, open(tmp_csv, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    tmp_csv.rename(processed_file)

    print("Download complete!")


class HIGGSDataset(Dataset):
    """Thin, index-based view over a shared HIGGS array with deterministic per-epoch resampling."""

    def __init__(
        self,
        full_x: torch.Tensor,
        full_y: torch.Tensor,
        indices: Optional[Sequence[int] | np.ndarray] = None,
        transforms: Optional[TransformLike] = None,
        epoch_size: Union[float, int] = 0.1,
        normalize: bool = True,
        split_name: Optional[str] = None,
        base_seed: int = 42,
    ) -> None:
        super().__init__()

        self.full_x = full_x
        self.full_y = full_y
        self.transforms = transforms
        self.normalize = normalize
        self.split_name = split_name
        self.base_seed = base_seed

        self.indices = (
            np.asarray(indices, dtype=np.int64)
            if indices is not None
            else np.arange(len(full_x), dtype=np.int64)
        )
        self.split_size = len(self.indices)

        self.epoch_size = self._compute_epoch_size(epoch_size)

        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None

        self._epoch = -1
        self.x: Optional[torch.Tensor] = None
        self.y: Optional[torch.Tensor] = None

    def _compute_epoch_size(self, epoch_size: Union[float, int]) -> int:
        if isinstance(epoch_size, int) and not isinstance(epoch_size, bool):
            n = min(epoch_size, self.split_size)
        elif isinstance(epoch_size, float):
            if not (0.0 < epoch_size <= 1.0):
                raise ValueError(
                    f"epoch_size as float must be in (0.0, 1.0], got {epoch_size}"
                )
            n = int(epoch_size * self.split_size)
        else:
            raise TypeError(f"epoch_size must be int or float, got {type(epoch_size)}")

        if n < 1:
            warnings.warn(
                f"HIGGSDataset[{self.split_name}]: computed epoch_size={n} "
                f"(split_size={self.split_size}); flooring to 1 to avoid an "
                f"empty dataset."
            )
            n = 1
        return n

    def set_epoch(self, epoch: int) -> None:
        """Resample ``epoch_size`` items from this split deterministically for *epoch*."""
        rng = np.random.default_rng(self.base_seed + epoch)
        epoch_indices = rng.choice(self.indices, size=self.epoch_size, replace=False)

        sorted_idx = np.sort(epoch_indices)
        self.x = self.full_x[sorted_idx]
        self.y = self.full_y[sorted_idx].long()
        self._epoch = epoch

    def compute_normalization(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute normalization stats from a bounded sample of the split."""
        sample_size = min(100_000, self.split_size)
        rng = np.random.default_rng(self.base_seed)
        stat_indices = np.sort(rng.choice(self.indices, size=sample_size, replace=False))
        x_sample = self.full_x[stat_indices].float()

        self.mean = x_sample.mean(dim=0)
        self.std = x_sample.std(dim=0)
        return self.mean, self.std

    def apply_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean = mean
        self.std = std

    def _get_raw_item(self, real_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.full_x[real_idx].float()
        y = self.full_y[real_idx].long()

        if self.normalize and self.mean is not None and self.std is not None:
            x = (x - self.mean) / (self.std + 1e-8)

        if self.transforms is not None:
            x = apply_transform(self.transforms, x)

        return x, y

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._epoch < 0:
            self.set_epoch(0)

        x = self.x[idx].float()
        y = self.y[idx].long()

        if self.normalize and self.mean is not None and self.std is not None:
            x = (x - self.mean) / (self.std + 1e-8)

        if self.transforms is not None:
            x = apply_transform(self.transforms, x)

        return x, y
