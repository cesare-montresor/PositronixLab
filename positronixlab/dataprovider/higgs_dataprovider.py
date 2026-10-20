# =============================================================================
# HIGGS Dataset
# =============================================================================
#
# Design notes (why this looks the way it does):
#
# 1. SINGLE SHARED LOAD. The raw array is loaded/memmapped exactly once by the
#    provider and handed to every split as a *view* (indices), never copied.
#    Old version: dataset_full, dataset_train, dataset_val, dataset_test each
#    called _load_full_data() independently -> 4x the RAM for the same data.
#
# 2. MEMMAP, NOT A FULL IN-RAM ARRAY. HIGGS.npy is opened with
#    np.load(..., mmap_mode='r'). Pages are lazily faulted in by the OS and,
#    critically, shared across DataLoader worker processes (they're the same
#    file-backed pages) instead of being duplicated on fork/copy-on-write.
#
# 3. REAL PER-EPOCH RESAMPLING. Previously _sample_epoch() only ran once
#    (guarded by `if self.x is None`), so the "epoch_size samples per epoch"
#    behavior described in the docstring never actually happened after epoch 0.
#    This version exposes set_epoch(epoch), matching the standard
#    DistributedSampler-style contract: call dataset.set_epoch(e) at the start
#    of each training epoch and it resamples deterministically from a local
#    RNG. If the caller never calls it, the dataset still resamples on first
#    access, but with a clear, documented, single-owner mechanism instead of
#    a silent no-op after the first call.
#
# 4. LOCAL RNG. No more np.random.seed(...) global mutation, which previously
#    clobbered RNG state for every other consumer of np.random in the process.
#
# 5. EDGE CASES. epoch_size is floored to >=1 with a warning instead of
#    silently producing a zero-length (and therefore empty/crashing) split.
#
# 6. SEEDS ARE SPLIT INTO THREE INDEPENDENT CONCERNS:
#    - test_seed: carves the test set out of the FULL dataset first, before
#      anything else is split. It has its own default (1234) so that test
#      membership never moves when you change split_seed to try different
#      train/val partitions.
#    - split_seed: splits whatever's left (after test is removed) into
#      train/val. Safe to vary across experiments.
#    - epoch_seed: drives per-epoch resampling within a split (see
#      HIGGSDataset.set_epoch), independent of which rows belong to which
#      split. Train/val/test each get an offset from epoch_seed so their
#      resampling streams don't collide.
#    dataset_test is sampled once (epoch 0, test_seed) and held fixed for
#    the whole run -- it does not participate in HIGGSDataProvider.set_epoch.

from pathlib import Path
import gzip
import shutil
import warnings

import numpy as np
import pandas as pd
import requests
import torch
from torch.utils.data import Dataset, random_split

from positronixlab.dataprovider.dataprovider import DataProvider


class HIGGSDataProvider(DataProvider):
    """
    HIGGS Dataset Provider for binary classification - optimized for large datasets.

    Features:
    - 21 low-level features (raw detector measurements)
    - 7 high-level features (physics-derived quantities)
    - Total: 28 features
    - 11M samples (process via epoch_size for efficiency)
    - Binary classification: signal (1) vs background (0)

    Design for large physics datasets:
    - Loads/memmaps the raw array exactly once and shares it across splits
    - Within each split, randomly (re)samples epoch_size samples per epoch
    - Ideal for multi-job / multi-worker scenarios on shared hardware, since
      the memmap is OS-page-cache-shared rather than duplicated per split
      or per DataLoader worker

    Args:
        batch_size: Batch size for dataloaders
        split: Train/val/test split ratios (default [0.7, 0.15, 0.15])
        num_workers: Number of dataloader workers
        transforms: Optional transforms
        normalize: Whether to normalize features (computed from train set)
        epoch_size: Samples per epoch. Two modes:
            - Float (0.0-1.0): percentage applied independently to each split
              e.g., 0.1 = 10% of train, 10% of val, 10% of test
            - Int (>1): absolute sample count for training set; percentage
              computed and applied to val/test as well
        root: Root directory for data
        split_seed: seed for the train/val partition. Safe to vary across
            experiments -- it only reshuffles which rows land in train vs val.
        test_seed: seed for carving the test set out of the full dataset.
            Deliberately separate from split_seed and defaulted differently,
            so the test set stays identical across experiments even if you
            change split_seed for train/val ablations. Test is carved out
            FIRST, before train/val are split from what's left.
        epoch_seed: base seed for per-epoch subsampling (see
            HIGGSDataset.set_epoch). Controls which rows get drawn into a
            given epoch's `epoch_size` subsample, independent of the
            partition seeds above.
    """

    def __init__(self,
                 root='./data',
                 epoch_size=0.1,
                 normalize=True,
                 split=[0.7, 0.15, 0.15],
                 split_seed=42,
                 test_seed=1234,
                 epoch_seed=42,
                 **kwargs):
        super().__init__(split=split, **kwargs)

        self.name = 'higgs'
        self.num_features = 28
        self.num_classes = 2
        self.x_shape = (28,)
        self.y_shape = ()
        self.normalize = normalize
        self.root = Path(root)
        self.epoch_size = epoch_size
        self.split_seed = split_seed
        self.test_seed = test_seed
        self.epoch_seed = epoch_seed

        # Load (or memmap) the raw array exactly once. Shared by every split.
        full_x, full_y = _load_or_build_npy(self.root)
        self.full_x = full_x
        self.full_y = full_y
        true_dataset_size = len(full_x)

        # ---- split into index arrays only (no data copies) ----
        # Test is carved out FIRST with its own dedicated seed so it stays
        # fixed across runs even if split_seed is changed to try different
        # train/val partitions. Only the remainder is then split into
        # train/val using split_seed.
        if len(self.dataset_split) == 3:
            train_frac, val_frac, test_frac = self.dataset_split
            test_size = int(test_frac * true_dataset_size)
            remainder_size = true_dataset_size - test_size

            test_split = random_split(
                np.arange(true_dataset_size),
                [remainder_size, test_size],
                generator=torch.Generator().manual_seed(self.test_seed)
            )
            remainder_indices = np.asarray(test_split[0].indices)
            self.test_indices = np.asarray(test_split[1].indices)

            # train/val ratios renormalized over the remainder
            train_ratio = train_frac / (train_frac + val_frac)
            train_size = int(train_ratio * remainder_size)
            val_size = remainder_size - train_size

            trainval_split = random_split(
                remainder_indices,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.split_seed)
            )
            self.train_indices = remainder_indices[np.asarray(trainval_split[0].indices)]
            self.val_indices = remainder_indices[np.asarray(trainval_split[1].indices)]
        else:
            train_size = int(self.dataset_split[0] * true_dataset_size)
            val_size = true_dataset_size - train_size
            test_size = 0
            self.train_indices = np.arange(train_size)
            self.val_indices = np.arange(train_size, true_dataset_size)
            self.test_indices = None

        # ---- effective epoch_size per split ----
        if isinstance(epoch_size, int) and epoch_size > 1:
            percentage = epoch_size / train_size
            epoch_size_train = epoch_size
            epoch_size_val = int(percentage * val_size)
            epoch_size_test = int(percentage * test_size) if test_size > 0 else 0
        else:
            epoch_size_train = epoch_size
            epoch_size_val = epoch_size
            epoch_size_test = epoch_size

        # ---- build split datasets as thin views over the shared array ----
        # epoch_seed offsets keep train/val/test resampling streams
        # independent of each other despite sharing the same base seed.
        self.dataset_train = HIGGSDataset(
            full_x, full_y, indices=self.train_indices,
            epoch_size=epoch_size_train, normalize=normalize, split_name='train',
            transforms=self.transforms, base_seed=self.epoch_seed,
        )
        self.dataset_val = HIGGSDataset(
            full_x, full_y, indices=self.val_indices,
            epoch_size=epoch_size_val, normalize=normalize, split_name='val',
            transforms=self.transforms, base_seed=self.epoch_seed + 10_000,
        )
        self.dataset_test = None
        if self.test_indices is not None and len(self.test_indices) > 0:
            self.dataset_test = HIGGSDataset(
                full_x, full_y, indices=self.test_indices,
                epoch_size=epoch_size_test, normalize=normalize, split_name='test',
                transforms=self.transforms, base_seed=self.test_seed,
            )
            # Test is deliberately NOT wired into set_epoch() below: it's
            # sampled once here, using test_seed, and then held fixed for
            # the entire run so evaluation numbers stay comparable across
            # epochs and across experiments that only change epoch_seed.
            self.dataset_test.set_epoch(0)

        # ---- normalization: computed on train, applied to val/test ----
        if normalize:
            self.dataset_train.compute_normalization()
            self.dataset_val.apply_normalization(self.dataset_train.mean, self.dataset_train.std)
            if self.dataset_test is not None:
                self.dataset_test.apply_normalization(self.dataset_train.mean, self.dataset_train.std)

    def set_epoch(self, epoch: int):
        """
        Per-epoch resampling hook. Call this at the start of every training
        epoch so dataset_train/dataset_val actually draw a fresh epoch_size
        subsample (see HIGGSDataset.set_epoch) -- without this call the old
        bug (subsample fixed after epoch 0) effectively comes back.

        dataset_test is intentionally excluded: it's sampled once at
        construction time with test_seed and held fixed for the whole run.

        NOTE: I don't have BaseDataProvider's source, so I don't know if it
        already defines a per-epoch lifecycle hook (possibly under a
        different name, e.g. on_epoch_start). If it does, rename/route this
        method to match that hook instead of calling it manually; if it
        doesn't, call `provider.set_epoch(epoch)` explicitly from the
        training loop at the top of each epoch.
        """
        if hasattr(super(), 'set_epoch'):
            super().set_epoch(epoch)
        self.dataset_train.set_epoch(epoch)
        self.dataset_val.set_epoch(epoch)

    def get_feature_groups(self):
        """Returns feature groups: low-level (0-20) and high-level (21-27)"""
        return {
            'low_level': list(range(21)),
            'high_level': list(range(21, 28)),
        }


def _load_or_build_npy(root: Path):
    """
    Ensure HIGGS.npy exists (downloading/converting the CSV if needed), then
    open it memmap'd so the backing pages are shared OS-wide instead of
    duplicated per split or per DataLoader worker process.

    Returns (full_x, full_y) as torch tensors backed by the memmap (no copy;
    torch.from_numpy on a memmap array shares the same underlying buffer).
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    raw_file = root / "HIGGS.csv.gz"
    processed_file = root / "HIGGS.csv"
    npy_file = root / "HIGGS.npy"

    if not npy_file.exists():
        if not processed_file.exists():
            _download(raw_file, processed_file)

        print(f"Converting {processed_file} -> {npy_file} (one-time)...")
        columns = ['label'] + [f'feature_{i}' for i in range(28)]
        df = pd.read_csv(processed_file, names=columns)
        data = df.values.astype(np.float32)
        # atomic-ish write: build under a temp name, then rename into place
        tmp_npy = npy_file.with_suffix('.npy.tmp')
        np.save(tmp_npy, data)
        tmp_npy.rename(npy_file)
        del df, data

    data = np.load(npy_file, mmap_mode='r')
    # torch.from_numpy on a memmap keeps the memmap view -- no copy, no cast
    # here. Casting happens lazily per-item in __getitem__ instead.
    full_x = torch.from_numpy(data[:, 1:])
    full_y = torch.from_numpy(data[:, 0].astype(np.int64))
    return full_x, full_y


def _download(raw_file: Path, processed_file: Path):
    """Download + decompress HIGGS with a resumable/atomic-ish tmp file."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
    tmp_file = raw_file.with_suffix('.csv.gz.tmp')

    print(f"Downloading HIGGS dataset to {raw_file}...")
    print("Warning: this is ~2.8GB compressed, ~7.5GB uncompressed.")

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(tmp_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    tmp_file.rename(raw_file)  # only "commit" the file once fully downloaded

    print("Decompressing...")
    tmp_csv = processed_file.with_suffix('.csv.tmp')
    with gzip.open(raw_file, 'rb') as f_in, open(tmp_csv, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    tmp_csv.rename(processed_file)

    print("Download complete!")


class HIGGSDataset(Dataset):
    """
    A thin, index-based view over a shared HIGGS array with true per-epoch
    resampling.

    Call `set_epoch(epoch)` at the start of every training epoch (the same
    contract as torch's DistributedSampler) to get a fresh random subsample
    of `epoch_size` items drawn from this split's indices. If you never call
    it, the first access resamples once with epoch=0 and stays fixed for the
    life of the object -- which is at least explicit and documented, instead
    of being an accidental one-shot.
    """

    def __init__(self,
                 full_x: torch.Tensor,
                 full_y: torch.Tensor,
                 indices=None,
                 transforms=None,
                 epoch_size=0.1,
                 normalize=True,
                 split_name=None,
                 base_seed=42):
        super().__init__()

        self.full_x = full_x
        self.full_y = full_y
        self.transforms = transforms
        self.normalize = normalize
        self.split_name = split_name
        self.base_seed = base_seed

        self.indices = indices if isinstance(indices, np.ndarray) else np.arange(len(full_x))
        self.split_size = len(self.indices)

        self.epoch_size = self._compute_epoch_size(epoch_size)

        self.mean = None
        self.std = None

        self._epoch = -1  # forces a resample on first access
        self.x = None
        self.y = None

    def _compute_epoch_size(self, epoch_size):
        if isinstance(epoch_size, int):
            n = min(epoch_size, self.split_size)
        elif isinstance(epoch_size, float):
            if not (0.0 < epoch_size <= 1.0):
                raise ValueError(f"epoch_size as float must be in (0.0, 1.0], got {epoch_size}")
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

    def set_epoch(self, epoch: int):
        """Resample epoch_size items from this split, deterministically per-epoch."""
        rng = np.random.default_rng(self.base_seed + epoch)
        epoch_indices = rng.choice(self.indices, size=self.epoch_size, replace=False)

        # Sorting before the fancy-index read is friendlier to the memmap's
        # page cache (fewer random seeks) than reading in shuffled order.
        order = np.argsort(epoch_indices)
        sorted_idx = epoch_indices[order]

        self.x = self.full_x[sorted_idx]
        self.y = self.full_y[sorted_idx]
        self._epoch = epoch

    def compute_normalization(self):
        """Compute normalization stats from a bounded sample of the split."""
        sample_size = min(100_000, self.split_size)
        rng = np.random.default_rng(self.base_seed)
        stat_indices = np.sort(rng.choice(self.indices, size=sample_size, replace=False))
        x_sample = self.full_x[stat_indices].float()

        self.mean = x_sample.mean(dim=0)
        self.std = x_sample.std(dim=0)

    def apply_normalization(self, mean, std):
        self.mean = mean
        self.std = std

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        if self._epoch < 0:
            self.set_epoch(0)

        x = self.x[idx].float()
        y = self.y[idx]

        if self.normalize and self.mean is not None:
            x = (x - self.mean) / (self.std + 1e-8)

        if self.transforms is not None:
            x = self.transforms.apply(x)

        return x, y