# =============================================================================
# HIGGS Dataset
# =============================================================================

from pathlib import Path
import gzip
import shutil
import requests
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from modularnet.dataprovider.base_dataprovider import BaseDataProvider


class HIGGSDataProvider(BaseDataProvider):
    """
    HIGGS Dataset Provider for binary classification - optimized for large datasets.
    
    Features:
    - 21 low-level features (raw detector measurements)
    - 7 high-level features (physics-derived quantities)
    - Total: 28 features
    - 11M samples (process via epoch_size for efficiency)
    - Binary classification: signal (1) vs background (0)
    
    Design for large physics datasets:
    - Splits data into train/val/test chunks first
    - Within each chunk, randomly samples epoch_size for each epoch
    - No need to load entire dataset into memory
    - Ideal for multi-job scenarios on shared hardware
    
    Args:
        batch_size: Batch size for dataloaders
        split: Train/val/test split ratios (default [0.7, 0.15, 0.15])
        num_workers: Number of dataloader workers
        transforms: Optional transforms
        normalize: Whether to normalize features (computed from train set)
        epoch_size: Samples per epoch. Two modes:
            - Float (0.0-1.0): percentage applied independently to each split
              e.g., 0.1 = 10% of train, 10% of val, 10% of test
            - Int (>1): absolute sample count for training set; percentage computed
              and applied to val/test as well
              e.g., 1000000 samples in train → compute %, apply to val and test
        root: Root directory for data
    """
    def __init__(self,
                 root='./data',
                 epoch_size=0.1,
                 normalize=True,
                 split=[0.7, 0.15, 0.15],
                 **kwargs
                 ):
        super().__init__(
            split=split,
           **kwargs
        )
        
        self.name = 'higgs'
        self.num_features = 28
        self.num_classes = 2
        self.x_shape = (28,)
        self.y_shape = ()
        self.normalize = normalize
        self.root = Path(root)
        self.epoch_size = epoch_size

        # Load full dataset metadata (lazy load) - always use 100% for accurate split size
        self.dataset_full = HIGGSDataset(
            root=root,
            transforms=self.transforms,
            epoch_size=1.0  # Use full dataset to compute accurate split indices
        )
        
        # Get true dataset size from full_x/full_y (set by _load_full_data)
        true_dataset_size = len(self.dataset_full.full_x)
        
        # Split into train/val/test indices (doesn't load data yet)
        if len(self.dataset_split) == 3:
            train_size = int(self.dataset_split[0] * true_dataset_size)
            val_size = int(self.dataset_split[1] * true_dataset_size)
            test_size = true_dataset_size - train_size - val_size
            
            splits = random_split(
                self.dataset_full, 
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            self.train_indices = splits[0].indices
            self.val_indices = splits[1].indices
            self.test_indices = splits[2].indices
        else:
            train_size = int(self.dataset_split[0] * true_dataset_size)
            self.train_indices = np.arange(train_size)
            self.val_indices = np.arange(train_size, true_dataset_size)
            self.test_indices = None
            val_size = true_dataset_size - train_size
            test_size = 0
        
        # Compute effective epoch_size for each split
        # If epoch_size is int (>1): compute percentage from training set, apply to all splits
        # If epoch_size is float (0.0-1.0): apply independently to each split
        if isinstance(epoch_size, int) and epoch_size > 1:
            # epoch_size is absolute number for training set
            percentage = epoch_size / train_size
            epoch_size_train = epoch_size
            epoch_size_val = int(percentage * val_size)
            epoch_size_test = int(percentage * test_size) if test_size > 0 else 0
        else:
            # epoch_size is percentage, apply to each split independently
            epoch_size_train = epoch_size
            epoch_size_val = epoch_size
            epoch_size_test = epoch_size
        
        # Create split datasets with computed epoch_size
        self.dataset_train = HIGGSDataset(
            root=root,
            transforms=self.transforms,
            indices=self.train_indices,
            epoch_size=epoch_size_train,
            normalize=normalize,
            split_name='train'
        )
        
        self.dataset_val = HIGGSDataset(
            root=root,
            transforms=self.transforms,
            indices=self.val_indices,
            epoch_size=epoch_size_val,
            normalize=normalize,
            split_name='val'
        )
        
        if self.test_indices is not None:
            self.dataset_test = HIGGSDataset(
                root=root,
                transforms=self.transforms,
                indices=self.test_indices,
                epoch_size=epoch_size_test,
                normalize=normalize,
                split_name='test'
            )
        else:
            self.dataset_test = None
        
        # Compute normalization from training set
        if normalize:
            self.dataset_train.compute_normalization()
            self.dataset_val.apply_normalization(self.dataset_train.mean, self.dataset_train.std)
            if self.dataset_test is not None:
                self.dataset_test.apply_normalization(self.dataset_train.mean, self.dataset_train.std)
        
    
    def get_feature_groups(self):
        """Returns feature groups: low-level (0-20) and high-level (21-27)"""
        return {
            'low_level': list(range(21)),
            'high_level': list(range(21, 28))
        }


class HIGGSDataset(Dataset):
    """
    HIGGS Dataset with lazy loading, efficient splits, and epoch-based sampling.
    
    Features:
    - Lazy loading: doesn't load data until needed
    - Epoch sampling: randomly samples epoch_size from the split each time __getitem__ is called
    - Memory efficient: only keeps active epoch in memory
    - Supports arbitrary train/val/test splits
    """
    
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
    
    def __init__(self,
                 root='./data',
                 indices=None,
                 transforms=None,
                 epoch_size=0.1,
                 normalize=True,
                 split_name=None):
        super().__init__()
        
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.transforms = transforms
        self.normalize = normalize
        self.split_name = split_name
        
        # File paths
        self.raw_file = self.root / "HIGGS.csv.gz"
        self.processed_file = self.root / "HIGGS.csv"
        self.npy_file = self.root / "HIGGS.npy"
        
        # Download if needed
        if not self.processed_file.exists() and not self.npy_file.exists():
            self._download()
        
        # Load full dataset first (to get indices)
        self._load_full_data()
        
        # Apply split indices if provided
        if indices is not None:
            self.indices = indices if isinstance(indices, np.ndarray) else np.array(indices)
            self.split_size = len(self.indices)
        else:
            self.indices = np.arange(len(self.full_x))
            self.split_size = len(self.full_x)
        
        # Compute epoch size
        self.epoch_size = self._compute_epoch_size(epoch_size)
        
        # Current epoch data (sampled on demand)
        self.x = None
        self.y = None
        self.current_epoch_indices = None
        
        # Normalization stats
        self.mean = None
        self.std = None
    
    def _compute_epoch_size(self, epoch_size):
        """Convert epoch_size to integer sample count"""
        if isinstance(epoch_size, int):
            return min(epoch_size, self.split_size)
        elif isinstance(epoch_size, float):
            if not (0.0 < epoch_size <= 1.0):
                raise ValueError(f"epoch_size as float must be in (0.0, 1.0], got {epoch_size}")
            return max(1, int(epoch_size * self.split_size))
        else:
            raise TypeError(f"epoch_size must be int or float, got {type(epoch_size)}")
    
    def _download(self):
        """Download HIGGS dataset"""
        print(f"Downloading HIGGS dataset to {self.raw_file}...")
        print("Warning: This is ~2.8GB compressed, ~7.5GB uncompressed!")
        print("This may take a while...")
        
        response = requests.get(self.URL, stream=True)
        response.raise_for_status()
        
        with open(self.raw_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print("Decompressing...")
        with gzip.open(self.raw_file, 'rb') as f_in:
            with open(self.processed_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        print("Download complete!")
    
    def _load_full_data(self):
        """Load full dataset from CSV or NPY (cached for efficiency)"""
        if self.npy_file.exists():
            print(f"Loading HIGGS from {self.npy_file}...")
            data = np.load(self.npy_file)
        else:
            print(f"Loading HIGGS from {self.processed_file}...")
            columns = ['label'] + [f'feature_{i}' for i in range(28)]
            df = pd.read_csv(self.processed_file, names=columns)
            data = df.values
            np.save(self.npy_file, data)
        
        self.full_x = torch.from_numpy(data[:, 1:].astype(np.float32))
        self.full_y = torch.from_numpy(data[:, 0].astype(np.int64))
    
    def _sample_epoch(self, seed=None):
        """Randomly sample epoch_size from the split"""
        if seed is None:
            # Use a seed derived from current time for variety across runs
            seed = np.random.randint(0, 2**31)
        
        np.random.seed(seed)
        epoch_indices = np.random.choice(self.indices, size=self.epoch_size, replace=False)
        
        self.x = self.full_x[epoch_indices]
        self.y = self.full_y[epoch_indices]
        self.current_epoch_indices = epoch_indices
    
    def compute_normalization(self):
        """Compute normalization stats from full split (for efficiency, sample if large)"""
        # Sample a reasonable size for computing stats (max 100k samples)
        sample_size = min(100000, self.split_size)
        if sample_size < self.split_size:
            np.random.seed(42)
            stat_indices = np.random.choice(self.indices, size=sample_size, replace=False)
            x_sample = self.full_x[stat_indices]
        else:
            x_sample = self.full_x[self.indices]
        
        self.mean = x_sample.mean(dim=0)
        self.std = x_sample.std(dim=0)
    
    def apply_normalization(self, mean, std):
        """Apply precomputed normalization (from training set)"""
        self.mean = mean
        self.std = std
    
    def __len__(self):
        """Returns epoch_size (samples available per epoch)"""
        return self.epoch_size
    
    def __getitem__(self, idx):
        """Get item, sampling new epoch if needed"""
        # Sample a new epoch on first access or after resets
        if self.x is None:
            self._sample_epoch()
        
        # Apply normalization if available
        x = self.x[idx % len(self.x)]
        y = self.y[idx % len(self.y)]
        
        if self.normalize and self.mean is not None:
            x = (x - self.mean) / (self.std + 1e-8)
        
        if self.transforms is not None:
            x = self.transforms.apply(x)
        
        return x, y
