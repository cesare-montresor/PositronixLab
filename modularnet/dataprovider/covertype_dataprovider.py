
# =============================================================================
# Covertype Dataset
# =============================================================================

from modularnet.dataprovider.base_dataprovider import BaseDataProvider
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import numpy as np
from pathlib import Path

class CovertypeDataProvider(BaseDataProvider):
    """
    Covertype (Forest Cover Type) Dataset Provider for multi-class classification.
    
    Features (54 total):
    - Elevation, Aspect, Slope (3 continuous)
    - Distances to hydrology, roadways, fire points (4 continuous)
    - Hillshade 9am/12pm/3pm (3 continuous) 
    - Wilderness area (4 binary)
    - Soil type (40 binary)
    
    Args:
        batch_size: Batch size for dataloaders
        split: Train/val/test split ratios (default [0.7, 0.15, 0.15])
        device: Device to use
        num_workers: Number of dataloader workers
        transforms: Optional transforms
        normalize: Whether to normalize continuous features
        root: Root directory for data
    """
    def __init__(self,
                 batch_size=32,
                 split=None,
                 device=None,
                 num_workers=None,
                 transforms=None,
                 normalize=True,
                 root='./data'):
        super().__init__(
            batch_size=batch_size,
            split=split if split is not None else [0.7, 0.15, 0.15],
            device=device,
            num_workers=num_workers,
            transforms=transforms
        )
        
        self.name = 'covertype'
        self.num_features = 54
        self.num_classes = 7
        self.x_shape = (54,)
        self.y_shape = ()
        self.normalize = normalize
        self.root = Path(root)
        
        # Create dataset
        self.dataset = CovertypeDataset(
            root=root,
            normalize=normalize,
            device=device,
            transforms=transforms
        )
        
        # Split dataset
        if len(self.dataset_split) == 3:
            # Three-way split
            train_size = int(self.dataset_split[0] * len(self.dataset))
            val_size = int(self.dataset_split[1] * len(self.dataset))
            test_size = len(self.dataset) - train_size - val_size
            
            self.dataset_train, self.dataset_val, self.dataset_test = random_split(
                self.dataset,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            # Create subset datasets
            self.dataset_train = self.dataset.copy(self.dataset_train.indices)
            self.dataset_val = self.dataset.copy(self.dataset_val.indices)
            self.dataset_test = self.dataset.copy(self.dataset_test.indices)
            
            # Normalize using train statistics
            if normalize:
                self.dataset_train.compute_normalization()
                self.dataset_val.apply_normalization(self.dataset_train.mean, self.dataset_train.std)
                self.dataset_test.apply_normalization(self.dataset_train.mean, self.dataset_train.std)
        else:
            # Two-way split (train/val only)
            self.dataset_train, self.dataset_val = self.dataset.random_split(self.dataset_split)
            if normalize:
                self.dataset_train.compute_normalization()
                self.dataset_val.apply_normalization(self.dataset_train.mean, self.dataset_train.std)
            self.dataset_test = None
        
        # Create dataloaders
        self.train = DataLoader(
            self.dataset_train,
            batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True
        )
        self.val = DataLoader(
            self.dataset_val,
            batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=True
        )
        
        if self.dataset_test is not None:
            self.test = DataLoader(
                self.dataset_test,
                batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=True
            )
    
    def get_feature_groups(self):
        """Returns feature groups by semantic meaning"""
        return {
            'elevation': [0],
            'aspect': [1],
            'slope': [2],
            'hydrology_distance': [3, 4],
            'roadways_distance': [5],
            'hillshade': [6, 7, 8],
            'firepoints_distance': [9],
            'wilderness_area': list(range(10, 14)),  # 4 binary
            'soil_type': list(range(14, 54))  # 40 binary
        }


class CovertypeDataset(Dataset):
    """Covertype Dataset with normalization support"""
    
    def __init__(self,
                 root='./data',
                 normalize=True,
                 device=None,
                 indices=None,
                 transforms=None):
        super().__init__()
        
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.normalize = normalize
        self.indices = indices
        self.transforms = transforms
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Load data
        self._load_data()
        
        # Normalization stats (computed later)
        self.mean = None
        self.std = None
    
    def _load_data(self):
        """Load covertype data using sklearn"""
        from sklearn.datasets import fetch_covtype
        
        print("Loading Covertype dataset...")
        data = fetch_covtype(data_home=str(self.root), download_if_missing=True)
        
        # Convert to tensors
        self.x = torch.from_numpy(data.data.astype(np.float32))
        self.y = torch.from_numpy(data.target.astype(np.int64)) - 1  # Convert from 1-7 to 0-6
        
        # Apply indices if provided
        if self.indices is not None:
            self.x = self.x[self.indices]
            self.y = self.y[self.indices]
    
    def compute_normalization(self):
        """Compute mean and std for continuous features only (first 10 features)"""
        self.mean = self.x[:, :10].mean(dim=0)
        self.std = self.x[:, :10].std(dim=0)
        if self.normalize:
            self.x[:, :10] = (self.x[:, :10] - self.mean) / (self.std + 1e-8)
    
    def apply_normalization(self, mean, std):
        """Apply normalization to continuous features using provided statistics"""
        if self.normalize:
            self.mean = mean
            self.std = std
            self.x[:, :10] = (self.x[:, :10] - mean) / (std + 1e-8)
    
    def copy(self, indices=None):
        """Create a copy with subset of indices"""
        return CovertypeDataset(
            root=str(self.root),
            normalize=self.normalize,
            device=self.device,
            indices=indices,
            transforms=self.transforms
        )
    
    def random_split(self, split):
        """Split dataset randomly"""
        subsets = random_split(self, split, generator=torch.Generator().manual_seed(42))
        ds_subs = []
        for subset in subsets:
            ds_sub = self.copy(subset.indices)
            ds_subs.append(ds_sub)
        return ds_subs
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        if self.transforms is not None:
            x = self.transforms.apply(self.x[idx])
            return x, self.y[idx]
        return self.x[idx], self.y[idx]

