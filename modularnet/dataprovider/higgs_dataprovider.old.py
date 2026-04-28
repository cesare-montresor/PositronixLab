
# =============================================================================
# HIGGS Dataset
# =============================================================================

from modularnet.dataprovider.base_dataprovider import BaseDataProvider


class HIGGSDataProviderOld(BaseDataProvider):
    """
    HIGGS Dataset Provider for binary classification.
    
    Features:
    - 21 low-level features (raw detector measurements)
    - 7 high-level features (physics-derived quantities)
    - Total: 28 features
    - 11M samples (use subset_size for smaller experiments)
    - Binary classification: signal (1) vs background (0)
    
    Args:
        batch_size: Batch size for dataloaders
        split: Train/val/test split ratios (default [0.7, 0.15, 0.15])
        device: Device to use
        num_workers: Number of dataloader workers
        transforms: Optional transforms
        normalize: Whether to normalize features
        subset_size: Use only this many samples (for testing)
        root: Root directory for data
    """
    def __init__(self,
                 batch_size=32,
                 split=None,
                 device=None,
                 num_workers=None,
                 transforms=None,
                 normalize=True,
                 subset_size=None,
                 root='./data'):
        super().__init__(
            batch_size=batch_size,
            split=split if split is not None else [0.7, 0.15, 0.15],
            device=device,
            num_workers=num_workers,
            transforms=transforms
        )
        
        self.name = 'higgs'
        self.num_features = 28
        self.num_classes = 2
        self.x_shape = (28,)
        self.y_shape = ()
        self.normalize = normalize
        self.root = Path(root)
        
        # Create dataset
        self.dataset = HIGGSDataset(
            root=root,
            normalize=normalize,
            subset_size=subset_size,
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
        """Returns feature groups: low-level (0-20) and high-level (21-27)"""
        return {
            'low_level': list(range(21)),
            'high_level': list(range(21, 28))
        }


class HIGGSDatasetOld(Dataset):
    """HIGGS Dataset with lazy loading and normalization support"""
    
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
    
    def __init__(self,
                 root='./data',
                 normalize=True,
                 subset_size=None,
                 device=None,
                 indices=None,
                 transforms=None):
        super().__init__()
        
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.normalize = normalize
        self.subset_size = subset_size
        self.indices = indices
        self.transforms = transforms
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # File paths
        self.raw_file = self.root / "HIGGS.csv.gz"
        self.processed_file = self.root / "HIGGS.csv"
        self.npy_file = self.root / "HIGGS.npy"
        
        # Download if needed
        if not self.processed_file.exists() and not self.npy_file.exists():
            self._download()
        
        # Load data
        self._load_data()
        
        # Normalization stats (computed later)
        self.mean = None
        self.std = None
    
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
    
    def _load_data(self):
        """Load data from CSV or NPY"""
        # Try to load from npy (faster)
        if self.npy_file.exists():
            print(f"Loading HIGGS from {self.npy_file}...")
            data = np.load(self.npy_file)
            if self.subset_size:
                data = data[:self.subset_size]
        else:
            print(f"Loading HIGGS from {self.processed_file}...")
            columns = ['label'] + [f'feature_{i}' for i in range(28)]
            
            if self.subset_size:
                df = pd.read_csv(self.processed_file, names=columns, nrows=self.subset_size)
            else:
                df = pd.read_csv(self.processed_file, names=columns)
            
            data = df.values
            # Save as npy for faster loading next time
            np.save(self.npy_file, data)
        
        # Split features and labels
        self.x = torch.from_numpy(data[:, 1:].astype(np.float32))
        self.y = torch.from_numpy(data[:, 0].astype(np.int64))
        
        # Apply indices if provided
        if self.indices is not None:
            self.x = self.x[self.indices]
            self.y = self.y[self.indices]
    
    def compute_normalization(self):
        """Compute mean and std for normalization (call on training set)"""
        self.mean = self.x.mean(dim=0)
        self.std = self.x.std(dim=0)
        if self.normalize:
            self.x = (self.x - self.mean) / (self.std + 1e-8)
    
    def apply_normalization(self, mean, std):
        """Apply normalization using provided statistics (for val/test sets)"""
        if self.normalize:
            self.mean = mean
            self.std = std
            self.x = (self.x - mean) / (std + 1e-8)
    
    def copy(self, indices=None):
        """Create a copy with subset of indices"""
        return HIGGSDataset(
            root=str(self.root),
            normalize=self.normalize,
            subset_size=self.subset_size,
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

