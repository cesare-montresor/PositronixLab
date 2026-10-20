# positronixlab/dataprovider/scan_dataprovider.py
import torch
import os
from torch.utils.data import Dataset, random_split
from positronixlab.dataprovider.dataprovider import DataProvider

class SCANDataProvider(DataProvider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'scan'
        self.num_features = None  # Sequence-based, depends on embedding dim
        self.num_classes = None   # Target vocab size
        
        self.dataset = SCANDataset(transforms=self.transforms)
        self.dataset_train, self.dataset_val = self.dataset.random_split(self.dataset_split)
        self.dataset_train.transforms = self.transforms
        self.dataset_val.transforms = self.transforms
        self.dataset_test = SCANDataset(train=False, transforms=self.transforms)

class SCANDataset(Dataset):
    def __init__(self, train=True, indices=None, transforms=None):
        super().__init__()
        self.train = train
        self.indices = indices
        self.transforms = transforms
        
        # NOTE: Implement your tokenization/loading logic here.
        # This assumes a pre-processed tensor of shape (N, max_seq_length)
        file_path = './data/scan/train.pt' if train else './data/scan/test.pt'
        
        if os.path.exists(file_path):
            data = torch.load(file_path)
            self.x = data['inputs']
            self.y = data['targets']
        else:
            # Fallback dummies to prevent crash before data is downloaded
            self.x = torch.zeros((100, 20), dtype=torch.long)
            self.y = torch.zeros((100, 30), dtype=torch.long)

        if indices is not None:
            self.x = self.x[indices]
            self.y = self.y[indices]
            
    def copy(self, indices=None):
        return SCANDataset(self.train, indices=indices, transforms=self.transforms)

    def random_split(self, split):
        subsets = random_split(self, split)
        return [self.copy(subset.indices) for subset in subsets]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.transforms is not None:
            x = self.transforms.apply(self.x[idx])
            return x, self.y[idx]
        return self.x[idx], self.y[idx]