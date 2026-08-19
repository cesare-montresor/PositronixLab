import torch
import os
from torch.utils.data import Dataset, random_split
from modularnet.dataprovider.dataprovider import DataProvider

class OddOneOutDataProvider(DataProvider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'odd_one_out'
        self.num_features = None 
        self.num_classes = 4  # Targets are indices 0, 1, 2, or 3
        
        self.dataset = OddOneOutDataset(transforms=self.transforms)
        
        # Subset splits utilizing the base architecture
        self.dataset_train, self.dataset_val = self.dataset.random_split(self.dataset_split)
        self.dataset_train.transforms = self.transforms
        self.dataset_val.transforms = self.transforms
        self.dataset_test = OddOneOutDataset(train=False, transforms=self.transforms)

class OddOneOutDataset(Dataset):
    def __init__(self, train=True, indices=None, transforms=None):
        super().__init__()
        self.train = train
        self.indices = indices
        self.transforms = transforms
        
        # NOTE: Adjust the path to your pre-processed tensor location.
        # Expected input shape: (N, 4, C, H, W)
        # Expected target shape: (N,)
        file_path = './data/odd_one_out/train.pt' if train else './data/odd_one_out/test.pt'
        
        if os.path.exists(file_path):
            data = torch.load(file_path)
            self.x = data['inputs']
            self.y = data['targets']
        else:
            # Fallback dummy tensors to prevent crashes before data generation/download
            self.x = torch.zeros((100, 4, 3, 64, 64), dtype=torch.float)
            self.y = torch.randint(0, 4, (100,), dtype=torch.long)

        if indices is not None:
            self.x = self.x[indices]
            self.y = self.y[indices]
            
    def copy(self, indices=None):
        return OddOneOutDataset(self.train, indices=indices, transforms=self.transforms)

    def random_split(self, split):
        subsets = random_split(self, split)
        return [self.copy(subset.indices) for subset in subsets]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        
        if self.transforms is not None:
            # Applies the transform sequentially to each of the 4 images in the sample
            try:
                x = torch.stack([self.transforms.apply(img) for img in x])
            except AttributeError:
                # Fallback if self.transforms is a standard callable rather than MetaTransform
                x = torch.stack([self.transforms(img) for img in x])
                
        return x, self.y[idx]