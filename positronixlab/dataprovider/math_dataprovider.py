# positronixlab/dataprovider/math_dataprovider.py
import torch
import os
from torch.utils.data import Dataset, random_split
from positronixlab.dataprovider.dataprovider import DataProvider

class MathDataProvider(DataProvider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'deepmind_math'
        self.num_features = None 
        self.num_classes = None  
        
        self.dataset = MathDataset(transforms=self.transforms)
        self.dataset_train, self.dataset_val = self.dataset.random_split(self.dataset_split)
        self.dataset_train.transforms = self.transforms
        self.dataset_val.transforms = self.transforms
        self.dataset_test = MathDataset(train=False, transforms=self.transforms)

class MathDataset(Dataset):
    def __init__(self, train=True, indices=None, transforms=None):
        super().__init__()
        self.train = train
        self.indices = indices
        self.transforms = transforms
        
        # NOTE: Requires a tokenized representation of the math text files.
        file_path = './data/math/train.pt' if train else './data/math/test.pt'
        
        if os.path.exists(file_path):
            data = torch.load(file_path)
            self.x = data['inputs']
            self.y = data['targets']
        else:
            self.x = torch.zeros((100, 50), dtype=torch.long)
            self.y = torch.zeros((100, 10), dtype=torch.long)

        if indices is not None:
            self.x = self.x[indices]
            self.y = self.y[indices]
            
    def copy(self, indices=None):
        return MathDataset(self.train, indices=indices, transforms=self.transforms)

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