# modularnet/dataprovider/syneval_oddoneout_dataprovider.py
import torch
import os
from torch.utils.data import Dataset, random_split
from modularnet.dataprovider.base_dataprovider import BaseDataProvider

class SynevalOddOneOutDataProvider(BaseDataProvider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'syneval_odd_one_out'
        self.num_features = None 
        self.num_classes = 4  
        
        self.dataset = SynevalOddOneOutDataset(transforms=self.transforms)
        self.dataset_train, self.dataset_val = self.dataset.random_split(self.dataset_split)
        
        self.dataset_train.transforms = self.transforms
        self.dataset_val.transforms = self.transforms
        self.dataset_test = SynevalOddOneOutDataset(train=False, transforms=self.transforms)

class SynevalOddOneOutDataset(Dataset):
    def __init__(self, train=True, indices=None, transforms=None):
        super().__init__()
        self.train = train
        self.indices = indices
        self.transforms = transforms
        
        # Expected input shape: (N, 4, max_seq_len)
        # Expected target shape: (N,) containing indices 0-3
        file_path = './data/syneval_odd_one_out/train.pt' if train else './data/syneval_odd_one_out/test.pt'
        
        if os.path.exists(file_path):
            data = torch.load(file_path)
            self.x = data['inputs']
            self.y = data['targets']
        else:
            self.x = torch.zeros((100, 4, 32), dtype=torch.long)
            self.y = torch.randint(0, 4, (100,), dtype=torch.long)

        if indices is not None:
            self.x = self.x[indices]
            self.y = self.y[indices]
            
    def copy(self, indices=None):
        return SynevalOddOneOutDataset(self.train, indices=indices, transforms=self.transforms)

    def random_split(self, split):
        subsets = random_split(self, split)
        return [self.copy(subset.indices) for subset in subsets]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        
        if self.transforms is not None:
            try:
                x = torch.stack([self.transforms.apply(seq) for seq in x])
            except AttributeError:
                x = torch.stack([self.transforms(seq) for seq in x])
                
        return x, self.y[idx]