from torchvision import datasets

from modularnet.dataprovider.base_dataprovider import BaseDataProvider
from torch.utils.data import Dataset, random_split
import torch

from torch import nn

### mnist ###


class MnistDataProvider(BaseDataProvider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.name = 'mnist'
        self.num_features = 784
        self.num_classes = 10
        self.criterion = nn.CrossEntropyLoss()

        self.dataset = MnistDataset(transforms=self.transforms)
        self.dataset_train, self.dataset_val = self.dataset.random_split(self.dataset_split)
        self.dataset_train.transforms = self.transforms
        self.dataset_val.transforms = self.transforms
        self.dataset_test = MnistDataset(train=False, transforms=self.transforms)


class MnistDataset(Dataset):
    
    def __init__(self, train=True, indices=None, transforms=None):
        super().__init__()
        self.train=train
        self.indices=indices
        self.ds = datasets.MNIST('./data', train=train, download=True)
        self.name_class = self.ds.classes
        self.num_class = len(self.name_class)
        
        self.x = self.ds.data.to(dtype=torch.float) / 255
        self.y = self.ds.targets

        if indices is not None:
            self.x = self.x[indices]
            self.y = self.y[indices]

        #y = torch.nn.functional.one_hot(y,self.num_class)
        #self.y = y.to(dtype=torch.float)
        self.transforms = transforms
        
    def copy(self, indices=None):
        return MnistDataset(self.train, indices=indices)

    def random_split(self, split):
        subsets = random_split(self, split)
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
    
