
from torchvision import datasets

from modularnet.dataprovider.dataprovider import DataProvider
from torch.utils.data import Dataset, random_split
import torch

### fashion mnist ###

class FashionMnistDataProvider(DataProvider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'fmnist'
        self.num_features = 784
        self.num_classes = 10

        self.dataset = FashionMnistDataset(transforms=self.transforms)
        self.dataset_train, self.dataset_val = self.dataset.random_split(self.dataset_split)
        self.dataset_train.transforms = self.transforms
        self.dataset_val.transforms = self.transforms
        self.dataset_test = FashionMnistDataset(train=False, transforms=self.transforms)


class FashionMnistDataset(Dataset):

    def __init__(self, train=True, indices=None, transforms=None):
        super().__init__()
        self.train=train
        self.indices=indices
        self.ds = datasets.FashionMNIST('./data', train=train, download=True)
        self.name_class = self.ds.classes
        self.num_class = len(self.name_class)
        
        self.x = self.ds.data.to(dtype=torch.float) / 255
        self.y = self.ds.targets

        if indices is not None:
            self.x = self.x[indices]
            self.y = self.y[indices]

        self.transforms = transforms
        
        
        
    def copy(self, indices=None):
        return FashionMnistDataset(self.train, indices=indices, transforms=self.transforms)

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
    