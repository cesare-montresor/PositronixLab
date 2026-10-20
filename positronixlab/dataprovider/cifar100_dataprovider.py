# positronixlab/dataprovider/cifar100_dataprovider.py
import torch
from torchvision import datasets
from torch.utils.data import Dataset, random_split
from positronixlab.dataprovider.dataprovider import DataProvider

class CIFAR100DataProvider(DataProvider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'cifar100'
        self.num_features = 3072
        self.num_classes = 100

        self.dataset = CIFAR100Dataset(transforms=self.transforms)
        self.dataset_train, self.dataset_val = self.dataset.random_split(self.dataset_split)
        self.dataset_train.transforms = self.transforms
        self.dataset_val.transforms = self.transforms
        self.dataset_test = CIFAR100Dataset(train=False, transforms=self.transforms)

class CIFAR100Dataset(Dataset):
    def __init__(self, train=True, indices=None, transforms=None):
        super().__init__()
        self.train = train
        self.indices = indices
        self.ds = datasets.CIFAR100('./data', train=train, download=True)
        self.name_class = self.ds.classes
        self.num_class = len(self.name_class)
        
        raw_tensor = torch.tensor(self.ds.data, dtype=torch.float).permute(0, 3, 1, 2)
        self.x = raw_tensor / 255.0
        self.y = torch.tensor(self.ds.targets, dtype=torch.long)

        if indices is not None:
            self.x = self.x[indices]
            self.y = self.y[indices]

        self.transforms = transforms
        
    def copy(self, indices=None):
        return CIFAR100Dataset(self.train, indices=indices, transforms=self.transforms)

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