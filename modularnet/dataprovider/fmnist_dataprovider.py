
### fashion mnist ###

class FashionMnistDataProvider(BaseDataProvider):
    def __init__(self, 
                 batch_size=32,
                 split=None,
                 device=None,
                 num_workers=None,
                 transforms=None,
                 ):
        super().__init__(
            batch_size=batch_size, 
            split=split,
            device=device,
            num_workers=num_workers
        )
        self.name = 'fmnist'
        self.transforms = transforms
        self.num_features = 784
        self.num_classes = 10
        self.num_workers = num_workers

        self.dataset = FashionMnistDataset(transforms=self.transforms)
        self.dataset_train, self.dataset_val = self.dataset.random_split(self.dataset_split)
        self.dataset_train.transforms = self.transforms
        self.dataset_val.transforms = self.transforms
        self.dataset_test = FashionMnistDataset(train=False, device=self.device, transforms=self.transforms)

        self.train = DataLoader(self.dataset_train, batch_size, True, num_workers=num_workers, drop_last=True)
        self.val = DataLoader(self.dataset_val, batch_size, False, num_workers=num_workers, drop_last=True)
        self.test = DataLoader(self.dataset_test, batch_size, False, num_workers=num_workers, drop_last=True)


class FashionMnistDataset(Dataset):

    def __init__(self, train=True, device=None, indices=None, transforms=None):
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
        #y = torch.nn.functional.one_hot(y,self.num_class)
        #self.y = y.to(dtype=torch.float)
        
        
        if device is None: self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else: self.device = device
        
    def copy(self, indices=None):
        return FashionMnistDataset(self.train, self.device, indices=indices, transforms=self.transforms)

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
        #return self.x[idx].to(self.device), self.y[idx].to(self.device)
    