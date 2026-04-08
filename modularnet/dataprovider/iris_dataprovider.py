
### iris ###

class IrisDataProvider(BaseDataProvider):
    def __init__(self, 
                 batch_size=32,
                 device=None,
                 split=None, 
                 num_workers=None):
        super().__init__(batch_size=batch_size, device=device, num_workers=num_workers, split=split)
        self.name = 'iris'
        self.num_features = 4
        self.num_classes = 3
        
        self.criterion = nn.CrossEntropyLoss()
        self.dataset = IrisDataset(device=device)

        self.train = DataLoader(self.dataset,batch_size,True,num_workers=num_workers, drop_last=True)
        self.val = DataLoader(self.dataset,batch_size,False,num_workers=num_workers, drop_last=True)

class IrisDataset(Dataset):
    
    def __init__(self):
        self.ds = iris = sklearn_datasets.load_iris()
        self.name_class = iris['target_names']
        self.num_class = len(self.name_class)
        
        self.x = torch.FloatTensor( iris['data'] )
        y = torch.LongTensor(iris['target'])
        self.y = y
        #y = torch.nn.functional.one_hot(y,self.num_class)
        #self.y = y.to(dtype=torch.float)
        


    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    

