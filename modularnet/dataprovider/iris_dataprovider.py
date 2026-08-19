from modularnet.dataprovider.dataprovider import DataProvider
from torch.utils.data import Dataset
import torch
from torch import nn
import sklearn.datasets as sklearn_datasets

### iris ###

class IrisDataProvider(DataProvider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'iris'
        self.num_features = 4
        self.num_classes = 3
        
        self.criterion = nn.CrossEntropyLoss()
        self.dataset = IrisDataset()
        self.dataset_train, self.dataset_val, self.dataset_test = self.dataset, self.dataset, self.dataset # 150 items 

class IrisDataset(Dataset):
    
    def __init__(self):
        self.ds = iris = sklearn_datasets.load_iris()
        self.name_class = iris['target_names']
        self.num_class = len(self.name_class)
        
        self.x = torch.FloatTensor( iris['data'] )
        y = torch.LongTensor(iris['target'])
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    

