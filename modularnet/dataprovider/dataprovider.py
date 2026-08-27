
from typing import List, Self, TypeVar

import torch


from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x,y = self.dataset[idx]
        return x, y, idx


class DataReader(DataLoader):
    _registry: dict[str, type[Self]] = {}
    
    def __init_subclass__(cls, plugin_name: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        
        # Determine registry key: custom plugin_name or default class name
        key = plugin_name or cls.__name__
        
        if key in cls._registry:
            raise ValueError(f"Plugin '{key}' is already registered.")
            
        cls._registry[key] = cls

    def __init__(self, 
        dataset: Dataset,
        batch_size: int = 32,
        num_workers: int = 2,
        persistent_workers: bool = True,
        multiprocessing_context: str = None,
        drop_last=True,
        indexed_dataset: bool = False,
        **kwargs
    ):
        if multiprocessing_context is None and num_workers > 0:
            multiprocessing_context = "fork" if not torch.cuda.is_initialized() else "spawn"

        

        super().__init__(
            dataset=dataset if not indexed_dataset else IndexedDataset(dataset),
            batch_size=batch_size,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            multiprocessing_context=multiprocessing_context if num_workers > 0 else None,
            drop_last=drop_last,
            num_workers=num_workers,
            **kwargs
        )

class DataProvider():
    # Registry dictionary to store subclasses by their name
    _registry: dict[str, type[Self]] = {}

    def __init_subclass__(cls, plugin_name: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        
        # Determine registry key: custom plugin_name or default class name
        key = plugin_name or cls.__name__
        
        if key in cls._registry:
            raise ValueError(f"Plugin '{key}' is already registered.")
            
        cls._registry[key] = cls

    def __init__(self, 
                 batch_size:int =32,
                 split: List[float]=None,
                 num_workers:int=None,
                 transforms: transforms.Compose =None,
                 indexed_dataset: bool = False,
                 ):
        self.dataset_split = split if split is not None else [0.7, 0.3]
        self.num_features = None
        self.num_classes = None
        self.num_workers = 2 if num_workers is None else num_workers
        self.transforms = transforms
        self.indexed_dataset = indexed_dataset

        self.x_shape = None
        self.y_shape = None

        self.dataset: Dataset = None
        self.dataset_train: Dataset = None
        self.dataset_val: Dataset = None
        self.dataset_test: Dataset = None
        

        self.batch_size = batch_size
        self._train:DataReader = None
        self._val:DataReader = None
        self._test:DataReader = None
        
        
    @property
    def train(self) -> DataReader:
        if self._train is None and self.dataset_train is not None:
            self._train = DataReader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers, indexed_dataset=self.indexed_dataset)
        return self._train
    
    @property
    def val(self) -> DataReader:
        if self._val is None and self.dataset_val is not None:
            self._val = DataReader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers, indexed_dataset=self.indexed_dataset)
        return self._val
    
    @property
    def test(self) -> DataReader:
        if self._test is None and self.dataset_test is not None:
            self._test = DataReader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers, indexed_dataset=self.indexed_dataset)
        return self._test


