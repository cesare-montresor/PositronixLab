
import torch


class BaseDataProvider():
    def __init__(self, 
                 batch_size=32,
                 split=None,
                 device = None,
                 num_workers=None,
                 transforms=None,
                 ):
        self.dataset_split = split if split is not None else [0.7, 0.3]
        self.num_features = None
        self.num_classes = None
        self.num_workers = 2 if num_workers is None else num_workers
        self.transforms = transforms
        
        self.x_shape = None
        self.y_shape = None

        self.dataset = None
        
        self.batch_size = batch_size
        self.train = None
        self.val = None
        self.test = None
        
        self.device = device
        if self.device is None: 
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
