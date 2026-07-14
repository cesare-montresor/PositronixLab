
from modularnet.trial.trial_config import TrialConfig
from modularnet.dataprovider.covertype_dataprovider import CovertypeDataProvider
from modularnet.dataprovider.fmnist_dataprovider import FashionMnistDataProvider
from modularnet.dataprovider.higgs_dataprovider import HIGGSDataProvider
from modularnet.dataprovider.iris_dataprovider import IrisDataProvider
from modularnet.dataprovider.mnist_dataprovider import MnistDataProvider
from modularnet.metamodel.metatransform import MetaTransform


class DataProvider:

    @staticmethod
    def from_config(config: TrialConfig, device=None, split=None, num_workers=None):
        name = config.dataset_name
        bs = config.batch_size
        flatten = config.model_type == 'fc'

        if name == 'iris':
            return IrisDataProvider(bs, device=device, split=split, num_workers=num_workers)
        elif name == 'mnist':
            transforms = [] # ['augment_image'] 
            if flatten: transforms.append('flatten')
            t = MetaTransform(transforms)
            return MnistDataProvider(bs, device=device, split=split, transforms=t, num_workers=num_workers)
        elif name == 'fashion_mnist':
            transforms = [] #['augment_image'] 
            if flatten: transforms.append('flatten')
            t = MetaTransform(transforms)
            return FashionMnistDataProvider(bs, device=device, split=split, transforms=t, num_workers=num_workers   )
        elif name == 'covertype':
            return CovertypeDataProvider(bs, device=device, split=split, num_workers=num_workers)
        elif name == 'higgs':
            return HIGGSDataProvider(bs, device=device, split=split, num_workers=num_workers)
        else:
            raise ValueError(f"Unknown dataset: {name}")

