from positronixlab.trial.trial_config import TrialConfig
from positronixlab.dataprovider.covertype_dataprovider import CovertypeDataProvider
from positronixlab.dataprovider.fmnist_dataprovider import FashionMnistDataProvider
from positronixlab.dataprovider.higgs_dataprovider import HIGGSDataProvider
from positronixlab.dataprovider.iris_dataprovider import IrisDataProvider
from positronixlab.dataprovider.mnist_dataprovider import MnistDataProvider
from positronixlab.dataprovider.cifar10_dataprovider import CIFAR10DataProvider
from positronixlab.dataprovider.cifar100_dataprovider import CIFAR100DataProvider
from positronixlab.dataprovider.scan_dataprovider import SCANDataProvider
from positronixlab.dataprovider.math_dataprovider import MathDataProvider
from positronixlab.dataprovider.cvr_oddoneout_dataprovider import CVROddOneOutDataProvider
from positronixlab.dataprovider.syneval_oddoneout_dataprovider import SynevalOddOneOutDataProvider
from positronixlab.metamodel.metatransform import MetaTransform
from positronixlab.dataprovider.dataprovider import DataProvider




class DataProviderFactory:

    @staticmethod
    def from_config(config: TrialConfig, split=None, num_workers=None) -> DataProvider:
        name = config.dataset_name
        batch_size = config.batch_size
        flatten = config.model_type == 'fc'

        if name == 'iris':
            return IrisDataProvider(batch_size=batch_size, split=split, num_workers=num_workers)
        elif name == 'mnist':
            transforms = [] 
            if flatten: transforms.append('flatten')
            t = MetaTransform(transforms)
            return MnistDataProvider(batch_size=batch_size, split=split, transforms=t, num_workers=num_workers)
        elif name == 'fashion_mnist':
            transforms = [] 
            if flatten: transforms.append('flatten')
            t = MetaTransform(transforms)
            return FashionMnistDataProvider(batch_size=batch_size, split=split, transforms=t, num_workers=num_workers)
        elif name == 'covertype':
            return CovertypeDataProvider(batch_size=batch_size, split=split, num_workers=num_workers)
        elif name == 'higgs':
            return HIGGSDataProvider(batch_size=batch_size, split=split, num_workers=num_workers)
        elif name == 'cifar10':
            transforms = []
            if flatten: transforms.append('flatten')
            t = MetaTransform(transforms)
            return CIFAR10DataProvider(batch_size=batch_size, split=split, transforms=t, num_workers=num_workers)
        elif name == 'cifar100':
            transforms = []
            if flatten: transforms.append('flatten')
            t = MetaTransform(transforms)
            return CIFAR100DataProvider(batch_size=batch_size, split=split, transforms=t, num_workers=num_workers)
        elif name == 'scan':
            return SCANDataProvider(batch_size=batch_size, split=split, num_workers=num_workers)
        elif name == 'deepmind_math':
            return MathDataProvider(batch_size=batch_size, split=split, num_workers=num_workers)
        elif name == 'cvr_odd_one_out':
            transforms = []
            if flatten: transforms.append('flatten')
            t = MetaTransform(transforms)
            return CVROddOneOutDataProvider(batch_size=batch_size, split=split, transforms=t, num_workers=num_workers)
        elif name == 'syneval_odd_one_out':
            return SynevalOddOneOutDataProvider(batch_size=batch_size, split=split, num_workers=num_workers)
        else:
            raise ValueError(f"Unknown dataset: {name}")