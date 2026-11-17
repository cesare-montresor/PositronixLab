from positronixlab.pipeline.pipeline_config import PipelineConfig
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
    def from_config(config: PipelineConfig, split=None, num_workers=None, indexed_dataset=False) -> DataProvider:
        name = config.dataset_name
        batch_size = config.batch_size
        flatten = config.model_type == 'fc'

        kwargs = {
            "batch_size": batch_size,
            "split": split,
            "num_workers": num_workers,
            "indexed_dataset": indexed_dataset,
        }

        if name == 'iris':
            return IrisDataProvider(**kwargs)
        elif name == 'mnist':
            transforms = [] 
            if flatten: transforms.append('flatten')
            t = MetaTransform(transforms)
            kwargs["transforms"] = t
            return MnistDataProvider(**kwargs)
        elif name == 'fashion_mnist':
            transforms = [] 
            if flatten: transforms.append('flatten')
            t = MetaTransform(transforms)
            kwargs["transforms"] = t
            return FashionMnistDataProvider(**kwargs)
        elif name == 'covertype':
            return CovertypeDataProvider(**kwargs)
        elif name == 'higgs':
            return HIGGSDataProvider(**kwargs)
        elif name == 'cifar10':
            transforms = []
            if flatten: transforms.append('flatten')
            t = MetaTransform(transforms)
            kwargs["transforms"] = t
            return CIFAR10DataProvider(**kwargs)
        elif name == 'cifar100':
            transforms = []
            if flatten: transforms.append('flatten')
            t = MetaTransform(transforms)
            kwargs["transforms"] = t
            return CIFAR100DataProvider(**kwargs)
        elif name == 'scan':
            return SCANDataProvider(**kwargs)
        elif name == 'deepmind_math':
            return MathDataProvider(**kwargs)
        elif name == 'cvr_odd_one_out':
            transforms = []
            if flatten: transforms.append('flatten')
            t = MetaTransform(transforms)
            kwargs["transforms"] = t
            return CVROddOneOutDataProvider(**kwargs)
        elif name == 'syneval_odd_one_out':
            return SynevalOddOneOutDataProvider(**kwargs)
        else:
            raise ValueError(f"Unknown dataset: {name}")