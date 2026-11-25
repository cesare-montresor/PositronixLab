"""DataProvider package with lazy exports and full static-typing autocomplete."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from positronixlab.baselines import DataProviderLoader
    from positronixlab.dataprovider.addmul_dataprovider import (
        AddMulDataProvider,
        AddMulDataset,
        AddMulLoss,
        addmul_exact_accuracy,
    )
    from positronixlab.dataprovider.cifar10_dataprovider import (
        CIFAR10DataProvider,
        CIFAR10Dataset,
    )
    from positronixlab.dataprovider.cifar100_dataprovider import (
        CIFAR100DataProvider,
        CIFAR100Dataset,
    )
    from positronixlab.dataprovider.class_removal_dataprovider import (
        ClassRemovalDataProvider,
    )
    from positronixlab.dataprovider.config import DataConfig, DatasetName
    from positronixlab.dataprovider.covertype_dataprovider import (
        CovertypeDataProvider,
        CovertypeDataset,
    )
    from positronixlab.dataprovider.cvr_oddoneout_dataprovider import (
        CVR_FAMILIES,
        CVRFamily,
        CVROddOneOutDataProvider,
        CVROddOneOutDataset,
    )
    from positronixlab.dataprovider.dataprovider import (
        BaseTensorDataset,
        CompositionalDataProvider,
        DataProvider,
        DataProviderOptions,
        DataReader,
        EpochSubsampleSampler,
        IndexedDataset,
        SubtaskSplit,
        apply_transform,
        resolve_data_root,
    )
    from positronixlab.dataprovider.dataprovider_factory import DataProviderFactory
    from positronixlab.dataprovider.fmnist_dataprovider import (
        FashionMnistDataProvider,
        FashionMnistDataset,
    )
    from positronixlab.dataprovider.higgs_dataprovider import (
        HIGGSDataProvider,
        HIGGSDataset,
    )
    from positronixlab.dataprovider.iris_dataprovider import (
        IrisDataProvider,
        IrisDataset,
    )
    from positronixlab.dataprovider.math_dataprovider import (
        MathDataProvider,
        MathDataset,
    )
    from positronixlab.dataprovider.mnist_dataprovider import (
        MnistDataProvider,
        MnistDataset,
    )
    from positronixlab.dataprovider.oddoneout_dataprovider import (
        OddOneOutDataProvider,
        OddOneOutDataset,
    )
    from positronixlab.dataprovider.permuted_mnist_dataprovider import (
        PermutedMnistDataProvider,
        PermutedMnistDataset,
    )
    from positronixlab.dataprovider.scan_dataprovider import (
        SCANDataProvider,
        SCANDataset,
    )
    from positronixlab.dataprovider.syneval_oddoneout_dataprovider import (
        SYNEVAL_FAMILIES,
        SynEvalFamily,
        SynevalOddOneOutDataProvider,
        SynevalOddOneOutDataset,
    )
    from positronixlab.dataprovider.tuple_arithmetic_dataprovider import (
        TupleArithmeticDataProvider,
        TupleArithmeticDataset,
        TupleArithmeticLoss,
        tuple_arithmetic_accuracy,
    )

_LAZY_EXPORTS = {
    "DataConfig": ("positronixlab.dataprovider.config", "DataConfig"),
    "DatasetName": ("positronixlab.dataprovider.config", "DatasetName"),
    "BaseTensorDataset": ("positronixlab.dataprovider.dataprovider", "BaseTensorDataset"),
    "CompositionalDataProvider": ("positronixlab.dataprovider.dataprovider", "CompositionalDataProvider"),
    "SubtaskSplit": ("positronixlab.dataprovider.dataprovider", "SubtaskSplit"),
    "DataProvider": ("positronixlab.dataprovider.dataprovider", "DataProvider"),
    "DataProviderOptions": ("positronixlab.dataprovider.dataprovider", "DataProviderOptions"),
    "DataReader": ("positronixlab.dataprovider.dataprovider", "DataReader"),
    "EpochSubsampleSampler": ("positronixlab.dataprovider.dataprovider", "EpochSubsampleSampler"),
    "IndexedDataset": ("positronixlab.dataprovider.dataprovider", "IndexedDataset"),
    "apply_transform": ("positronixlab.dataprovider.dataprovider", "apply_transform"),
    "resolve_data_root": ("positronixlab.dataprovider.dataprovider", "resolve_data_root"),
    "DataProviderFactory": ("positronixlab.dataprovider.dataprovider_factory", "DataProviderFactory"),
    "CIFAR10DataProvider": ("positronixlab.dataprovider.cifar10_dataprovider", "CIFAR10DataProvider"),
    "CIFAR10Dataset": ("positronixlab.dataprovider.cifar10_dataprovider", "CIFAR10Dataset"),
    "CIFAR100DataProvider": ("positronixlab.dataprovider.cifar100_dataprovider", "CIFAR100DataProvider"),
    "CIFAR100Dataset": ("positronixlab.dataprovider.cifar100_dataprovider", "CIFAR100Dataset"),
    "ClassRemovalDataProvider": ("positronixlab.dataprovider.class_removal_dataprovider", "ClassRemovalDataProvider"),
    "CovertypeDataProvider": ("positronixlab.dataprovider.covertype_dataprovider", "CovertypeDataProvider"),
    "CovertypeDataset": ("positronixlab.dataprovider.covertype_dataprovider", "CovertypeDataset"),
    "CVR_FAMILIES": ("positronixlab.dataprovider.cvr_oddoneout_dataprovider", "CVR_FAMILIES"),
    "CVRFamily": ("positronixlab.dataprovider.cvr_oddoneout_dataprovider", "CVRFamily"),
    "CVROddOneOutDataProvider": ("positronixlab.dataprovider.cvr_oddoneout_dataprovider", "CVROddOneOutDataProvider"),
    "CVROddOneOutDataset": ("positronixlab.dataprovider.cvr_oddoneout_dataprovider", "CVROddOneOutDataset"),
    "FashionMnistDataProvider": ("positronixlab.dataprovider.fmnist_dataprovider", "FashionMnistDataProvider"),
    "FashionMnistDataset": ("positronixlab.dataprovider.fmnist_dataprovider", "FashionMnistDataset"),
    "HIGGSDataProvider": ("positronixlab.dataprovider.higgs_dataprovider", "HIGGSDataProvider"),
    "HIGGSDataset": ("positronixlab.dataprovider.higgs_dataprovider", "HIGGSDataset"),
    "IrisDataProvider": ("positronixlab.dataprovider.iris_dataprovider", "IrisDataProvider"),
    "IrisDataset": ("positronixlab.dataprovider.iris_dataprovider", "IrisDataset"),
    "MathDataProvider": ("positronixlab.dataprovider.math_dataprovider", "MathDataProvider"),
    "MathDataset": ("positronixlab.dataprovider.math_dataprovider", "MathDataset"),
    "MnistDataProvider": ("positronixlab.dataprovider.mnist_dataprovider", "MnistDataProvider"),
    "MnistDataset": ("positronixlab.dataprovider.mnist_dataprovider", "MnistDataset"),
    "OddOneOutDataProvider": ("positronixlab.dataprovider.oddoneout_dataprovider", "OddOneOutDataProvider"),
    "OddOneOutDataset": ("positronixlab.dataprovider.oddoneout_dataprovider", "OddOneOutDataset"),
    "SCANDataProvider": ("positronixlab.dataprovider.scan_dataprovider", "SCANDataProvider"),
    "SCANDataset": ("positronixlab.dataprovider.scan_dataprovider", "SCANDataset"),
    "SYNEVAL_FAMILIES": ("positronixlab.dataprovider.syneval_oddoneout_dataprovider", "SYNEVAL_FAMILIES"),
    "SynEvalFamily": ("positronixlab.dataprovider.syneval_oddoneout_dataprovider", "SynEvalFamily"),
    "SynevalOddOneOutDataProvider": ("positronixlab.dataprovider.syneval_oddoneout_dataprovider", "SynevalOddOneOutDataProvider"),
    "SynevalOddOneOutDataset": ("positronixlab.dataprovider.syneval_oddoneout_dataprovider", "SynevalOddOneOutDataset"),
    "AddMulDataProvider": ("positronixlab.dataprovider.addmul_dataprovider", "AddMulDataProvider"),
    "AddMulDataset": ("positronixlab.dataprovider.addmul_dataprovider", "AddMulDataset"),
    "AddMulLoss": ("positronixlab.dataprovider.addmul_dataprovider", "AddMulLoss"),
    "addmul_exact_accuracy": ("positronixlab.dataprovider.addmul_dataprovider", "addmul_exact_accuracy"),
    "TupleArithmeticDataProvider": ("positronixlab.dataprovider.tuple_arithmetic_dataprovider", "TupleArithmeticDataProvider"),
    "TupleArithmeticDataset": ("positronixlab.dataprovider.tuple_arithmetic_dataprovider", "TupleArithmeticDataset"),
    "TupleArithmeticLoss": ("positronixlab.dataprovider.tuple_arithmetic_dataprovider", "TupleArithmeticLoss"),
    "tuple_arithmetic_accuracy": ("positronixlab.dataprovider.tuple_arithmetic_dataprovider", "tuple_arithmetic_accuracy"),
    "PermutedMnistDataProvider": ("positronixlab.dataprovider.permuted_mnist_dataprovider", "PermutedMnistDataProvider"),
    "PermutedMnistDataset": ("positronixlab.dataprovider.permuted_mnist_dataprovider", "PermutedMnistDataset"),
    "DataProviderLoader": ("positronixlab.baselines", "DataProviderLoader"),
}

__all__ = sorted(_LAZY_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_path, attr = _LAZY_EXPORTS[name]
        module = __import__(module_path, fromlist=[attr])
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
