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
        load_cvr_png_strip,
    )
    from positronixlab.dataprovider.cvr_simclr_dataprovider import (
        CVR_SIMCLR_BASE_TASKS,
        ContrastiveTransformations,
        CVRSimCLRDataProvider,
        CVRSimCLRDataset,
        build_simclr_transforms,
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
    from positronixlab.dataprovider.fast_loader import (
        FastTensorLoader,
        make_fast_loader,
        materialize_dataset_tensors,
    )
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
        CSORDAS_DM_MATH_PAPER_TASKS,
        DM_MATH_EXTRAPOLATE_MAP,
        DM_MATH_TASKS,
        HF_DM_MATH_REPO,
        MathDataProvider,
        MathDataset,
        MathSeq2SeqLoss,
        MathVocab,
        download_dm_math,
        download_dm_math_file,
        math_exact_accuracy,
        math_token_accuracy,
    )
    from positronixlab.dataprovider.mnist_dataprovider import (
        MnistDataProvider,
        MnistDataset,
    )
    from positronixlab.dataprovider.permuted_mnist_dataprovider import (
        PermutedMnistDataProvider,
        PermutedMnistDataset,
    )
    from positronixlab.dataprovider.scan_dataprovider import (
        SCAN_URLS,
        SCANDataProvider,
        SCANDataset,
        SCANSplitType,
        SCANVocab,
        download_scan_file,
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
    "FastTensorLoader": ("positronixlab.dataprovider.fast_loader", "FastTensorLoader"),
    "make_fast_loader": ("positronixlab.dataprovider.fast_loader", "make_fast_loader"),
    "materialize_dataset_tensors": ("positronixlab.dataprovider.fast_loader", "materialize_dataset_tensors"),
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
    "load_cvr_png_strip": ("positronixlab.dataprovider.cvr_oddoneout_dataprovider", "load_cvr_png_strip"),
    "CVR_SIMCLR_BASE_TASKS": ("positronixlab.dataprovider.cvr_simclr_dataprovider", "CVR_SIMCLR_BASE_TASKS"),
    "ContrastiveTransformations": ("positronixlab.dataprovider.cvr_simclr_dataprovider", "ContrastiveTransformations"),
    "CVRSimCLRDataProvider": ("positronixlab.dataprovider.cvr_simclr_dataprovider", "CVRSimCLRDataProvider"),
    "CVRSimCLRDataset": ("positronixlab.dataprovider.cvr_simclr_dataprovider", "CVRSimCLRDataset"),
    "build_simclr_transforms": ("positronixlab.dataprovider.cvr_simclr_dataprovider", "build_simclr_transforms"),
    "FashionMnistDataProvider": ("positronixlab.dataprovider.fmnist_dataprovider", "FashionMnistDataProvider"),
    "FashionMnistDataset": ("positronixlab.dataprovider.fmnist_dataprovider", "FashionMnistDataset"),
    "HIGGSDataProvider": ("positronixlab.dataprovider.higgs_dataprovider", "HIGGSDataProvider"),
    "HIGGSDataset": ("positronixlab.dataprovider.higgs_dataprovider", "HIGGSDataset"),
    "IrisDataProvider": ("positronixlab.dataprovider.iris_dataprovider", "IrisDataProvider"),
    "IrisDataset": ("positronixlab.dataprovider.iris_dataprovider", "IrisDataset"),
    "CSORDAS_DM_MATH_PAPER_TASKS": ("positronixlab.dataprovider.math_dataprovider", "CSORDAS_DM_MATH_PAPER_TASKS"),
    "DM_MATH_EXTRAPOLATE_MAP": ("positronixlab.dataprovider.math_dataprovider", "DM_MATH_EXTRAPOLATE_MAP"),
    "DM_MATH_TASKS": ("positronixlab.dataprovider.math_dataprovider", "DM_MATH_TASKS"),
    "HF_DM_MATH_REPO": ("positronixlab.dataprovider.math_dataprovider", "HF_DM_MATH_REPO"),
    "MathDataProvider": ("positronixlab.dataprovider.math_dataprovider", "MathDataProvider"),
    "MathDataset": ("positronixlab.dataprovider.math_dataprovider", "MathDataset"),
    "MathSeq2SeqLoss": ("positronixlab.dataprovider.math_dataprovider", "MathSeq2SeqLoss"),
    "MathVocab": ("positronixlab.dataprovider.math_dataprovider", "MathVocab"),
    "download_dm_math": ("positronixlab.dataprovider.math_dataprovider", "download_dm_math"),
    "download_dm_math_file": ("positronixlab.dataprovider.math_dataprovider", "download_dm_math_file"),
    "math_exact_accuracy": ("positronixlab.dataprovider.math_dataprovider", "math_exact_accuracy"),
    "math_token_accuracy": ("positronixlab.dataprovider.math_dataprovider", "math_token_accuracy"),
    "MnistDataProvider": ("positronixlab.dataprovider.mnist_dataprovider", "MnistDataProvider"),
    "MnistDataset": ("positronixlab.dataprovider.mnist_dataprovider", "MnistDataset"),
    "SCAN_URLS": ("positronixlab.dataprovider.scan_dataprovider", "SCAN_URLS"),
    "SCANDataProvider": ("positronixlab.dataprovider.scan_dataprovider", "SCANDataProvider"),
    "SCANDataset": ("positronixlab.dataprovider.scan_dataprovider", "SCANDataset"),
    "SCANSplitType": ("positronixlab.dataprovider.scan_dataprovider", "SCANSplitType"),
    "SCANVocab": ("positronixlab.dataprovider.scan_dataprovider", "SCANVocab"),
    "download_scan_file": ("positronixlab.dataprovider.scan_dataprovider", "download_scan_file"),
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
