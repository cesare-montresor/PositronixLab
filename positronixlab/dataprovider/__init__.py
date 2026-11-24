"""DataProvider package with lazy exports to avoid eager ``parallelazy`` coupling."""

from __future__ import annotations

from typing import Any

_LAZY_EXPORTS = {
    "DataConfig": ("positronixlab.dataprovider.config", "DataConfig"),
    "DataProvider": ("positronixlab.dataprovider.dataprovider", "DataProvider"),
    "DataReader": ("positronixlab.dataprovider.dataprovider", "DataReader"),
    "IndexedDataset": ("positronixlab.dataprovider.dataprovider", "IndexedDataset"),
    "DataProviderFactory": ("positronixlab.dataprovider.dataprovider_factory", "DataProviderFactory"),
    "CovertypeDataProvider": ("positronixlab.dataprovider.covertype_dataprovider", "CovertypeDataProvider"),
    "FashionMnistDataProvider": ("positronixlab.dataprovider.fmnist_dataprovider", "FashionMnistDataProvider"),
    "HIGGSDataProvider": ("positronixlab.dataprovider.higgs_dataprovider", "HIGGSDataProvider"),
    "IrisDataProvider": ("positronixlab.dataprovider.iris_dataprovider", "IrisDataProvider"),
    "MnistDataProvider": ("positronixlab.dataprovider.mnist_dataprovider", "MnistDataProvider"),
    "AddMulDataProvider": ("positronixlab.dataprovider.addmul_dataprovider", "AddMulDataProvider"),
    "AddMulDataset": ("positronixlab.dataprovider.addmul_dataprovider", "AddMulDataset"),
    "TupleArithmeticDataProvider": ("positronixlab.dataprovider.tuple_arithmetic_dataprovider", "TupleArithmeticDataProvider"),
    "TupleArithmeticDataset": ("positronixlab.dataprovider.tuple_arithmetic_dataprovider", "TupleArithmeticDataset"),
    "PermutedMnistDataProvider": ("positronixlab.dataprovider.permuted_mnist_dataprovider", "PermutedMnistDataProvider"),
    "PermutedMnistDataset": ("positronixlab.dataprovider.permuted_mnist_dataprovider", "PermutedMnistDataset"),
    "CompositionalDataProvider": ("positronixlab.dataprovider.sota", "CompositionalDataProvider"),
    "DataProviderLoader": ("positronixlab.dataprovider.sota", "DataProviderLoader"),
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