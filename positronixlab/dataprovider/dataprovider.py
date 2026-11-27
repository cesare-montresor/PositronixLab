"""Core DataProvider, DataReader, Dataset view, and Sampler abstractions."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Literal,
    Protocol,
    TypedDict,
    TypeVar,
    runtime_checkable,
)

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, random_split

SplitName = Literal["train", "val", "test"]


@runtime_checkable
class _ApplyTransform(Protocol):
    def apply(self, x: Any) -> Any: ...


TransformLike = Callable[[Any], Any] | _ApplyTransform


class DataProviderOptions(TypedDict, total=False):
    """Shared keyword options accepted by all ``DataProvider`` subclasses.

    Used with ``typing.Unpack[DataProviderOptions]`` so IDEs provide full
    parameter autocomplete on every concrete provider without duplicating
    signatures.
    """

    batch_size: int
    split: Sequence[float] | None
    num_workers: int | None
    transforms: TransformLike | None
    indexed_dataset: bool
    root: str | Path | None
    seed: int
    shuffle_train: bool
    drop_last: bool
    persistent_workers: bool
    pin_memory: bool


def resolve_data_root(root: str | Path | None = None) -> Path:
    """Resolve dataset storage root directory (explicit *root* or default ``./data``)."""
    return Path(root) if root is not None else Path("./data")


def apply_transform(transforms: TransformLike | None, x: Any) -> Any:
    """Apply either a standard callable transform or a legacy ``.apply()`` transform."""
    if transforms is None:
        return x
    if callable(transforms):
        return transforms(x)
    return transforms.apply(x)


class IndexedDataset(Dataset):
    """Wrap any ``Dataset`` so ``__getitem__`` appends the sample index."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> tuple[Any, ...]:
        item = self.dataset[idx]
        if isinstance(item, tuple):
            return (*item, idx)
        return (item, idx)


class BaseTensorDataset(Dataset):
    """Zero-copy tensor-backed dataset with index views, lazy normalization, and transforms.

    Splitting via :meth:`copy` or :meth:`random_split` shares the underlying
    ``_full_x`` / ``_full_y`` tensors instead of reloading data from disk.
    """

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        indices: Sequence[int] | np.ndarray | torch.Tensor | None = None,
        transforms: TransformLike | None = None,
        normalize: bool = False,
        norm_cols: slice | None = None,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
        transform_per_element: bool = False,
        name_class: Sequence[Any] | None = None,
    ) -> None:
        super().__init__()
        self._full_x = x
        self._full_y = y
        self.indices: np.ndarray | None = (
            np.asarray(indices, dtype=np.int64) if indices is not None else None
        )
        self.transforms = transforms
        self.normalize = normalize
        self.norm_cols = norm_cols
        self.mean = mean
        self.std = std
        self.transform_per_element = transform_per_element
        self.name_class = list(name_class) if name_class is not None else None
        self.num_class = len(self.name_class) if self.name_class is not None else None

    @property
    def x(self) -> torch.Tensor:
        if self.indices is None:
            return self._full_x
        return self._full_x[self.indices]

    @property
    def y(self) -> torch.Tensor:
        if self.indices is None:
            return self._full_y
        return self._full_y[self.indices]

    def compute_normalization(self, max_samples: int | None = None, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-feature mean/std without mutating backing tensors in-place."""
        n = len(self)
        if max_samples is not None and n > max_samples:
            rng = np.random.default_rng(seed)
            local_idx = np.sort(rng.choice(n, size=max_samples, replace=False))
            real_idx = self.indices[local_idx] if self.indices is not None else local_idx
            sample = self._full_x[real_idx].float()
        else:
            sample = self.x.float()

        cols = sample[:, self.norm_cols] if self.norm_cols is not None else sample
        self.mean = cols.mean(dim=0)
        self.std = cols.std(dim=0)
        return self.mean, self.std

    def apply_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Attach precomputed mean/std for lazy normalization in ``__getitem__``."""
        self.mean = mean
        self.std = std

    def copy(
        self,
        indices: Sequence[int] | np.ndarray | torch.Tensor | None = None,
        *,
        transforms: Any = ...,
    ) -> BaseTensorDataset:
        """Create a zero-copy view sharing the same backing tensors."""
        if indices is None:
            new_indices = self.indices
        elif self.indices is None:
            new_indices = np.asarray(indices, dtype=np.int64)
        else:
            new_indices = self.indices[np.asarray(indices, dtype=np.int64)]

        return BaseTensorDataset(
            self._full_x,
            self._full_y,
            indices=new_indices,
            transforms=self.transforms if transforms is ... else transforms,
            normalize=self.normalize,
            norm_cols=self.norm_cols,
            mean=self.mean,
            std=self.std,
            transform_per_element=self.transform_per_element,
            name_class=self.name_class,
        )

    def random_split(
        self,
        split: Sequence[float],
        *,
        seed: int | None = 42,
    ) -> list[BaseTensorDataset]:
        """Split into zero-copy views using ``torch.utils.data.random_split``."""
        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        subsets = random_split(self, list(split), generator=generator)
        return [self.copy(subset.indices) for subset in subsets]

    def __len__(self) -> int:
        if self.indices is not None:
            return int(len(self.indices))
        return int(len(self._full_x))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        real_idx = int(self.indices[idx]) if self.indices is not None else idx
        x = self._full_x[real_idx]
        y = self._full_y[real_idx]

        if self.normalize and self.mean is not None and self.std is not None:
            x = x.float().clone()
            if self.norm_cols is not None:
                x[self.norm_cols] = (x[self.norm_cols] - self.mean) / (self.std + 1e-8)
            else:
                x = (x - self.mean) / (self.std + 1e-8)

        if self.transforms is not None:
            if self.transform_per_element:
                x = torch.stack([apply_transform(self.transforms, elem) for elem in x])
            else:
                x = apply_transform(self.transforms, x)

        return x, y


class EpochSubsampleSampler(Sampler[int]):
    """Deterministic per-epoch subsampling sampler compatible with ``persistent_workers=True``.

    Because the sampler runs in the main process and feeds fresh indices to
    worker queues each epoch, calling :meth:`set_epoch` updates what persistent
    workers fetch without respawning workers or mutating dataset state.
    """

    def __init__(
        self,
        pool_size: int,
        epoch_size: int,
        *,
        base_seed: int = 42,
        shuffle: bool = True,
        sort_for_locality: bool = False,
    ) -> None:
        super().__init__()
        if epoch_size < 1 or epoch_size > pool_size:
            raise ValueError(f"epoch_size must be in [1, {pool_size}], got {epoch_size}")
        self.pool_size = int(pool_size)
        self.epoch_size = int(epoch_size)
        self.base_seed = int(base_seed)
        self.shuffle = bool(shuffle)
        self.sort_for_locality = bool(sort_for_locality)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.base_seed + self._epoch)
        chosen = rng.choice(self.pool_size, size=self.epoch_size, replace=False)
        if self.sort_for_locality and not self.shuffle:
            chosen = np.sort(chosen)
        elif self.shuffle:
            rng.shuffle(chosen)
        return iter(chosen.tolist())

    def __len__(self) -> int:
        return self.epoch_size


TReader = TypeVar("TReader", bound="DataReader")


class DataReader(DataLoader):
    """PyTorch ``DataLoader`` with CUDA-aware multiprocessing defaults and plugin registry."""

    _registry: dict[str, type[DataReader]] = {}

    def __init_subclass__(
        cls,
        plugin_name: str | None = None,
        override: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init_subclass__(**kwargs)
        key = plugin_name or cls.__name__
        if key in cls._registry and not override:
            raise ValueError(f"Plugin '{key}' is already registered.")
        cls._registry[key] = cls

    @classmethod
    def get(cls, name: str) -> type[DataReader]:
        if name not in cls._registry:
            raise ValueError(
                f"Unknown DataReader '{name}'. Available: {cls.list_registered()}"
            )
        return cls._registry[name]

    @classmethod
    def list_registered(cls) -> list[str]:
        return sorted(cls._registry.keys())

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        num_workers: int = 2,
        persistent_workers: bool = True,
        multiprocessing_context: str | None = None,
        drop_last: bool = True,
        indexed_dataset: bool = False,
        **kwargs: Any,
    ) -> None:
        if multiprocessing_context is None and num_workers > 0:
            multiprocessing_context = "fork" if not torch.cuda.is_initialized() else "spawn"

        super().__init__(
            dataset=dataset if not indexed_dataset else IndexedDataset(dataset),
            batch_size=batch_size,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            multiprocessing_context=multiprocessing_context if num_workers > 0 else None,
            drop_last=drop_last,
            num_workers=num_workers,
            **kwargs,
        )


class DataProvider:
    """Base dataset provider managing train/val/test datasets, lazy ``DataReader``s, and registry."""

    _registry: dict[str, type[DataProvider]] = {}
    _aliases: dict[str, str] = {}
    _plugin_name: str = ""

    default_split: tuple[float, ...] = (0.7, 0.3)
    supports_flatten: bool = False

    def __init_subclass__(
        cls,
        plugin_name: str | None = None,
        aliases: Sequence[str] = (),
        override: bool = False,
        register: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init_subclass__(**kwargs)
        if not register:
            return
        key = plugin_name or cls.__name__
        if key in cls._registry and not override:
            raise ValueError(f"Plugin '{key}' is already registered.")
        cls._registry[key] = cls
        cls._plugin_name = key
        for alias in aliases:
            if alias in cls._registry and cls._registry[alias] is not cls and not override:
                raise ValueError(f"Alias '{alias}' collides with registered plugin.")
            cls._aliases[alias] = key

    @classmethod
    def get(cls, name: str) -> type[DataProvider]:
        canonical = cls._aliases.get(name, name)
        if canonical not in cls._registry:
            raise ValueError(
                f"Unknown dataset: {name}. Available: {cls.list_registered()}"
            )
        return cls._registry[canonical]

    @classmethod
    def list_registered(cls) -> list[str]:
        return sorted(cls._registry.keys())

    def __init__(
        self,
        batch_size: int = 32,
        split: Sequence[float] | None = None,
        num_workers: int | None = None,
        transforms: TransformLike | None = None,
        indexed_dataset: bool = False,
        root: str | Path | None = None,
        seed: int = 42,
        shuffle_train: bool = True,
        drop_last: bool = True,
        persistent_workers: bool = True,
        pin_memory: bool = False,
    ) -> None:
        self.dataset_split: list[float] = (
            list(split) if split is not None else list(self.default_split)
        )
        self.num_features: int | None = None
        self.num_classes: int | None = None
        self.num_workers: int = 2 if num_workers is None else num_workers
        self.transforms: TransformLike | None = transforms
        self.indexed_dataset: bool = indexed_dataset
        self.root: Path = resolve_data_root(root)
        self.seed: int = seed
        self.shuffle_train: bool = shuffle_train
        self.drop_last: bool = drop_last
        self.persistent_workers: bool = persistent_workers
        self.pin_memory: bool = pin_memory

        self.name: str = getattr(self, "_plugin_name", self.__class__.__name__)
        self.criterion: torch.nn.Module | None = None
        self.x_shape: tuple[int, ...] | None = None
        self.y_shape: tuple[int, ...] | None = None

        self.dataset: Dataset | None = None
        self.dataset_train: Dataset | None = None
        self.dataset_val: Dataset | None = None
        self.dataset_test: Dataset | None = None

        self.batch_size: int = batch_size
        self._train: DataReader | None = None
        self._val: DataReader | None = None
        self._test: DataReader | None = None

    def _split_dataset(
        self,
        dataset: Dataset,
        *,
        test_dataset: Dataset | None = None,
        split: Sequence[float] | None = None,
        seed: int | None = None,
    ) -> None:
        """Populate ``dataset_train``, ``dataset_val``, and ``dataset_test`` for 2- or 3-way splits."""
        ratios = list(split) if split is not None else self.dataset_split
        use_seed = self.seed if seed is None else seed

        if len(ratios) not in (2, 3):
            raise ValueError(f"split must have 2 or 3 entries, got {ratios!r}")

        parts: Sequence[Dataset]
        if isinstance(dataset, BaseTensorDataset):
            parts = dataset.random_split(ratios, seed=use_seed)
        else:
            generator = torch.Generator().manual_seed(use_seed) if use_seed is not None else None
            parts = list(random_split(dataset, ratios, generator=generator))

        self.dataset_train = parts[0]
        self.dataset_val = parts[1]
        if len(parts) == 3:
            self.dataset_test = parts[2]
        else:
            self.dataset_test = test_dataset

    def set_epoch(self, epoch: int) -> None:
        """Optional per-epoch lifecycle hook (e.g. for subsampled datasets like HIGGS)."""
        return None

    def get_loader(
        self,
        split: SplitName,
        *,
        batch_size: int | None = None,
        shuffle: bool | None = None,
        drop_last: bool | None = None,
        num_workers: int | None = None,
        indexed_dataset: bool | None = None,
        persistent_workers: bool | None = None,
        pin_memory: bool | None = None,
        sampler: Sampler | None = None,
        **kwargs: Any,
    ) -> DataReader | None:
        """Build a fresh ``DataReader`` for *split* with optional per-call overrides."""
        dataset_map = {
            "train": self.dataset_train,
            "val": self.dataset_val,
            "test": self.dataset_test,
        }
        if split not in dataset_map:
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")
        ds = dataset_map[split]
        if ds is None:
            return None

        use_shuffle = (self.shuffle_train if split == "train" else False) if shuffle is None else shuffle
        if sampler is not None:
            use_shuffle = False

        return DataReader(
            ds,
            batch_size=self.batch_size if batch_size is None else batch_size,
            num_workers=self.num_workers if num_workers is None else num_workers,
            persistent_workers=(
                self.persistent_workers if persistent_workers is None else persistent_workers
            ),
            drop_last=self.drop_last if drop_last is None else drop_last,
            indexed_dataset=(
                self.indexed_dataset if indexed_dataset is None else indexed_dataset
            ),
            shuffle=use_shuffle,
            pin_memory=self.pin_memory if pin_memory is None else pin_memory,
            sampler=sampler,
            **kwargs,
        )

    @property
    def train(self) -> DataReader | None:
        if self._train is None and self.dataset_train is not None:
            self._train = self.get_loader("train")
        return self._train

    @property
    def val(self) -> DataReader | None:
        if self._val is None and self.dataset_val is not None:
            self._val = self.get_loader("val")
        return self._val

    @property
    def test(self) -> DataReader | None:
        if self._test is None and self.dataset_test is not None:
            self._test = self.get_loader("test")
        return self._test


@dataclass
class SubtaskSplit:
    """Train/val/test datasets & ``DataReader`` builders for a single compositional subtask."""

    name: str
    dataset_train: Dataset | None = None
    dataset_val: Dataset | None = None
    dataset_test: Dataset | None = None
    batch_size: int = 64
    num_workers: int = 0
    drop_last: bool = True
    indexed_dataset: bool = False
    persistent_workers: bool = True
    pin_memory: bool = False

    def train_loader(
        self,
        batch_size: int | None = None,
        shuffle: bool = True,
        drop_last: bool | None = None,
        indexed_dataset: bool | None = None,
    ) -> DataReader | None:
        if self.dataset_train is None:
            return None
        return DataReader(
            self.dataset_train,
            batch_size=self.batch_size if batch_size is None else batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last if drop_last is None else drop_last,
            indexed_dataset=(
                self.indexed_dataset if indexed_dataset is None else indexed_dataset
            ),
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )

    def val_loader(
        self,
        batch_size: int | None = None,
        drop_last: bool | None = None,
        indexed_dataset: bool | None = None,
    ) -> DataReader | None:
        if self.dataset_val is None:
            return None
        return DataReader(
            self.dataset_val,
            batch_size=self.batch_size if batch_size is None else batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last if drop_last is None else drop_last,
            indexed_dataset=(
                self.indexed_dataset if indexed_dataset is None else indexed_dataset
            ),
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )

    def test_loader(
        self,
        batch_size: int | None = None,
        drop_last: bool | None = None,
        indexed_dataset: bool | None = None,
    ) -> DataReader | None:
        if self.dataset_test is None:
            return None
        return DataReader(
            self.dataset_test,
            batch_size=self.batch_size if batch_size is None else batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last if drop_last is None else drop_last,
            indexed_dataset=(
                self.indexed_dataset if indexed_dataset is None else indexed_dataset
            ),
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )


class CompositionalDataProvider(DataProvider, register=False):
    """``DataProvider`` exposing a base task plus named compositional subtasks & eval matrix."""

    source: str = "custom"
    task_type: str = "classification"

    def __init__(
        self,
        name: str | None = None,
        batch_size: int = 64,
        num_workers: int | None = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            num_workers=0 if num_workers is None else num_workers,
            **kwargs,
        )
        if name is not None:
            self.name = name
        self.subtasks: dict[str, SubtaskSplit] = {}
        self._eval_matrix: dict[str, list[str]] = {}

    def _sync_base_split(self) -> None:
        """Bind ``dataset_train``/``dataset_val``/``dataset_test`` to :attr:`base_split`."""
        if not self.subtasks:
            return
        base = self.base_split
        self.dataset_train = base.dataset_train
        self.dataset_val = base.dataset_val
        self.dataset_test = base.dataset_test
        self.dataset = base.dataset_train
        self._train = None
        self._val = None
        self._test = None

    def list_subtasks(self) -> list[str]:
        return list(self.subtasks.keys())

    def get_subtask(self, name: str) -> SubtaskSplit:
        if name not in self.subtasks:
            raise KeyError(
                f"Unknown subtask {name!r} for {self.name!r}. "
                f"Available: {self.list_subtasks()}"
            )
        return self.subtasks[name]

    def eval_matrix_spec(self) -> dict[str, list[str]]:
        """Map each discoverable subroutine name to the subtask splits to evaluate on."""
        return dict(self._eval_matrix)

    @property
    def base_split(self) -> SubtaskSplit:
        for key in ("full", "base", "all"):
            if key in self.subtasks:
                return self.subtasks[key]
        return next(iter(self.subtasks.values()))

    def train_loader(
        self,
        batch_size: int | None = None,
        shuffle: bool = True,
        drop_last: bool | None = None,
    ) -> DataReader:
        loader = self.base_split.train_loader(
            batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
        )
        if loader is None:
            raise ValueError(f"{self.name!r} base split has no train dataset.")
        return loader

    def val_loader(
        self,
        batch_size: int | None = None,
        drop_last: bool | None = None,
    ) -> DataReader:
        loader = self.base_split.val_loader(batch_size=batch_size, drop_last=drop_last)
        if loader is None:
            raise ValueError(f"{self.name!r} base split has no val dataset.")
        return loader

    def test_loader(
        self,
        batch_size: int | None = None,
        drop_last: bool | None = None,
    ) -> DataReader:
        loader = self.base_split.test_loader(batch_size=batch_size, drop_last=drop_last)
        if loader is None:
            raise ValueError(f"{self.name!r} base split has no test dataset.")
        return loader

