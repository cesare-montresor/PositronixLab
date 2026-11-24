"""Base abstractions for compositional SOTA modularity benchmarks.

Intentionally independent of ``positronixlab.pipeline`` / ``parallelazy`` so
benchmarks can run standalone or inside a full pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from torch.utils.data import DataLoader, Dataset


@dataclass
class SubtaskSplit:
    """Train/val/test datasets & loaders for a single subtask or evaluation split."""

    name: str
    dataset_train: Optional[Dataset] = None
    dataset_val: Optional[Dataset] = None
    dataset_test: Optional[Dataset] = None
    batch_size: int = 64
    num_workers: int = 0

    def train_loader(
        self, batch_size: Optional[int] = None, shuffle: bool = True
    ) -> Optional[DataLoader]:
        if self.dataset_train is None:
            return None
        return DataLoader(
            self.dataset_train,
            batch_size=batch_size or self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

    def val_loader(self, batch_size: Optional[int] = None) -> Optional[DataLoader]:
        if self.dataset_val is None:
            return None
        return DataLoader(
            self.dataset_val,
            batch_size=batch_size or self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_loader(self, batch_size: Optional[int] = None) -> Optional[DataLoader]:
        if self.dataset_test is None:
            return None
        return DataLoader(
            self.dataset_test,
            batch_size=batch_size or self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


@dataclass(frozen=True)
class SOTADatasetSpec:
    """Registry metadata for a SOTA benchmark dataset."""

    name: str
    source: str  # "csordas" | "lepori"
    task_type: str  # "classification" | "odd_one_out" | "arithmetic"
    builder: Callable[..., "CompositionalDataProvider"]
    description: str = ""


class CompositionalDataProvider:
    """Base provider exposing a base task plus named compositional subtasks.

    Supports both Csordas-style subtask restrictions (``full``, ``tuple_0``,
    ``tuple_1``; ``full``, ``add``, ``mul``; ``all``, ``without_c``) and
    Lepori-style compositional ablation families (``base``, ``sub_a``, ``sub_b``,
    ``both_ooo``, ``ablate_a``, ``ablate_b``).
    """

    source: str = "custom"
    task_type: str = "classification"

    def __init__(
        self,
        name: str,
        batch_size: int = 64,
        num_workers: int = 0,
    ) -> None:
        self.name = name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_features: Optional[int] = None
        self.num_classes: Optional[int] = None
        self.subtasks: Dict[str, SubtaskSplit] = {}
        self._eval_matrix: Dict[str, List[str]] = {}

    def list_subtasks(self) -> List[str]:
        return list(self.subtasks.keys())

    def get_subtask(self, name: str) -> SubtaskSplit:
        if name not in self.subtasks:
            raise KeyError(
                f"Unknown subtask {name!r} for {self.name!r}. "
                f"Available: {self.list_subtasks()}"
            )
        return self.subtasks[name]

    def eval_matrix_spec(self) -> Dict[str, List[str]]:
        """Map each discoverable subroutine name to the subtask splits to evaluate on."""
        return dict(self._eval_matrix)

    @property
    def base_split(self) -> SubtaskSplit:
        for key in ("full", "base", "all"):
            if key in self.subtasks:
                return self.subtasks[key]
        return next(iter(self.subtasks.values()))

    def train_loader(self, batch_size: Optional[int] = None, shuffle: bool = True) -> DataLoader:
        loader = self.base_split.train_loader(batch_size=batch_size, shuffle=shuffle)
        if loader is None:
            raise ValueError(f"{self.name!r} base split has no train dataset.")
        return loader

    def val_loader(self, batch_size: Optional[int] = None) -> DataLoader:
        loader = self.base_split.val_loader(batch_size=batch_size)
        if loader is None:
            raise ValueError(f"{self.name!r} base split has no val dataset.")
        return loader

    def test_loader(self, batch_size: Optional[int] = None) -> DataLoader:
        loader = self.base_split.test_loader(batch_size=batch_size)
        if loader is None:
            raise ValueError(f"{self.name!r} base split has no test dataset.")
        return loader
