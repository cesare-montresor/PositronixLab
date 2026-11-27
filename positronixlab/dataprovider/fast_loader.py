"""In-memory, device-resident tensor batching to eliminate DataLoader overhead."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset

from positronixlab.tasks.discoverer import unpack_batch


def materialize_dataset_tensors(dataset: Dataset) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Fast vectorized extraction of ``(X, Y)`` tensors from known datasets.

    Returns ``None`` if the dataset requires generic iteration (e.g., transforms
    must be applied per sample or the dataset type is unknown).
    """
    from positronixlab.dataprovider.addmul_dataprovider import AddMulDataset, _split_digits
    from positronixlab.dataprovider.tuple_arithmetic_dataprovider import TupleArithmeticDataset

    # 1. Vectorized AddMulDataset
    if isinstance(dataset, AddMulDataset):
        if getattr(dataset, "transforms", None) is not None:
            return None
        arg_digits = torch.from_numpy(_split_digits(dataset.n_digits, dataset.args)).long()
        onehot_args = F.one_hot(arg_digits, num_classes=10).reshape(len(dataset), -1).float()
        onehot_op = F.one_hot(torch.from_numpy(dataset.ops.astype(np.int64)), num_classes=2).float()
        x = torch.cat([onehot_args, onehot_op], dim=1)
        if dataset.target_mode == "integer":
            y = torch.from_numpy(dataset.res.astype(np.int64))
        else:
            y = torch.from_numpy(_split_digits(dataset.n_digits, dataset.res)).long()
        return x, y

    # 2. Vectorized TupleArithmeticDataset
    if isinstance(dataset, TupleArithmeticDataset):
        if getattr(dataset, "transforms", None) is not None:
            return None
        in_digits = torch.from_numpy(_split_digits(dataset.n_digits, dataset.inputs)).long()
        onehot_in = F.one_hot(in_digits, num_classes=10).float()  # (N, T, 2, D, 10)
        if dataset.restrict_tuple is not None:
            mask = torch.zeros(1, dataset.n_tuples, 1, 1, 1, dtype=onehot_in.dtype)
            mask[0, dataset.restrict_tuple] = 1.0
            onehot_in = onehot_in * mask
        x = onehot_in.reshape(len(dataset), -1)
        if dataset.target_mode == "integer":
            t_idx = dataset.restrict_tuple if dataset.restrict_tuple is not None else 0
            y = torch.from_numpy(dataset.outputs[:, t_idx].astype(np.int64))
        else:
            y = torch.from_numpy(_split_digits(dataset.n_digits, dataset.outputs)).long()
        return x, y

    # 3. Subset wrapping a fast dataset or BaseTensorDataset
    if isinstance(dataset, Subset):
        base = dataset.dataset
        idxs = torch.as_tensor(dataset.indices, dtype=torch.long)
        if (
            hasattr(base, "x")
            and hasattr(base, "y")
            and isinstance(base.x, torch.Tensor)
            and isinstance(base.y, torch.Tensor)
        ):
            if getattr(base, "transforms", None) is None:
                return base.x[idxs].clone(), base.y[idxs].clone()
            return None
        parent_xy = materialize_dataset_tensors(base)
        if parent_xy is not None:
            return parent_xy[0][idxs].clone(), parent_xy[1][idxs].clone()

    # 4. BaseTensorDataset (e.g. MNIST, PermutedMNIST, Iris, FashionMNIST)
    if (
        hasattr(dataset, "x")
        and hasattr(dataset, "y")
        and isinstance(dataset.x, torch.Tensor)
        and isinstance(dataset.y, torch.Tensor)
    ):
        if getattr(dataset, "transforms", None) is None:
            return dataset.x.clone(), dataset.y.clone()

    return None


class FastTensorLoader:
    """In-memory, device-resident batch iterator eliminating PyTorch ``DataLoader`` overhead."""

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        batch_size: int,
        shuffle: bool = True,
        device: torch.device | str | None = None,
        max_steps: int | None = None,
        subtask: str | None = None,
    ) -> None:
        self.device = torch.device(device) if device is not None else x.device
        self.x = x.to(self.device, non_blocking=True).contiguous()
        self.y = y.to(self.device, non_blocking=True).contiguous()
        self.batch_size = max(1, int(batch_size))
        self.shuffle = shuffle
        self.max_steps = max_steps
        self.subtask = subtask
        self.n_samples = int(self.x.shape[0])

    def __len__(self) -> int:
        n_batches = (self.n_samples + self.batch_size - 1) // self.batch_size
        return min(n_batches, self.max_steps) if self.max_steps else n_batches

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        if self.shuffle:
            perm = torch.randperm(self.n_samples, device=self.device)
            x_src = self.x.index_select(0, perm)
            y_src = self.y.index_select(0, perm)
        else:
            x_src = self.x
            y_src = self.y

        steps = 0
        for start in range(0, self.n_samples, self.batch_size):
            end = min(start + self.batch_size, self.n_samples)
            yield x_src[start:end], y_src[start:end]
            steps += 1
            if self.max_steps and steps >= self.max_steps:
                break


def make_fast_loader(
    dataloader: Any,
    *,
    device: torch.device | str,
    batch_size: int,
    shuffle: bool,
    preload_to_device: bool = True,
    max_preload_mb: float = 512.0,
    max_steps: int | None = None,
    subtask: str | None = None,
) -> Iterable[Any]:
    """Return a device-resident :class:`FastTensorLoader` if the dataset fits in
    ``max_preload_mb``, otherwise return ``dataloader`` unchanged."""
    if not preload_to_device:
        return dataloader

    dataset = getattr(dataloader, "dataset", None)
    if dataset is None:
        return dataloader

    xy = materialize_dataset_tensors(dataset)
    if xy is None:
        try:
            n = len(dataset)
        except Exception:
            return dataloader
        if n == 0 or n > 65536:
            return dataloader
        sample_x, sample_y = unpack_batch(dataset[0])
        if not isinstance(sample_x, torch.Tensor):
            return dataloader
        y_bytes = (
            sample_y.numel() * sample_y.element_size()
            if isinstance(sample_y, torch.Tensor)
            else 8
        )
        est_bytes = n * (sample_x.numel() * sample_x.element_size() + y_bytes)
        if est_bytes > max_preload_mb * 1024 * 1024:
            return dataloader
        xs, ys = [], []
        for batch in dataloader:
            bx, by = unpack_batch(batch)
            xs.append(bx)
            ys.append(by if isinstance(by, torch.Tensor) else torch.as_tensor(by))
        xy = (torch.cat(xs, dim=0), torch.cat(ys, dim=0))

    x, y = xy
    total_bytes = x.numel() * x.element_size() + y.numel() * y.element_size()
    if total_bytes > max_preload_mb * 1024 * 1024:
        return dataloader

    return FastTensorLoader(
        x,
        y,
        batch_size=batch_size,
        shuffle=shuffle,
        device=device,
        max_steps=max_steps,
        subtask=subtask,
    )
