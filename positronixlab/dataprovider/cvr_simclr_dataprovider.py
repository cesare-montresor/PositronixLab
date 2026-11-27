"""CVR SimCLR self-supervised pretraining dataset provider (Lepori et al., NeurIPS 2023).

Reference: ``libs/Compositional_Subnetworks/Vision/CVR/datasets/SimCLR_cvrt.py``

Aggregates the 3 base CVR compositional tasks (indices 103, 109, 115):
- ``sn_task_1_contact_inside`` (103)
- ``sn_task_2_inside_count`` (109)
- ``sn_task_3_contact_count`` (115)

For each quartet, extracts the last 2 images (1 rule-following + 1 odd-one-out)
and applies 2-view :class:`ContrastiveTransformations` (random horizontal flip,
color jitter, affine, 90-degree rotation, grayscale, and CVR normalization).

Data loading priority per task:
1. Pre-generated horizontal-strip PNGs (``<root>/<task_fn_name>/{train,val}/{05d}.png``)
2. Pre-generated ``.pt`` tensor cache (``<root>/<task_fn_name>/{train,val}.pt``)
3. On-the-fly CVR quartet renderer (``use_real_generator=True`` when OpenCV is available)
4. Deterministic synthetic images in ``[0, 1]`` for offline/CI runs
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from positronixlab.dataprovider.cvr_oddoneout_dataprovider import load_cvr_png_strip
from positronixlab.dataprovider.dataprovider import (
    DataProvider,
    resolve_data_root,
)

CVR_SIMCLR_BASE_TASKS: tuple[tuple[int, str], ...] = (
    (103, "sn_task_1_contact_inside"),
    (109, "sn_task_2_inside_count"),
    (115, "sn_task_3_contact_count"),
)


def build_simclr_transforms() -> Callable[[torch.Tensor], torch.Tensor]:
    """Return Lepori et al.'s exact CVR SimCLR augmentation pipeline."""
    from torchvision import transforms as tvt

    return tvt.Compose(
        [
            tvt.RandomHorizontalFlip(),
            tvt.RandomApply([tvt.ColorJitter(contrast=0.5, saturation=0.5)], p=0.5),
            tvt.RandomApply(
                [
                    tvt.RandomAffine(
                        degrees=30,
                        translate=(0.05, 0.05),
                        scale=(0.9, 1.0),
                        shear=20,
                        fill=1,
                    )
                ],
                p=0.5,
            ),
            tvt.RandomRotation(degrees=90, fill=1),
            tvt.RandomGrayscale(p=0.2),
            tvt.Normalize((0.9, 0.9, 0.9), (0.1, 0.1, 0.1)),
        ]
    )


class ContrastiveTransformations:
    """Apply *base_transforms* independently *n_views* times to create contrastive pairs."""

    def __init__(
        self,
        base_transforms: Callable[[torch.Tensor], torch.Tensor],
        n_views: int = 2,
        as_list: bool = False,
    ) -> None:
        self.base_transforms = base_transforms
        self.n_views = n_views
        self.as_list = as_list

    def __call__(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        views = [self.base_transforms(x) for _ in range(self.n_views)]
        if self.as_list:
            return views
        return torch.stack(views, dim=0)


_DEFAULT_CVR_SIMCLR_TASKS: tuple[str, ...] = tuple(fn for _, fn in CVR_SIMCLR_BASE_TASKS)


class CVRSimCLRDataset(Dataset):
    """CVR SimCLR contrastive dataset yielding ``(views, task_idx)``.

    When ``as_list=False`` (default), ``views`` has shape ``(n_views, 2, 3, H, W)``
    where dim 0 indexes the contrastive view and dim 1 holds the 2 extracted CVR
    images (1 rule-following + 1 odd-one-out). When ``as_list=True``, returns a
    list of ``n_views`` tensors each of shape ``(2, 3, H, W)`` matching
    ``SimClrCVRT``.
    """

    def __init__(
        self,
        split: str = "train",
        n_samples_per_task: int = 128,
        image_size: int = 64,
        tasks: Sequence[str] = _DEFAULT_CVR_SIMCLR_TASKS,
        n_views: int = 2,
        as_list: bool = False,
        seed: int = 0,
        use_real_generator: bool = False,
        root: str | Path | None = None,
        transform: Callable[[torch.Tensor], torch.Tensor | list[torch.Tensor]] | None = None,
    ) -> None:
        super().__init__()
        self.split = split
        self.tasks = tuple(tasks)
        self.n_samples_per_task = n_samples_per_task
        self.image_size = image_size
        self.n_views = n_views
        self.as_list = as_list
        self.seed = seed
        self.transform = (
            transform
            if transform is not None
            else ContrastiveTransformations(
                build_simclr_transforms(),
                n_views=n_views,
                as_list=as_list,
            )
        )

        data_root = resolve_data_root(root) if root is not None else None
        self._png_files: list[list[Path] | None] = []
        self._cached_pt: list[torch.Tensor | None] = []
        self._use_renderer = False

        any_disk = False
        for fn_name in self.tasks:
            pngs: list[Path] | None = None
            pt_tensor: torch.Tensor | None = None
            if data_root is not None:
                png_dir = data_root / fn_name / split
                if png_dir.is_dir():
                    found = sorted(png_dir.glob("*.png"))
                    if found:
                        pngs = found[:n_samples_per_task] if n_samples_per_task > 0 else found
                        any_disk = True
                if pngs is None:
                    pt_path = data_root / fn_name / f"{split}.pt"
                    if pt_path.exists():
                        payload = torch.load(pt_path, map_location="cpu", weights_only=True)
                        pt_tensor = payload["inputs"][:n_samples_per_task]
                        any_disk = True
            self._png_files.append(pngs)
            self._cached_pt.append(pt_tensor)

        if any_disk:
            counts = []
            for pngs, pt in zip(self._png_files, self._cached_pt, strict=False):
                if pngs is not None:
                    counts.append(len(pngs))
                elif pt is not None:
                    counts.append(int(pt.shape[0]))
            if counts:
                self.n_samples_per_task = min(counts)

        if not any_disk and use_real_generator:
            from positronixlab.baselines.lepori._cvr_gen import (
                is_cvr_renderer_available,
            )

            if is_cvr_renderer_available():
                self._use_renderer = True

        self._synthetic: torch.Tensor | None = None
        if not any_disk and not self._use_renderer:
            gen = torch.Generator().manual_seed(seed)
            total = len(self.tasks) * self.n_samples_per_task
            base = torch.rand(total, 1, 3, image_size, image_size, generator=gen) * 0.6 + 0.2
            noise = torch.rand(total, 2, 3, image_size, image_size, generator=gen) * 0.15
            imgs = (base.expand(-1, 2, -1, -1, -1) + noise).clamp(0.0, 1.0)
            imgs[:, 1] = (1.0 - imgs[:, 1]).clamp(0.0, 1.0)
            self._synthetic = imgs

    def __len__(self) -> int:
        return len(self.tasks) * self.n_samples_per_task

    def _load_pair(self, task_idx: int, sample_idx: int, flat_idx: int) -> torch.Tensor:
        pngs = self._png_files[task_idx]
        if pngs is not None:
            quartet = load_cvr_png_strip(pngs[sample_idx], image_size=self.image_size)
            return quartet[2:]

        pt = self._cached_pt[task_idx]
        if pt is not None:
            quartet = pt[sample_idx]
            return quartet[2:]

        if self._use_renderer:
            from positronixlab.baselines.lepori._cvr_gen import render_cvr_quartet

            imgs_np = render_cvr_quartet(
                self.tasks[task_idx],
                image_size=self.image_size,
                seed=self.seed + task_idx * 100_000 + sample_idx,
            )
            quartet = torch.from_numpy(imgs_np).permute(0, 3, 1, 2).float() / 255.0
            return quartet[2:]

        if self._synthetic is None:
            raise RuntimeError("CVRSimCLRDataset synthetic buffer is not initialized.")
        return self._synthetic[flat_idx]

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor | list[torch.Tensor], torch.Tensor]:
        task_idx = idx // self.n_samples_per_task
        sample_idx = idx % self.n_samples_per_task
        pair = self._load_pair(task_idx, sample_idx, idx)
        views = self.transform(pair) if self.transform is not None else pair
        return views, torch.tensor(task_idx, dtype=torch.long)


class CVRSimCLRDataProvider(
    DataProvider,
    plugin_name="cvr_simclr",
    aliases=("lepori:cvr_simclr",),
):
    """Lepori et al. CVR SimCLR self-supervised contrastive pretraining provider."""

    source = "lepori"
    task_type = "contrastive"
    supports_flatten = False

    def __init__(
        self,
        n_train_per_task: int = 128,
        n_val_per_task: int = 32,
        image_size: int = 64,
        tasks: Sequence[str] = _DEFAULT_CVR_SIMCLR_TASKS,
        n_views: int = 2,
        as_list: bool = False,
        batch_size: int = 16,
        use_real_generator: bool = False,
        root: str | Path | None = None,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            root=root,
            seed=seed,
            **kwargs,
        )
        self.name = "lepori:cvr_simclr"
        self.tasks = tuple(tasks)
        self.image_size = image_size
        self.n_views = n_views
        self.as_list = as_list
        self.use_real_generator = use_real_generator
        self.num_features = None
        self.num_classes = len(self.tasks)
        self.x_shape = (n_views, 2, 3, image_size, image_size)
        self.y_shape = ()
        from positronixlab.baselines.lepori.models import SimCLRLoss

        self.criterion = SimCLRLoss(temperature=0.07)

        self.dataset_train = CVRSimCLRDataset(
            split="train",
            n_samples_per_task=n_train_per_task,
            image_size=image_size,
            tasks=self.tasks,
            n_views=n_views,
            as_list=as_list,
            seed=seed,
            use_real_generator=use_real_generator,
            root=root,
        )
        self.dataset_val = CVRSimCLRDataset(
            split="val",
            n_samples_per_task=n_val_per_task,
            image_size=image_size,
            tasks=self.tasks,
            n_views=n_views,
            as_list=as_list,
            seed=seed + 10_000,
            use_real_generator=use_real_generator,
            root=root,
        )
        self.dataset_test = self.dataset_val
