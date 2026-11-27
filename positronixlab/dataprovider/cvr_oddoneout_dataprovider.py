"""CVR compositional visual Odd-One-Out dataset provider (Lepori et al., NeurIPS 2023).

Supports all 3 Lepori et al. CVR compositional families (task indices 103..120):
- ``inside_contact`` (6 subtasks: ``base``, ``inside``, ``contact``, ``both_ooo``, ``ablate_contact``, ``ablate_inside``)
- ``inside_count`` (6 subtasks: ``base``, ``inside``, ``count``, ``both_ooo``, ``ablate_inside``, ``ablate_count``)
- ``contact_count`` (6 subtasks: ``base``, ``contact``, ``count``, ``both_ooo``, ``ablate_contact``, ``ablate_count``)

Data loading priority:
1. Pre-generated ``.pt`` tensor file (``<root>/<task_fn_name>/{train,val,test}.pt`` or ``<root>/cvr_odd_one_out/{train,test}.pt``)
2. Pre-generated Lepori horizontal-strip PNGs (``<root>/<task_fn_name>/{train,val,test}/{05d}.png``)
3. On-the-fly CVR quartet renderer (``use_real_generator=True`` when OpenCV is available)
4. Deterministic synthetic 4-way Odd-One-Out visual tensors for offline/CI runs
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from positronixlab.dataprovider.dataprovider import (
    CompositionalDataProvider,
    SubtaskSplit,
    TransformLike,
    apply_transform,
    resolve_data_root,
)

CVRFamily = Literal["inside_contact", "inside_count", "contact_count"]

# Exact Lepori et al. CVR compositional task families (indices 103..120)
CVR_FAMILIES: dict[str, dict[str, tuple[int, str]]] = {
    "inside_contact": {
        "base": (103, "sn_task_1_contact_inside"),
        "inside": (104, "sn_task_1_inside"),
        "contact": (105, "sn_task_1_contact"),
        "both_ooo": (106, "sn_task_1_contact_inside_both_ooo"),
        "ablate_contact": (107, "sn_task_1_contact_inside_ablate_contact"),
        "ablate_inside": (108, "sn_task_1_contact_inside_ablate_inside"),
    },
    "inside_count": {
        "base": (109, "sn_task_2_inside_count"),
        "inside": (110, "sn_task_2_inside"),
        "count": (111, "sn_task_2_count"),
        "both_ooo": (112, "sn_task_2_inside_count_both_ooo"),
        "ablate_inside": (113, "sn_task_2_inside_count_ablate_inside"),
        "ablate_count": (114, "sn_task_2_inside_count_ablate_count"),
    },
    "contact_count": {
        "base": (115, "sn_task_3_contact_count"),
        "contact": (116, "sn_task_3_contact"),
        "count": (117, "sn_task_3_count"),
        "both_ooo": (118, "sn_task_3_contact_count_both_ooo"),
        "ablate_contact": (119, "sn_task_3_contact_count_ablate_contact"),
        "ablate_count": (120, "sn_task_3_contact_count_ablate_count"),
    },
}


def load_cvr_png_strip(
    path: str | Path,
    image_size: int = 128,
) -> torch.Tensor:
    """Load a Lepori/CVR 4-image horizontal-strip PNG into ``(4, 3, image_size, image_size)`` float in ``[0, 1]``.

    Matches ``Vision/CVR/datasets/cvrt.py`` and ``SimCLR_cvrt.py``: reshapes the
    ``(3, H_pad, 4 * H_pad)`` strip into 4 images and strips uniform border
    padding around each ``image_size x image_size`` frame.
    """
    from PIL import Image

    with Image.open(path) as img:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0

    # (H_pad, 4 * H_pad, 3) -> (3, H_pad, 4 * H_pad)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    im_size = int(tensor.shape[1])
    if tensor.shape[2] != 4 * im_size:
        raise ValueError(
            f"Expected horizontal 4-image CVR strip with width == 4 * height, "
            f"got shape {tuple(tensor.shape)} from {path}"
        )
    pad = im_size - image_size
    quartet = tensor.reshape(3, im_size, 4, im_size).permute(2, 0, 1, 3).contiguous()
    if pad > 0:
        half = pad // 2
        quartet = quartet[:, :, half : half + image_size, half : half + image_size]
    return quartet


class CVROddOneOutDataset(Dataset):
    """4-way Odd-One-Out visual dataset yielding ``(images[4, 3, H, W], target_idx)``.

    Permutes the 4 images deterministically per item (seeded by ``seed + idx``)
    so the outlier is uniformly distributed across ``{0, 1, 2, 3}``.
    """

    def __init__(
        self,
        task_fn_name: str = "sn_task_1_contact_inside",
        n_samples: int = 64,
        image_size: int = 64,
        seed: int = 0,
        use_real_generator: bool = False,
        tensor_path: str | Path | None = None,
        png_dir: str | Path | None = None,
        transforms: TransformLike | None = None,
    ) -> None:
        super().__init__()
        self.task_fn_name = task_fn_name
        self.n_samples = n_samples
        self.image_size = image_size
        self.seed = seed
        self.transforms = transforms
        self._cached_x: torch.Tensor | None = None
        self._cached_y: torch.Tensor | None = None
        self._png_files: list[Path] | None = None
        self._use_renderer: bool = False

        if tensor_path is not None and Path(tensor_path).exists():
            payload = torch.load(tensor_path, map_location="cpu", weights_only=True)
            cached_x: torch.Tensor = payload["inputs"][:n_samples]
            self._cached_x = cached_x
            self._cached_y = payload["targets"][:n_samples]
            self.n_samples = int(cached_x.shape[0])
            return

        if png_dir is not None and Path(png_dir).is_dir():
            pngs = sorted(Path(png_dir).glob("*.png"))
            if pngs:
                self._png_files = pngs[:n_samples] if n_samples > 0 else pngs
                self.n_samples = len(self._png_files)
                return

        if use_real_generator:
            from positronixlab.baselines.lepori._cvr_gen import (
                is_cvr_renderer_available,
            )

            if is_cvr_renderer_available():
                self._use_renderer = True
                return

        # Deterministic synthetic OOO visual problems: 3 similar + 1 distinct outlier
        gen = torch.Generator().manual_seed(seed)
        base_pattern = torch.randn(n_samples, 1, 3, image_size, image_size, generator=gen) * 0.2
        noise = torch.randn(n_samples, 4, 3, image_size, image_size, generator=gen) * 0.05
        imgs = base_pattern.expand(-1, 4, -1, -1, -1) + noise
        imgs[:, 3] = -imgs[:, 3] + 0.5
        perms = torch.stack([torch.randperm(4, generator=gen) for _ in range(n_samples)])
        targets = perms.argmax(dim=1)
        batch_idx = torch.arange(n_samples).unsqueeze(1)
        self._cached_x = imgs[batch_idx, perms]
        self._cached_y = targets

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cached_x is not None and self._cached_y is not None:
            imgs, target = self._cached_x[idx], self._cached_y[idx]
        elif self._png_files is not None:
            imgs = load_cvr_png_strip(self._png_files[idx], image_size=self.image_size)
            imgs = (imgs - 0.9) / 0.1
            gen = torch.Generator().manual_seed(self.seed + idx)
            perm = torch.randperm(4, generator=gen)
            target = perm.argmax()
            imgs = imgs[perm]
        else:
            from positronixlab.baselines.lepori._cvr_gen import render_cvr_quartet

            imgs_np = render_cvr_quartet(
                self.task_fn_name, image_size=self.image_size, seed=self.seed + idx
            )
            imgs = torch.from_numpy(imgs_np).permute(0, 3, 1, 2).float() / 255.0
            # CVR normalization: Normalize((0.9, 0.9, 0.9), (0.1, 0.1, 0.1))
            imgs = (imgs - 0.9) / 0.1
            gen = torch.Generator().manual_seed(self.seed + idx)
            perm = torch.randperm(4, generator=gen)
            target = perm.argmax()
            imgs = imgs[perm]

        if self.transforms is not None:
            imgs = torch.stack([apply_transform(self.transforms, img) for img in imgs])
        return imgs, target


class CVROddOneOutDataProvider(
    CompositionalDataProvider,
    plugin_name="cvr_odd_one_out",
):
    """Lepori et al. CVR compositional visual Odd-One-Out provider (6 subtasks per family)."""

    source = "lepori"
    task_type = "odd_one_out"
    supports_flatten = True

    def __init__(
        self,
        family: CVRFamily = "inside_contact",
        n_train: int = 128,
        n_val: int = 32,
        n_test: int = 32,
        image_size: int = 64,
        batch_size: int = 16,
        use_real_generator: bool = False,
        root: str | Path | None = None,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        if family not in CVR_FAMILIES:
            raise ValueError(
                f"Unknown CVR family {family!r}. Choose from {sorted(CVR_FAMILIES.keys())}."
            )
        super().__init__(
            name=f"lepori:cvr_{family}",
            batch_size=batch_size,
            root=root,
            seed=seed,
            **kwargs,
        )
        self.family = family
        self.image_size = image_size
        self.use_real_generator = use_real_generator
        self.num_features = None
        self.num_classes = 4
        self.x_shape = (4, 3, image_size, image_size)
        self.y_shape = ()
        self.criterion = nn.CrossEntropyLoss()

        spec = CVR_FAMILIES[family]
        data_root = resolve_data_root(root) if root is not None else None
        split_kw: dict[str, Any] = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            indexed_dataset=self.indexed_dataset,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )

        for offset, (sub_key, (_idx, fn_name)) in enumerate(spec.items()):
            tr_path = (data_root / fn_name / "train.pt") if data_root is not None else None
            va_path = (data_root / fn_name / "val.pt") if data_root is not None else None
            te_path = (data_root / fn_name / "test.pt") if data_root is not None else None
            tr_png = (data_root / fn_name / "train") if data_root is not None else None
            va_png = (data_root / fn_name / "val") if data_root is not None else None
            te_png = (data_root / fn_name / "test") if data_root is not None else None
            self.subtasks[sub_key] = SubtaskSplit(
                name=sub_key,
                dataset_train=CVROddOneOutDataset(
                    fn_name,
                    n_train,
                    image_size,
                    seed + offset * 1000,
                    use_real_generator,
                    tr_path,
                    png_dir=tr_png,
                    transforms=self.transforms,
                ),
                dataset_val=CVROddOneOutDataset(
                    fn_name,
                    n_val,
                    image_size,
                    seed + offset * 1000 + 500,
                    use_real_generator,
                    va_path,
                    png_dir=va_png,
                    transforms=self.transforms,
                ),
                dataset_test=CVROddOneOutDataset(
                    fn_name,
                    n_test,
                    image_size,
                    seed + offset * 1000 + 800,
                    use_real_generator,
                    te_path,
                    png_dir=te_png,
                    transforms=self.transforms,
                ),
                **split_kw,
            )

        subroutines = [
            k for k in spec if k not in ("base", "both_ooo") and not k.startswith("ablate_")
        ]
        all_eval_keys = list(spec.keys())
        self._eval_matrix = {sr: all_eval_keys for sr in subroutines}
        self._sync_base_split()
