"""Compositional Odd-One-Out datasets from Lepori et al. (NeurIPS 2023).

Reference: ``libs/Compositional_Subnetworks/``
- :class:`CVROddOneOutSOTAProvider`: Vision CVR compositional families
  (``inside_contact`` [103-108], ``inside_count`` [109-114], ``contact_count`` [115-120]).
  Uses ``libs/Compositional_Subnetworks/Vision/CVR/data_generation/subnetwork_tasks.py``
  for real on-the-fly CVR problem rendering when available (or cached tensors),
  falling back cleanly to deterministic synthetic OOO tensors for offline/CI tests.
- :class:`SynEvalOddOneOutSOTAProvider`: Language SynEval syntactic OOO families
  (``sv_singular``, ``sv_plural``, ``pro_ant_singular``, ``pro_ant_plural``).
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from positronixlab.dataprovider.sota.base import CompositionalDataProvider, SubtaskSplit

# Exact Lepori et al. CVR compositional task families (indices 103..120)
CVR_FAMILIES: Dict[str, Dict[str, Tuple[int, str]]] = {
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

# Exact Lepori et al. SynEval compositional syntactic families (indices 0..19)
SYNEVAL_FAMILIES: Dict[str, Dict[str, str]] = {
    "sv_singular": {
        "base": "sn_task_1_sv_singular",
        "s": "sn_task_1_s_singular",
        "v": "sn_task_1_v_singular",
        "ablate_s": "sn_task_1_sv_singular_ablate_s",
        "ablate_v": "sn_task_1_sv_singular_ablate_v",
    },
    "sv_plural": {
        "base": "sn_task_2_sv_plural",
        "s": "sn_task_2_s_plural",
        "v": "sn_task_2_v_plural",
        "ablate_s": "sn_task_2_sv_plural_ablate_s",
        "ablate_v": "sn_task_2_sv_plural_ablate_v",
    },
    "pro_ant_singular": {
        "base": "sn_task_3_pro_ant_singular",
        "pro": "sn_task_3_pro_singular",
        "ant": "sn_task_3_ant_singular",
        "ablate_pro": "sn_task_3_pro_ant_singular_ablate_pro",
        "ablate_ant": "sn_task_3_pro_ant_singular_ablate_ant",
    },
    "pro_ant_plural": {
        "base": "sn_task_4_pro_ant_plural",
        "pro": "sn_task_4_pro_plural",
        "ant": "sn_task_4_ant_plural",
        "ablate_pro": "sn_task_4_pro_ant_plural_ablate_pro",
        "ablate_ant": "sn_task_4_pro_ant_plural_ablate_ant",
    },
}


def _try_load_lepori_cvr_generator(task_fn_name: str) -> Optional[Callable[..., Any]]:
    """Try importing a CVR task generator from ``libs/Compositional_Subnetworks``."""
    repo_root = Path(__file__).resolve().parents[3]
    cvr_dir = repo_root / "libs" / "Compositional_Subnetworks" / "Vision" / "CVR"
    if not cvr_dir.exists():
        return None
    if str(cvr_dir) not in sys.path:
        sys.path.insert(0, str(cvr_dir))
    try:
        from data_generation import subnetwork_tasks  # type: ignore

        return getattr(subnetwork_tasks, task_fn_name, None)
    except Exception:
        return None


class CVROddOneOutDataset(Dataset):
    """4-way Odd-One-Out visual dataset yielding ``(images[4, 3, H, W], target_idx)``.

    Permutes the 4 images deterministically per item (seeded by ``seed + idx``)
    so the outlier is uniformly distributed across ``{0, 1, 2, 3}``.
    """

    def __init__(
        self,
        task_fn_name: str,
        n_samples: int = 64,
        image_size: int = 64,
        seed: int = 0,
        use_real_generator: bool = True,
        tensor_path: Optional[str | Path] = None,
    ) -> None:
        self.task_fn_name = task_fn_name
        self.n_samples = n_samples
        self.image_size = image_size
        self.seed = seed
        self._cached_x: Optional[torch.Tensor] = None
        self._cached_y: Optional[torch.Tensor] = None
        self._gen_fn: Optional[Callable[..., Any]] = None

        if tensor_path is not None and Path(tensor_path).exists():
            payload = torch.load(tensor_path, map_location="cpu", weights_only=True)
            self._cached_x = payload["inputs"][:n_samples]
            self._cached_y = payload["targets"][:n_samples]
            self.n_samples = int(self._cached_x.shape[0])
            return

        if use_real_generator:
            self._gen_fn = _try_load_lepori_cvr_generator(task_fn_name)

        if self._gen_fn is None:
            # Deterministic synthetic OOO visual problems: 3 similar + 1 distinct outlier
            gen = torch.Generator().manual_seed(seed)
            base_pattern = torch.randn(n_samples, 1, 3, image_size, image_size, generator=gen) * 0.2
            noise = torch.randn(n_samples, 4, 3, image_size, image_size, generator=gen) * 0.05
            imgs = base_pattern.expand(-1, 4, -1, -1, -1) + noise
            # Outlier at index 3 has shifted mean/pattern before permutation
            imgs[:, 3] = -imgs[:, 3] + 0.5
            perms = torch.stack([torch.randperm(4, generator=gen) for _ in range(n_samples)])
            targets = perms.argmax(dim=1)
            batch_idx = torch.arange(n_samples).unsqueeze(1)
            self._cached_x = imgs[batch_idx, perms]
            self._cached_y = targets

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._cached_x is not None and self._cached_y is not None:
            return self._cached_x[idx], self._cached_y[idx]

        # Real Lepori CVR generator (returns 4 numpy HWC uint8 images, index 3 = OOO)
        assert self._gen_fn is not None
        imgs_np = self._gen_fn(seed=self.seed + idx, image_size=self.image_size)
        imgs = torch.from_numpy(imgs_np).permute(0, 3, 1, 2).float() / 255.0
        # CVR normalization: Normalize((0.9, 0.9, 0.9), (0.1, 0.1, 0.1))
        imgs = (imgs - 0.9) / 0.1
        gen = torch.Generator().manual_seed(self.seed + idx)
        perm = torch.randperm(4, generator=gen)
        target = perm.argmax()
        return imgs[perm], target


class CVROddOneOutSOTAProvider(CompositionalDataProvider):
    """Lepori et al. CVR compositional OOO family (6 splits per family)."""

    source = "lepori"
    task_type = "odd_one_out"

    def __init__(
        self,
        family: str = "inside_contact",
        n_train: int = 128,
        n_val: int = 32,
        n_test: int = 32,
        image_size: int = 64,
        batch_size: int = 16,
        use_real_generator: bool = False,
        root: Optional[str | Path] = None,
        seed: int = 0,
    ) -> None:
        if family not in CVR_FAMILIES:
            raise ValueError(
                f"Unknown CVR family {family!r}. Choose from {sorted(CVR_FAMILIES.keys())}."
            )
        super().__init__(name=f"lepori:cvr_{family}", batch_size=batch_size)
        self.family = family
        self.num_classes = 4
        spec = CVR_FAMILIES[family]

        for offset, (sub_key, (_idx, fn_name)) in enumerate(spec.items()):
            tr_path = Path(root) / fn_name / "train.pt" if root else None
            va_path = Path(root) / fn_name / "val.pt" if root else None
            te_path = Path(root) / fn_name / "test.pt" if root else None
            self.subtasks[sub_key] = SubtaskSplit(
                name=sub_key,
                dataset_train=CVROddOneOutDataset(
                    fn_name, n_train, image_size, seed + offset * 1000, use_real_generator, tr_path
                ),
                dataset_val=CVROddOneOutDataset(
                    fn_name, n_val, image_size, seed + offset * 1000 + 500, use_real_generator, va_path
                ),
                dataset_test=CVROddOneOutDataset(
                    fn_name, n_test, image_size, seed + offset * 1000 + 800, use_real_generator, te_path
                ),
                batch_size=batch_size,
            )

        subroutines = [k for k in spec if k not in ("base", "both_ooo") and not k.startswith("ablate_")]
        all_eval_keys = list(spec.keys())
        self._eval_matrix = {sr: all_eval_keys for sr in subroutines}


class SynEvalOddOneOutDataset(Dataset):
    """4-way Syntactic Odd-One-Out dataset yielding ``(stacked[2, 4, L], target_idx)``."""

    def __init__(
        self,
        task_name: str,
        split: str = "train",
        n_samples: int = 64,
        max_length: int = 50,
        vocab_size: int = 1000,
        root: Optional[str | Path] = None,
        tokenizer_name: str = "prajjwal1/bert-small",
        seed: int = 0,
    ) -> None:
        self.task_name = task_name
        pkl_path = Path(root) / task_name / f"{split}.pkl" if root is not None else None

        if pkl_path is not None and pkl_path.exists():
            from transformers import AutoTokenizer

            with open(pkl_path, "rb") as f:
                problems: List[List[str]] = pickle.load(f)[:n_samples]
            tok = AutoTokenizer.from_pretrained(tokenizer_name)
            flat_sents = [s for prob in problems for s in prob]
            enc = tok(
                flat_sents,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            n = len(problems)
            ids = enc["input_ids"].reshape(n, 4, max_length)
            attn = enc["attention_mask"].reshape(n, 4, max_length)
            gen = torch.Generator().manual_seed(seed)
            perms = torch.stack([torch.randperm(4, generator=gen) for _ in range(n)])
            batch_idx = torch.arange(n).unsqueeze(1)
            self.x = torch.stack([ids[batch_idx, perms], attn[batch_idx, perms]], dim=1)
            self.y = perms.argmax(dim=1)
        else:
            gen = torch.Generator().manual_seed(seed)
            base_ids = torch.randint(5, vocab_size, (n_samples, 1, max_length), generator=gen)
            ids = base_ids.expand(-1, 4, -1).clone()
            ids[:, 3, :4] = torch.randint(1, 5, (n_samples, 4), generator=gen)
            attn = torch.ones((n_samples, 4, max_length), dtype=torch.long)
            perms = torch.stack([torch.randperm(4, generator=gen) for _ in range(n_samples)])
            batch_idx = torch.arange(n_samples).unsqueeze(1)
            self.x = torch.stack([ids[batch_idx, perms], attn[batch_idx, perms]], dim=1)
            self.y = perms.argmax(dim=1)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class SynEvalOddOneOutSOTAProvider(CompositionalDataProvider):
    """Lepori et al. SynEval syntactic OOO family (5 splits per family)."""

    source = "lepori"
    task_type = "odd_one_out"

    def __init__(
        self,
        family: str = "sv_singular",
        n_train: int = 128,
        n_val: int = 32,
        n_test: int = 32,
        max_length: int = 32,
        vocab_size: int = 1000,
        batch_size: int = 16,
        root: Optional[str | Path] = None,
        seed: int = 0,
    ) -> None:
        if family not in SYNEVAL_FAMILIES:
            raise ValueError(
                f"Unknown SynEval family {family!r}. Choose from {sorted(SYNEVAL_FAMILIES.keys())}."
            )
        super().__init__(name=f"lepori:syneval_{family}", batch_size=batch_size)
        self.family = family
        self.num_classes = 4
        spec = SYNEVAL_FAMILIES[family]

        for offset, (sub_key, task_name) in enumerate(spec.items()):
            is_ablate = sub_key.startswith("ablate_")
            self.subtasks[sub_key] = SubtaskSplit(
                name=sub_key,
                dataset_train=(
                    None
                    if is_ablate
                    else SynEvalOddOneOutDataset(
                        task_name, "train", n_train, max_length, vocab_size, root, seed=seed + offset * 100
                    )
                ),
                dataset_val=SynEvalOddOneOutDataset(
                    task_name, "val", n_val, max_length, vocab_size, root, seed=seed + offset * 100 + 10
                ),
                dataset_test=SynEvalOddOneOutDataset(
                    task_name, "test", n_test, max_length, vocab_size, root, seed=seed + offset * 100 + 20
                ),
                batch_size=batch_size,
            )

        subroutines = [k for k in spec if k != "base" and not k.startswith("ablate_")]
        self._eval_matrix = {sr: list(spec.keys()) for sr in subroutines}
