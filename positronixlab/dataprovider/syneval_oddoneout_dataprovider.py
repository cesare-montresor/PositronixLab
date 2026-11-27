"""SynEval compositional syntactic Odd-One-Out dataset provider (Lepori et al., NeurIPS 2023).

Supports all 4 Lepori et al. SynEval syntactic families (5 subtasks per family):
- ``sv_singular`` (``base``, ``s``, ``v``, ``ablate_s``, ``ablate_v``)
- ``sv_plural`` (``base``, ``s``, ``v``, ``ablate_s``, ``ablate_v``)
- ``pro_ant_singular`` (``base``, ``pro``, ``ant``, ``ablate_pro``, ``ablate_ant``)
- ``pro_ant_plural`` (``base``, ``pro``, ``ant``, ``ablate_pro``, ``ablate_ant``)

Data loading priority:
1. Pre-generated pickled sentence quartets (``<root>/<task_name>/{train,val,test}.pkl``)
2. Real split generation from Marvin & Linzen ``RawData.zip`` / ``RawData/``
   (via ``raw_data_path``, ``<root>/RawData.zip``, or GitHub auto-download when
   ``download=True``), tokenized with HuggingFace ``AutoTokenizer``
3. Deterministic synthetic ``(2, 4, max_length)`` ``[input_ids, attention_mask]``
   tensors for offline/CI runs when no files are present and ``download=False``
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union, Unpack

import torch
from torch import nn
from torch.utils.data import Dataset

from positronixlab.dataprovider.dataprovider import (
    CompositionalDataProvider,
    DataProviderOptions,
    SubtaskSplit,
    TransformLike,
    apply_transform,
    resolve_data_root,
)

SynEvalFamily = Literal["sv_singular", "sv_plural", "pro_ant_singular", "pro_ant_plural"]

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


class SynevalOddOneOutDataset(Dataset):
    """4-way Syntactic Odd-One-Out dataset yielding ``(stacked[2, 4, L], target_idx)``."""

    def __init__(
        self,
        task_name: str = "sn_task_1_sv_singular",
        split: str = "train",
        n_samples: int = 64,
        max_length: int = 50,
        vocab_size: int = 1000,
        root: Optional[Union[str, Path]] = None,
        tokenizer_name: str = "prajjwal1/bert-small",
        tokenizer: Optional[Callable[..., Dict[str, Any]]] = None,
        seed: int = 0,
        transforms: Optional[TransformLike] = None,
    ) -> None:
        super().__init__()
        self.task_name = task_name
        self.transforms = transforms
        pkl_path = Path(root) / task_name / f"{split}.pkl" if root is not None else None

        if pkl_path is not None and pkl_path.exists():
            with open(pkl_path, "rb") as f:
                problems: List[List[str]] = pickle.load(f)[:n_samples]
            if tokenizer is not None:
                tok = tokenizer
            else:
                from transformers import AutoTokenizer

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
        x = self.x[idx]
        if self.transforms is not None:
            x = apply_transform(self.transforms, x)
        return x, self.y[idx]


class SynevalOddOneOutDataProvider(
    CompositionalDataProvider,
    plugin_name="syneval_odd_one_out",
):
    """Lepori et al. SynEval syntactic Odd-One-Out provider (5 subtasks per family)."""

    source = "lepori"
    task_type = "odd_one_out"

    def __init__(
        self,
        family: SynEvalFamily = "sv_singular",
        n_train: int = 128,
        n_val: int = 32,
        n_test: int = 32,
        max_length: int = 32,
        vocab_size: int = 1000,
        batch_size: int = 16,
        root: Optional[Union[str, Path]] = None,
        raw_data_path: Optional[Union[str, Path]] = None,
        download: bool = False,
        tokenizer_name: str = "prajjwal1/bert-small",
        tokenizer: Optional[Callable[..., Dict[str, Any]]] = None,
        seed: int = 0,
        **kwargs: Unpack[DataProviderOptions],
    ) -> None:
        if family not in SYNEVAL_FAMILIES:
            raise ValueError(
                f"Unknown SynEval family {family!r}. Choose from {sorted(SYNEVAL_FAMILIES.keys())}."
            )
        super().__init__(
            name=f"lepori:syneval_{family}",
            batch_size=batch_size,
            root=root,
            seed=seed,
            **kwargs,
        )
        self.family = family
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.tokenizer_name = tokenizer_name
        self.download = download
        self.num_features = None
        self.num_classes = 4
        self.x_shape = (2, 4, max_length)
        self.y_shape = ()
        self.criterion = nn.CrossEntropyLoss()

        spec = SYNEVAL_FAMILIES[family]

        # Resolve / generate real SynEval .pkl splits when raw data or download is requested
        resolved_root: Optional[Path] = Path(root) if root is not None else None
        if resolved_root is not None or raw_data_path is not None or download:
            from positronixlab.baselines.lepori._syneval_gen import (
                ensure_syneval_family_splits,
            )

            cache_dir = (
                resolved_root
                if resolved_root is not None
                else resolve_data_root(None) / "syneval"
            )

            ensured = ensure_syneval_family_splits(
                family=family,
                cache_dir=cache_dir,
                raw_data_path=raw_data_path,
                download=download,
                seed=seed,
            )
            if ensured is not None:
                resolved_root = ensured

        split_kw = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            indexed_dataset=self.indexed_dataset,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )

        for offset, (sub_key, task_name) in enumerate(spec.items()):
            is_ablate = sub_key.startswith("ablate_")
            self.subtasks[sub_key] = SubtaskSplit(
                name=sub_key,
                dataset_train=(
                    None
                    if is_ablate
                    else SynevalOddOneOutDataset(
                        task_name,
                        "train",
                        n_train,
                        max_length,
                        vocab_size,
                        resolved_root,
                        tokenizer_name=tokenizer_name,
                        tokenizer=tokenizer,
                        seed=seed + offset * 100,
                        transforms=self.transforms,
                    )
                ),
                dataset_val=SynevalOddOneOutDataset(
                    task_name,
                    "val",
                    n_val,
                    max_length,
                    vocab_size,
                    resolved_root,
                    tokenizer_name=tokenizer_name,
                    tokenizer=tokenizer,
                    seed=seed + offset * 100 + 10,
                    transforms=self.transforms,
                ),
                dataset_test=SynevalOddOneOutDataset(
                    task_name,
                    "test",
                    n_test,
                    max_length,
                    vocab_size,
                    resolved_root,
                    tokenizer_name=tokenizer_name,
                    tokenizer=tokenizer,
                    seed=seed + offset * 100 + 20,
                    transforms=self.transforms,
                ),
                **split_kw,
            )

        subroutines = [k for k in spec if k != "base" and not k.startswith("ablate_")]
        self._eval_matrix = {sr: list(spec.keys()) for sr in subroutines}
        self._sync_base_split()
