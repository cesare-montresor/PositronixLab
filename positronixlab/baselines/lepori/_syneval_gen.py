"""SynEval syntactic Odd-One-Out dataset generator (Lepori et al., NeurIPS 2023).

Reference: ``libs/Compositional_Subnetworks/Language/LM_syneval/src/create_datasets.py``

Generates the 4 compositional syntactic families (20 subtask folders total) directly
from Marvin & Linzen's ``RawData.zip`` (or an unpacked ``RawData/`` directory)
without mutating global NumPy RNG state:

- ``sv_singular`` / ``sv_plural`` (Subject-Verb Agreement):
  9,500 train / 500 val / 1,000 test (base & subroutines), 300 val / 300 test (ablation splits).
- ``pro_ant_singular`` / ``pro_ant_plural`` (Reflexive Anaphora):
  2,500 train / 200 val / 200 test (base, subroutines, and ablation splits).
"""

from __future__ import annotations

import pickle
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

SYNEVAL_RAWDATA_URL: str = (
    "https://raw.githubusercontent.com/mlepori1/Compositional_Subnetworks/"
    "main/Language/LM_syneval/RawData.zip"
)

SYNEVAL_PAPER_SIZES: Dict[str, Dict[str, int]] = {
    "sv_singular": {
        "train": 9_500,
        "val": 500,
        "test": 1_000,
        "part_val": 300,
        "part_test": 300,
    },
    "sv_plural": {
        "train": 9_500,
        "val": 500,
        "test": 1_000,
        "part_val": 300,
        "part_test": 300,
    },
    "pro_ant_singular": {
        "train": 2_500,
        "val": 200,
        "test": 200,
        "part_val": 200,
        "part_test": 200,
    },
    "pro_ant_plural": {
        "train": 2_500,
        "val": 200,
        "test": 200,
        "part_val": 200,
        "part_test": 200,
    },
}

# Exact Marvin & Linzen template files used by Lepori et al. (create_datasets.py)
_SV_RAW_FILES: Tuple[str, ...] = (
    "obj_rel_across_anim.pickle",
    "obj_rel_across_inanim.pickle",
    "obj_rel_no_comp_across_anim.pickle",
    "obj_rel_no_comp_across_inanim.pickle",
    "prep_anim.pickle",
    "prep_inanim.pickle",
    "simple_agrmt.pickle",
    "subj_rel.pickle",
)

_ANAPHORA_RAW_FILES: Tuple[str, ...] = (
    "reflexives_across.pickle",
    "simple_reflexives.pickle",
)

# Ordered 5-tuples matching create_datasets(ab, a_flip_b, a_b_flip, ..., dataset_names):
#   [0] base, [1] property_a, [2] property_b, [3] partition_a (ablate_b), [4] partition_b (ablate_a)
_FAMILY_TASK_ORDER: Dict[str, Tuple[str, str, str, str, str]] = {
    "sv_singular": (
        "sn_task_1_sv_singular",
        "sn_task_1_s_singular",
        "sn_task_1_v_singular",
        "sn_task_1_sv_singular_ablate_v",
        "sn_task_1_sv_singular_ablate_s",
    ),
    "sv_plural": (
        "sn_task_2_sv_plural",
        "sn_task_2_s_plural",
        "sn_task_2_v_plural",
        "sn_task_2_sv_plural_ablate_v",
        "sn_task_2_sv_plural_ablate_s",
    ),
    "pro_ant_singular": (
        "sn_task_3_pro_ant_singular",
        "sn_task_3_ant_singular",
        "sn_task_3_pro_singular",
        "sn_task_3_pro_ant_singular_ablate_pro",
        "sn_task_3_pro_ant_singular_ablate_ant",
    ),
    "pro_ant_plural": (
        "sn_task_4_pro_ant_plural",
        "sn_task_4_ant_plural",
        "sn_task_4_pro_plural",
        "sn_task_4_pro_ant_plural_ablate_pro",
        "sn_task_4_pro_ant_plural_ablate_ant",
    ),
}


def download_syneval_rawdata(
    dest_dir: Union[str, Path],
    *,
    url: str = SYNEVAL_RAWDATA_URL,
    timeout: int = 60,
) -> Path:
    """Download ``RawData.zip`` from ``Compositional_Subnetworks`` into *dest_dir*."""
    import requests

    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    zip_path = dest_path / "RawData.zip"
    if zip_path.exists():
        return zip_path

    tmp_path = dest_path / "RawData.zip.tmp"
    resp = requests.get(url, stream=True, timeout=timeout)
    resp.raise_for_status()
    with open(tmp_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            if chunk:
                f.write(chunk)
    tmp_path.replace(zip_path)
    return zip_path


def _load_raw_pickle(raw_source: Path, filename: str) -> Dict[str, List[Tuple[str, str]]]:
    if raw_source.is_file():
        with zipfile.ZipFile(raw_source, "r") as zf:
            names = set(zf.namelist())
            for candidate in (f"RawData/{filename}", filename):
                if candidate in names:
                    return pickle.loads(zf.read(candidate))
            raise FileNotFoundError(
                f"Expected {filename!r} inside {raw_source}, found {len(names)} entries."
            )

    for candidate in (raw_source / filename, raw_source / "RawData" / filename):
        if candidate.exists():
            with open(candidate, "rb") as f:
                return pickle.load(f)
    raise FileNotFoundError(f"Could not find {filename!r} under {raw_source}")


def _create_train_val_test(
    positive: np.ndarray,
    negative: np.ndarray,
    *,
    train_size: int,
    val_size: int,
    test_size: int,
    rng: np.random.Generator,
) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
    pos = positive.copy()
    neg = negative.copy()
    rng.shuffle(pos)
    rng.shuffle(neg)

    tr_end = train_size * 3
    va_end = tr_end + val_size * 3

    train_pos = pos[:tr_end]
    val_pos = pos[tr_end:va_end]
    test_pos = pos[va_end : va_end + test_size * 3]

    train_neg = neg[:train_size]
    val_neg = neg[train_size : train_size + val_size]
    test_neg = neg[train_size + val_size : train_size + val_size + test_size]

    def _pack(p_arr: np.ndarray, n_arr: np.ndarray) -> List[List[str]]:
        return [
            [str(p_arr[3 * i]), str(p_arr[3 * i + 1]), str(p_arr[3 * i + 2]), str(n_arr[i])]
            for i in range(len(n_arr))
        ]

    return _pack(train_pos, train_neg), _pack(val_pos, val_neg), _pack(test_pos, test_neg)


def _dump_split(base_dir: Path, task_name: str, split: str, data: List[List[str]]) -> None:
    task_dir = base_dir / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    with open(task_dir / f"{split}.pkl", "wb") as f:
        pickle.dump(data, f)


def is_syneval_family_cached(family: str, cache_dir: Union[str, Path]) -> bool:
    """Return ``True`` if all expected ``.pkl`` splits for *family* exist in *cache_dir*."""
    if family not in _FAMILY_TASK_ORDER:
        raise ValueError(
            f"Unknown SynEval family {family!r}. Choose from {sorted(_FAMILY_TASK_ORDER)}."
        )
    base = Path(cache_dir)
    names = _FAMILY_TASK_ORDER[family]
    for idx, task_name in enumerate(names):
        splits = ("val", "test") if idx >= 3 else ("train", "val", "test")
        for sp in splits:
            if not (base / task_name / f"{sp}.pkl").exists():
                return False
    return True


def generate_syneval_family_splits(
    family: str,
    raw_source: Union[str, Path],
    cache_dir: Union[str, Path],
    *,
    seed: int = 0,
    train_size: Optional[int] = None,
    val_size: Optional[int] = None,
    test_size: Optional[int] = None,
    part_val_size: Optional[int] = None,
    part_test_size: Optional[int] = None,
    overwrite: bool = False,
) -> Path:
    """Generate the 5 compositional SynEval subtask folders for *family* from *raw_source*."""
    if family not in _FAMILY_TASK_ORDER:
        raise ValueError(
            f"Unknown SynEval family {family!r}. Choose from {sorted(_FAMILY_TASK_ORDER)}."
        )

    out_path = Path(cache_dir)
    if not overwrite and is_syneval_family_cached(family, out_path):
        return out_path

    raw_path = Path(raw_source)
    sizes = SYNEVAL_PAPER_SIZES[family]
    tr_sz = sizes["train"] if train_size is None else train_size
    va_sz = sizes["val"] if val_size is None else val_size
    te_sz = sizes["test"] if test_size is None else test_size
    p_va_sz = sizes["part_val"] if part_val_size is None else part_val_size
    p_te_sz = sizes["part_test"] if part_test_size is None else part_test_size

    is_sv = family.startswith("sv_")
    is_singular = family.endswith("_singular")
    raw_files: Sequence[str] = _SV_RAW_FILES if is_sv else _ANAPHORA_RAW_FILES
    base_prefix = "singular_" if is_singular else "plural_"
    flip_prefix = "plural_" if is_singular else "singular_"

    ab_list: List[str] = []
    a_b_flip_list: List[str] = []
    a_flip_b_list: List[str] = []

    for datafile in raw_files:
        base_dict = _load_raw_pickle(raw_path, base_prefix + datafile)
        for pairs in base_dict.values():
            for gram, ungram in pairs:
                ab_list.append(gram)
                a_b_flip_list.append(ungram)

        flip_dict = _load_raw_pickle(raw_path, flip_prefix + datafile)
        for pairs in flip_dict.values():
            for _gram, ungram in pairs:
                a_flip_b_list.append(ungram)

    ab_sentences = np.asarray(ab_list, dtype=object)
    a_b_flip_sentences = np.asarray(a_b_flip_list, dtype=object)
    a_flip_b_sentences = np.asarray(a_flip_b_list, dtype=object)

    rng = np.random.default_rng(seed)
    base_neg = tr_sz + va_sz + te_sz
    base_pos = base_neg * 3
    part_neg = p_va_sz + p_te_sz
    part_pos = part_neg * 3

    half_base_neg = base_neg // 2
    other_half_base_neg = base_neg - half_base_neg
    half_base_pos = base_pos // 2
    other_half_base_pos = base_pos - half_base_pos

    random_ab = rng.choice(ab_sentences, base_pos + part_pos * 2, replace=False)
    random_a_flip = rng.choice(a_flip_b_sentences, half_base_neg + part_neg, replace=False)
    random_b_flip = rng.choice(a_b_flip_sentences, other_half_base_neg + part_neg, replace=False)

    names = _FAMILY_TASK_ORDER[family]

    # [0] Base compositional task (both properties)
    ab_train, ab_val, ab_test = _create_train_val_test(
        random_ab[:base_pos],
        np.concatenate([random_a_flip[:half_base_neg], random_b_flip[:other_half_base_neg]]),
        train_size=tr_sz,
        val_size=va_sz,
        test_size=te_sz,
        rng=rng,
    )
    _dump_split(out_path, names[0], "train", ab_train)
    _dump_split(out_path, names[0], "val", ab_val)
    _dump_split(out_path, names[0], "test", ab_test)

    # [1] Subroutine A mask dataset (S for SV; Antecedent for Anaphora)
    a_pos = np.concatenate(
        [
            rng.choice(ab_sentences, half_base_pos, replace=False),
            rng.choice(a_b_flip_sentences, other_half_base_pos, replace=False),
        ]
    )
    a_neg = rng.choice(a_flip_b_sentences, base_neg, replace=False)
    a_train, a_val, a_test = _create_train_val_test(
        a_pos, a_neg, train_size=tr_sz, val_size=va_sz, test_size=te_sz, rng=rng
    )
    _dump_split(out_path, names[1], "train", a_train)
    _dump_split(out_path, names[1], "val", a_val)
    _dump_split(out_path, names[1], "test", a_test)

    # [2] Subroutine B mask dataset (V for SV; Pronoun for Anaphora)
    b_pos = np.concatenate(
        [
            rng.choice(ab_sentences, half_base_pos, replace=False),
            rng.choice(a_flip_b_sentences, other_half_base_pos, replace=False),
        ]
    )
    b_neg = rng.choice(a_b_flip_sentences, base_neg, replace=False)
    b_train, b_val, b_test = _create_train_val_test(
        b_pos, b_neg, train_size=tr_sz, val_size=va_sz, test_size=te_sz, rng=rng
    )
    _dump_split(out_path, names[2], "train", b_train)
    _dump_split(out_path, names[2], "val", b_val)
    _dump_split(out_path, names[2], "test", b_test)

    # [3] Partition A ablation probe (distinguished only by property A)
    _, a_part_val, a_part_test = _create_train_val_test(
        random_ab[base_pos : base_pos + part_pos],
        random_a_flip[half_base_neg : half_base_neg + part_neg],
        train_size=0,
        val_size=p_va_sz,
        test_size=p_te_sz,
        rng=rng,
    )
    _dump_split(out_path, names[3], "val", a_part_val)
    _dump_split(out_path, names[3], "test", a_part_test)

    # [4] Partition B ablation probe (distinguished only by property B)
    _, b_part_val, b_part_test = _create_train_val_test(
        random_ab[base_pos + part_pos : base_pos + 2 * part_pos],
        random_b_flip[other_half_base_neg : other_half_base_neg + part_neg],
        train_size=0,
        val_size=p_va_sz,
        test_size=p_te_sz,
        rng=rng,
    )
    _dump_split(out_path, names[4], "val", b_part_val)
    _dump_split(out_path, names[4], "test", b_part_test)

    return out_path


def ensure_syneval_family_splits(
    family: str,
    cache_dir: Union[str, Path],
    *,
    raw_data_path: Optional[Union[str, Path]] = None,
    download: bool = False,
    url: str = SYNEVAL_RAWDATA_URL,
    seed: int = 0,
) -> Optional[Path]:
    """Ensure ``.pkl`` splits for *family* exist in *cache_dir*, generating from ``RawData.zip`` if available.

    Returns *cache_dir* if real splits are ready, or ``None`` if neither pre-generated
    splits nor ``RawData.zip`` are available and ``download=False``.
    """
    out_path = Path(cache_dir)
    if is_syneval_family_cached(family, out_path):
        return out_path

    candidates: List[Path] = []
    if raw_data_path is not None:
        candidates.append(Path(raw_data_path))
    candidates.extend([out_path / "RawData.zip", out_path / "RawData"])

    raw_source: Optional[Path] = next((p for p in candidates if p.exists()), None)
    if raw_source is None and download:
        raw_source = download_syneval_rawdata(out_path, url=url)

    if raw_source is None:
        return None

    return generate_syneval_family_splits(family, raw_source, out_path, seed=seed)
