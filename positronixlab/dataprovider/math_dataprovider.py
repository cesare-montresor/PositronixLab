"""DeepMind Mathematics dataset provider with Hugging Face auto-download, Csordas subtasks, and local caching.

Downloads pre-generated DeepMind Mathematics v1.0 ``.txt`` files on demand from
``https://huggingface.co/datasets/cesare-montresor/DeepMind-Mathematics-Dataset``
(avoiding GCS dependencies), deduplicates questions across splits following
Csordas et al. (ICLR 2021), character-tokenizes into fixed-length ``LongTensor``
sequences compatible with :class:`~positronixlab.baselines.csordas.CsordasTransformerEncDec`,
and exposes per-difficulty, per-task, ``iid``, ``interpolate``, and ``extrapolate``
subtasks via :class:`CompositionalDataProvider`.

Legacy preprocessed ``<root>/math/{train,test}.pt`` files continue to be loaded
automatically when present and ``tasks`` is not explicitly specified.
"""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import requests
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from positronixlab.dataprovider.dataprovider import (
    BaseTensorDataset,
    CompositionalDataProvider,
    SubtaskSplit,
    TransformLike,
    resolve_data_root,
)

HF_DM_MATH_REPO: str = "cesare-montresor/DeepMind-Mathematics-Dataset"
DEFAULT_DM_MATH_TASKS: tuple[str, ...] = ("arithmetic__add_or_sub",)
DM_MATH_DIFFICULTIES: tuple[str, ...] = ("easy", "medium", "hard")

# The 5 DeepMind Mathematics modules evaluated in Csordas et al. (ICLR 2021).
CSORDAS_DM_MATH_PAPER_TASKS: tuple[str, ...] = (
    "arithmetic__add_or_sub",
    "algebra__linear_1d",
    "calculus__differentiate",
    "comparison__sort",
    "polynomials__collect",
)

# All 56 DeepMind Mathematics v1.0 modules (train-{easy,medium,hard} & interpolate).
DM_MATH_TASKS: tuple[str, ...] = (
    "algebra__linear_1d",
    "algebra__linear_1d_composed",
    "algebra__linear_2d",
    "algebra__linear_2d_composed",
    "algebra__polynomial_roots",
    "algebra__polynomial_roots_composed",
    "algebra__sequence_next_term",
    "algebra__sequence_nth_term",
    "arithmetic__add_or_sub",
    "arithmetic__add_or_sub_in_base",
    "arithmetic__add_sub_multiple",
    "arithmetic__div",
    "arithmetic__mixed",
    "arithmetic__mul",
    "arithmetic__mul_div_multiple",
    "arithmetic__nearest_integer_root",
    "arithmetic__simplify_surd",
    "calculus__differentiate",
    "calculus__differentiate_composed",
    "comparison__closest",
    "comparison__closest_composed",
    "comparison__kth_biggest",
    "comparison__kth_biggest_composed",
    "comparison__pair",
    "comparison__pair_composed",
    "comparison__sort",
    "comparison__sort_composed",
    "measurement__conversion",
    "measurement__time",
    "numbers__base_conversion",
    "numbers__div_remainder",
    "numbers__div_remainder_composed",
    "numbers__gcd",
    "numbers__gcd_composed",
    "numbers__is_factor",
    "numbers__is_factor_composed",
    "numbers__is_prime",
    "numbers__is_prime_composed",
    "numbers__lcm",
    "numbers__lcm_composed",
    "numbers__list_prime_factors",
    "numbers__list_prime_factors_composed",
    "numbers__place_value",
    "numbers__place_value_composed",
    "numbers__round_number",
    "numbers__round_number_composed",
    "polynomials__add",
    "polynomials__coefficient_named",
    "polynomials__collect",
    "polynomials__compose",
    "polynomials__evaluate",
    "polynomials__evaluate_composed",
    "polynomials__expand",
    "polynomials__simplify_power",
    "probability__swr_p_level_set",
    "probability__swr_p_sequence",
)

# Mapping from base module name -> extrapolate filename stem (15 official modules).
DM_MATH_EXTRAPOLATE_MAP: dict[str, str] = {
    "algebra__polynomial_roots": "algebra__polynomial_roots_big",
    "arithmetic__add_or_sub": "arithmetic__add_or_sub_big",
    "arithmetic__add_sub_multiple": "arithmetic__add_sub_multiple_longer",
    "arithmetic__div": "arithmetic__div_big",
    "arithmetic__mixed": "arithmetic__mixed_longer",
    "arithmetic__mul": "arithmetic__mul_big",
    "arithmetic__mul_div_multiple": "arithmetic__mul_div_multiple_longer",
    "comparison__closest": "comparison__closest_more",
    "comparison__kth_biggest": "comparison__kth_biggest_more",
    "comparison__sort": "comparison__sort_more",
    "measurement__conversion": "measurement__conversion",
    "numbers__place_value": "numbers__place_value_big",
    "numbers__round_number": "numbers__round_number_big",
    "probability__swr_p_level_set": "probability__swr_p_level_set_more_samples",
    "probability__swr_p_sequence": "probability__swr_p_sequence_more_samples",
}


class MathVocab:
    """Deterministic printable-ASCII character vocabulary for DeepMind Mathematics.

    Token IDs are invariant across task subsets and splits:
    - ``0``: ``<pad>``
    - ``1``: ``<sos>``
    - ``2``: ``<eos>``
    - ``3``: ``<unk>``
    - ``4..98``: printable ASCII characters ``chr(32)`` .. ``chr(126)``
    """

    PAD_IDX: int = 0
    SOS_IDX: int = 1
    EOS_IDX: int = 2
    UNK_IDX: int = 3
    SPECIAL_TOKENS: tuple[str, ...] = ("<pad>", "<sos>", "<eos>", "<unk>")

    def __init__(self) -> None:
        ascii_chars = tuple(chr(c) for c in range(32, 127))
        self.itos: tuple[str, ...] = self.SPECIAL_TOKENS + ascii_chars
        self.stoi: dict[str, int] = {tok: idx for idx, tok in enumerate(self.itos)}
        self._byte_lut = np.full(256, self.UNK_IDX, dtype=np.int64)
        for c in range(32, 127):
            self._byte_lut[c] = len(self.SPECIAL_TOKENS) + (c - 32)

    def __len__(self) -> int:
        return len(self.itos)

    def encode(
        self,
        text: str,
        *,
        max_len: int,
        add_sos: bool = False,
        add_eos: bool = True,
    ) -> np.ndarray:
        """Encode *text* into a fixed-length ``int64`` token array padded with ``PAD_IDX``."""
        raw = np.frombuffer(text.encode("latin-1", errors="replace"), dtype=np.uint8)
        ids = self._byte_lut[raw]

        reserve = int(add_sos) + int(add_eos)
        max_body = max(0, max_len - reserve)
        if len(ids) > max_body:
            ids = ids[:max_body]

        out = np.full(max_len, self.PAD_IDX, dtype=np.int64)
        pos = 0
        if add_sos and pos < max_len:
            out[pos] = self.SOS_IDX
            pos += 1
        end = pos + len(ids)
        out[pos:end] = ids
        if add_eos and end < max_len:
            out[end] = self.EOS_IDX
        return out

    def decode(
        self,
        tokens: torch.Tensor | np.ndarray | Sequence[int],
        *,
        skip_special: bool = True,
    ) -> str:
        """Decode token IDs back to a string, stopping at ``EOS_IDX`` when *skip_special* is True."""
        if isinstance(tokens, torch.Tensor):
            seq = tokens.detach().cpu().tolist()
        elif isinstance(tokens, np.ndarray):
            seq = tokens.tolist()
        else:
            seq = list(tokens)

        chars: list[str] = []
        for idx in seq:
            i = int(idx)
            if skip_special:
                if i == self.EOS_IDX:
                    break
                if i in (self.PAD_IDX, self.SOS_IDX):
                    continue
            if 0 <= i < len(self.itos):
                chars.append(self.itos[i])
            else:
                chars.append("<unk>")
        return "".join(chars)


class MathSeq2SeqLoss(nn.Module):
    """Teacher-forced cross-entropy loss for DeepMind Math sequence prediction.

    Accepts ``logits`` of shape ``(B, T, V)`` and ``targets`` of shape either
    ``(B, T)`` (already shifted) or ``(B, T + 1)`` (full ``[<sos>, ..., <eos>]``,
    in which case ``targets[:, 1:]`` is used automatically).
    """

    def __init__(self, pad_idx: int = MathVocab.PAD_IDX) -> None:
        super().__init__()
        self.pad_idx = pad_idx

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.ndim == 2 and logits.ndim == 3 and targets.size(1) == logits.size(1) + 1:
            targets = targets[:, 1:]
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=self.pad_idx,
        )


def math_token_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    pad_idx: int = MathVocab.PAD_IDX,
) -> float:
    """Non-padding per-token accuracy for DeepMind Math seq2seq outputs."""
    if targets.ndim == 2 and logits.ndim == 3 and targets.size(1) == logits.size(1) + 1:
        targets = targets[:, 1:]
    preds = logits.argmax(dim=-1)
    mask = targets != pad_idx
    total = int(mask.sum().item())
    if total == 0:
        return 0.0
    correct = int(((preds == targets) & mask).sum().item())
    return correct / total


def math_exact_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    pad_idx: int = MathVocab.PAD_IDX,
) -> float:
    """Exact-sequence match accuracy (all non-pad target tokens match), matching Csordas."""
    if targets.ndim == 2 and logits.ndim == 3 and targets.size(1) == logits.size(1) + 1:
        targets = targets[:, 1:]
    preds = logits.argmax(dim=-1)
    mask = targets != pad_idx
    seq_match = ((preds == targets) | (~mask)).all(dim=-1)
    return float(seq_match.float().mean().item()) if seq_match.numel() > 0 else 0.0


def _normalize_tasks(tasks: str | Sequence[str] | None) -> tuple[str, ...]:
    if tasks is None:
        return DEFAULT_DM_MATH_TASKS
    if isinstance(tasks, str):
        if tasks == "all":
            return DM_MATH_TASKS
        if tasks == "paper":
            return CSORDAS_DM_MATH_PAPER_TASKS
        return (tasks,)
    resolved = tuple(tasks)
    if not resolved:
        raise ValueError("tasks must be a non-empty sequence of DeepMind Math module names.")
    return resolved


def _normalize_difficulties(difficulties: Sequence[str]) -> tuple[str, ...]:
    out = tuple(d[len("train-") :] if d.startswith("train-") else d for d in difficulties)
    if not out:
        raise ValueError("difficulties must be non-empty (e.g. ('easy', 'medium', 'hard')).")
    return out


def _resolve_split_files(
    tasks: Sequence[str],
    splits: Sequence[str],
    *,
    warn_missing_extrapolate: bool = True,
) -> list[tuple[str, str]]:
    """Return ``(split, filename_stem)`` pairs for the requested *tasks* and *splits*."""
    pairs: list[tuple[str, str]] = []
    for split in splits:
        for task in tasks:
            if split == "extrapolate":
                stem = DM_MATH_EXTRAPOLATE_MAP.get(task)
                if stem is None:
                    if task in DM_MATH_EXTRAPOLATE_MAP.values():
                        stem = task
                    else:
                        if warn_missing_extrapolate:
                            warnings.warn(
                                f"DeepMind Math task {task!r} has no extrapolate split; skipping.",
                                UserWarning,
                                stacklevel=3,
                            )
                        continue
                pairs.append((split, stem))
            else:
                pairs.append((split, task))
    return pairs


def download_dm_math_file(
    root: str | Path,
    split: str,
    task_stem: str,
    *,
    repo_id: str = HF_DM_MATH_REPO,
    revision: str = "main",
    timeout: int = 60,
    force: bool = False,
    show_progress: bool = True,
) -> Path:
    """Download a single DeepMind Mathematics ``.txt`` file from Hugging Face (atomic).

    Files are stored at ``<root>/math/raw/<split>/<task_stem>.txt``.
    """
    dest = resolve_data_root(root) / "math" / "raw" / split / f"{task_stem}.txt"
    if dest.exists() and not force:
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{split}/{task_stem}.txt"
    tmp = dest.with_suffix(".txt.tmp")

    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0)) or None
            with (
                open(tmp, "wb") as f,
                tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"dm_math {split}/{task_stem}",
                    disable=not show_progress,
                ) as pbar,
            ):
                for chunk in response.iter_content(chunk_size=1 << 18):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        tmp.rename(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise

    return dest


def download_dm_math(
    root: str | Path,
    tasks: str | Sequence[str] = DEFAULT_DM_MATH_TASKS,
    splits: Sequence[str] = ("train-easy", "train-medium", "train-hard", "interpolate"),
    *,
    repo_id: str = HF_DM_MATH_REPO,
    revision: str = "main",
    timeout: int = 60,
    force: bool = False,
    show_progress: bool = True,
) -> list[Path]:
    """Download all requested DeepMind Mathematics ``.txt`` files from Hugging Face."""
    task_list = _normalize_tasks(tasks)
    pairs = _resolve_split_files(task_list, splits)
    return [
        download_dm_math_file(
            root,
            split,
            stem,
            repo_id=repo_id,
            revision=revision,
            timeout=timeout,
            force=force,
            show_progress=show_progress,
        )
        for split, stem in pairs
    ]


def _parse_qa_file(
    path: Path,
    *,
    max_samples: int | None = None,
    seen_questions: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Read alternating ``question`` / ``answer`` lines with optional cross-split deduplication."""
    questions: list[str] = []
    answers: list[str] = []
    with open(path, encoding="utf-8") as f:
        while True:
            q_line = f.readline()
            if not q_line:
                break
            a_line = f.readline()
            if not a_line:
                break
            q = q_line.rstrip("\r\n")
            a = a_line.rstrip("\r\n")
            if seen_questions is not None:
                if q in seen_questions:
                    continue
                seen_questions.add(q)
            questions.append(q)
            answers.append(a)
            if max_samples is not None and len(questions) >= max_samples:
                break
    return questions, answers


def _tokenize_pairs(
    questions: Sequence[str],
    answers: Sequence[str],
    vocab: MathVocab,
    *,
    max_question_len: int = 160,
    max_answer_len: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert question/answer strings to padded ``LongTensor`` arrays."""
    n = len(questions)
    x_np = np.full((n, max_question_len), vocab.PAD_IDX, dtype=np.int64)
    y_np = np.full((n, max_answer_len), vocab.PAD_IDX, dtype=np.int64)
    for i, (q, a) in enumerate(zip(questions, answers, strict=False)):
        x_np[i] = vocab.encode(q, max_len=max_question_len, add_sos=False, add_eos=True)
        y_np[i] = vocab.encode(a, max_len=max_answer_len, add_sos=True, add_eos=True)
    return torch.from_numpy(x_np), torch.from_numpy(y_np)


def _carve_iid_test(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    test_size: int = 10_000,
    seed: int = 1234,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Deterministically carve an IID held-out test set from a difficulty split (Csordas seed=1234)."""
    n = int(len(x))
    if n <= 1:
        return (x, y), (x[:0], y[:0])
    # Match Csordas 10k held-out test when full file is loaded; cap at 20% for small subsamples.
    n_test = min(test_size, max(1, n // 5)) if n <= test_size else test_size
    rng = np.random.RandomState(seed)
    test_idx = np.sort(rng.choice(n, size=n_test, replace=False))
    mask = np.ones(n, dtype=bool)
    mask[test_idx] = False
    train_idx = np.nonzero(mask)[0]
    return (x[train_idx], y[train_idx]), (x[test_idx], y[test_idx])


def _cache_file_path(
    root: Path,
    pairs: Sequence[tuple[str, str]],
    max_question_len: int,
    max_answer_len: int,
    max_samples_per_file: int | None,
    deduplicate: bool = True,
) -> Path:
    sig_parts = [
        ",".join(f"{s}/{t}" for s, t in pairs),
        f"q={max_question_len}",
        f"a={max_answer_len}",
        f"n={max_samples_per_file}",
        f"dedup={int(deduplicate)}",
        "v2",
    ]
    digest = hashlib.sha256("|".join(sig_parts).encode("utf-8")).hexdigest()[:16]
    return root / "math" / "cache" / f"dm_math_{digest}.pt"


def _ensure_raw_files(
    root: Path,
    pairs: Sequence[tuple[str, str]],
    *,
    download: bool,
    repo_id: str,
    revision: str,
    show_progress: bool,
) -> tuple[list[Path], list[tuple[str, str]]]:
    """Return ``(raw_paths, missing_pairs)`` after optional Hugging Face auto-download."""
    missing: list[tuple[str, str]] = []
    for split, stem in pairs:
        p = root / "math" / "raw" / split / f"{stem}.txt"
        if not p.exists():
            missing.append((split, stem))

    if missing and download:
        for split, stem in missing:
            download_dm_math_file(
                root,
                split,
                stem,
                repo_id=repo_id,
                revision=revision,
                show_progress=show_progress,
            )
        missing = []

    raw_paths = [root / "math" / "raw" / split / f"{stem}.txt" for split, stem in pairs]
    return raw_paths, missing


def _load_or_build_tensors(
    root: Path,
    pairs: Sequence[tuple[str, str]],
    vocab: MathVocab,
    *,
    max_question_len: int,
    max_answer_len: int,
    max_samples_per_file: int | None,
    deduplicate: bool,
    download: bool,
    repo_id: str,
    revision: str,
    show_progress: bool,
    seen_questions: set[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load cached tokenized tensors or parse + tokenize raw ``.txt`` files."""
    use_cache = seen_questions is None
    cache_path = _cache_file_path(
        root, pairs, max_question_len, max_answer_len, max_samples_per_file, deduplicate
    )
    if use_cache and cache_path.exists():
        cached = torch.load(cache_path, weights_only=True)
        return cached["inputs"], cached["targets"]

    raw_paths, missing = _ensure_raw_files(
        root,
        pairs,
        download=download,
        repo_id=repo_id,
        revision=revision,
        show_progress=show_progress,
    )
    if missing:
        missing_str = ", ".join(f"{s}/{t}.txt" for s, t in missing)
        raise FileNotFoundError(
            f"Missing DeepMind Math files under {root / 'math' / 'raw'}: "
            f"{missing_str}. Pass download=True to auto-download from {repo_id}."
        )

    seen = seen_questions if seen_questions is not None else (set() if deduplicate else None)
    all_q: list[str] = []
    all_a: list[str] = []
    for p in raw_paths:
        q, a = _parse_qa_file(p, max_samples=max_samples_per_file, seen_questions=seen)
        all_q.extend(q)
        all_a.extend(a)

    x, y = _tokenize_pairs(
        all_q,
        all_a,
        vocab,
        max_question_len=max_question_len,
        max_answer_len=max_answer_len,
    )

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_cache = cache_path.with_suffix(".pt.tmp")
        torch.save({"inputs": x, "targets": y}, tmp_cache)
        tmp_cache.rename(cache_path)

    return x, y


class MathDataset(BaseTensorDataset):
    """DeepMind Mathematics tensor dataset with Hugging Face auto-download and local caching.

    Resolution order:
    1. If ``tasks is None`` and legacy ``<root>/math/{train,test}.pt`` exists, loads it directly.
    2. Otherwise resolves requested ``tasks`` and ``splits``:
       - Uses cached tokenized ``.pt`` under ``<root>/math/cache/`` when available.
       - Loads raw ``.txt`` from ``<root>/math/raw/<split>/<task>.txt``, auto-downloading from
         Hugging Face (``cesare-montresor/DeepMind-Mathematics-Dataset``) if ``download=True``.
       - Raises ``FileNotFoundError`` when ``download=False`` and required files are missing.
    """

    def __init__(
        self,
        train: bool = True,
        indices=None,
        transforms: TransformLike | None = None,
        root: str | Path | None = None,
        *,
        tasks: str | Sequence[str] | None = None,
        difficulties: Sequence[str] = DM_MATH_DIFFICULTIES,
        eval_split: Literal["interpolate", "extrapolate"] = "interpolate",
        max_question_len: int = 160,
        max_answer_len: int = 32,
        max_samples_per_file: int | None = None,
        deduplicate: bool = True,
        download: bool = True,
        repo_id: str = HF_DM_MATH_REPO,
        revision: str = "main",
        show_progress: bool = True,
    ) -> None:
        self.train = train
        self.root = resolve_data_root(root)
        self.vocab = MathVocab()
        self.max_question_len = max_question_len
        self.max_answer_len = max_answer_len
        self.max_samples_per_file = max_samples_per_file

        legacy_pt = self.root / "math" / ("train.pt" if train else "test.pt")
        if tasks is None and legacy_pt.exists():
            data = torch.load(legacy_pt, weights_only=True)
            x = data["inputs"]
            y = data["targets"]
            self.tasks: tuple[str, ...] = ()
            self.splits: tuple[str, ...] = ()
            super().__init__(x, y, indices=indices, transforms=transforms)
            return

        resolved_tasks = _normalize_tasks(tasks)
        diffs = _normalize_difficulties(difficulties)
        splits = tuple(f"train-{d}" for d in diffs) if train else (eval_split,)
        pairs = _resolve_split_files(resolved_tasks, splits)
        self.tasks = resolved_tasks
        self.splits = splits

        if not pairs:
            raise ValueError(
                f"No valid DeepMind Math files for tasks={resolved_tasks!r}, splits={splits!r}."
            )

        x, y = _load_or_build_tensors(
            self.root,
            pairs,
            self.vocab,
            max_question_len=max_question_len,
            max_answer_len=max_answer_len,
            max_samples_per_file=max_samples_per_file,
            deduplicate=deduplicate,
            download=download,
            repo_id=repo_id,
            revision=revision,
            show_progress=show_progress,
        )

        super().__init__(x, y, indices=indices, transforms=transforms)


def _concat_tensors(chunks: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    xs = [c[0] for c in chunks if len(c[0]) > 0]
    ys = [c[1] for c in chunks if len(c[1]) > 0]
    if not xs:
        return torch.empty((0, 0), dtype=torch.long), torch.empty((0, 0), dtype=torch.long)
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


class MathDataProvider(
    CompositionalDataProvider,
    plugin_name="deepmind_math",
    aliases=("csordas:deepmind_math",),
):
    """Compositional ``DataProvider`` for DeepMind Mathematics v1.0 matching Csordas et al. (ICLR 2021).

    Exposes:
    - ``full``: combined training across ``difficulties``, validated on ``interpolate`` (or IID split),
      tested on ``eval_split`` / IID test.
    - Per-difficulty subtasks (``easy``, ``medium``, ``hard``): each with its own deduplicated
      train split and held-out IID test split (enabling Csordas's ``masks_splits=["easy"]`` vs
      ``hard`` subnetwork discovery).
    - Evaluation subtasks: ``iid``, ``interpolate``, and ``extrapolate`` (when available for at
      least one requested module).
    - Per-task subtasks (when ``len(tasks) > 1``): ``subtasks[task_name]`` for per-module
      subnetwork discovery and cross-module evaluation.
    """

    source = "csordas"
    task_type = "seq2seq"

    def __init__(
        self,
        *,
        tasks: str | Sequence[str] | None = None,
        difficulties: Sequence[str] = DM_MATH_DIFFICULTIES,
        eval_split: Literal["interpolate", "extrapolate"] = "interpolate",
        include_extrapolate: bool = False,
        test_size_per_difficulty: int = 10_000,
        test_seed: int = 1234,
        max_question_len: int = 160,
        max_answer_len: int = 32,
        max_samples_per_file: int | None = None,
        deduplicate: bool = True,
        download: bool = True,
        repo_id: str = HF_DM_MATH_REPO,
        revision: str = "main",
        show_progress: bool = True,
        batch_size: int = 128,
        **kwargs: Any,
    ) -> None:
        super().__init__(name="deepmind_math", batch_size=batch_size, **kwargs)
        self.tasks_arg = tasks
        self.difficulties = _normalize_difficulties(difficulties)
        self.eval_split = eval_split
        self.include_extrapolate = include_extrapolate or (eval_split == "extrapolate")
        self.test_size_per_difficulty = test_size_per_difficulty
        self.test_seed = test_seed
        self.max_question_len = max_question_len
        self.max_answer_len = max_answer_len
        self.max_samples_per_file = max_samples_per_file
        self.deduplicate = deduplicate
        self.download = download
        self.repo_id = repo_id
        self.revision = revision

        self.vocab = MathVocab()
        self.vocab_size = len(self.vocab)
        self.criterion = MathSeq2SeqLoss(pad_idx=MathVocab.PAD_IDX)

        legacy_pt = self.root / "math" / "train.pt"
        if tasks is None and legacy_pt.exists():
            self._init_legacy_pt()
            return

        self._init_compositional(show_progress=show_progress)

    def _split_kwargs(self) -> dict[str, Any]:
        return dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            indexed_dataset=self.indexed_dataset,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )

    def _init_legacy_pt(self) -> None:
        """Backward-compatible path for legacy preprocessed ``train.pt``/``test.pt``."""
        self.tasks: tuple[str, ...] = ()
        self.dataset = MathDataset(
            train=True,
            transforms=self.transforms,
            root=self.root,
            download=False,
        )
        test_ds = (
            MathDataset(
                train=False,
                transforms=self.transforms,
                root=self.root,
                download=False,
            )
            if len(self.dataset_split) == 2
            else None
        )
        self.num_features = int(self.dataset._full_x.shape[-1])
        self.num_classes = self.vocab_size
        self.x_shape = tuple(self.dataset._full_x.shape[1:])
        self.y_shape = tuple(self.dataset._full_y.shape[1:])
        self._split_dataset(self.dataset, test_dataset=test_ds)
        self.subtasks["full"] = SubtaskSplit(
            name="full",
            dataset_train=self.dataset_train,
            dataset_val=self.dataset_val,
            dataset_test=self.dataset_test,
            **self._split_kwargs(),
        )

    def _init_compositional(self, *, show_progress: bool) -> None:
        resolved_tasks = _normalize_tasks(self.tasks_arg)
        self.tasks = resolved_tasks

        # Check combined compositional cache first so repeat runs skip parsing/dedup.
        cache_sig = "|".join(
            [
                "comp",
                ",".join(resolved_tasks),
                ",".join(self.difficulties),
                f"eval={self.eval_split}",
                f"extrap={int(self.include_extrapolate)}",
                f"q={self.max_question_len}",
                f"a={self.max_answer_len}",
                f"n={self.max_samples_per_file}",
                f"ts={self.test_size_per_difficulty}",
                f"tseed={self.test_seed}",
                f"dedup={int(self.deduplicate)}",
                "v1",
            ]
        )
        cache_digest = hashlib.sha256(cache_sig.encode("utf-8")).hexdigest()[:16]
        comp_cache = self.root / "math" / "cache" / f"dm_math_comp_{cache_digest}.pt"

        task_diff_train: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]]
        task_diff_test: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]]
        task_interp: dict[str, tuple[torch.Tensor, torch.Tensor]]
        task_extrap: dict[str, tuple[torch.Tensor, torch.Tensor]]

        if comp_cache.exists():
            payload = torch.load(comp_cache, weights_only=True)
            task_diff_train = payload["task_diff_train"]
            task_diff_test = payload["task_diff_test"]
            task_interp = payload["task_interp"]
            task_extrap = payload["task_extrap"]
        else:
            task_diff_train = {}
            task_diff_test = {}
            task_interp = {}
            task_extrap = {}

            for task in resolved_tasks:
                seen: set[str] | None = set() if self.deduplicate else None
                task_diff_train[task] = {}
                task_diff_test[task] = {}

                for diff in self.difficulties:
                    x_raw, y_raw = _load_or_build_tensors(
                        self.root,
                        [(f"train-{diff}", task)],
                        self.vocab,
                        max_question_len=self.max_question_len,
                        max_answer_len=self.max_answer_len,
                        max_samples_per_file=self.max_samples_per_file,
                        deduplicate=self.deduplicate,
                        download=self.download,
                        repo_id=self.repo_id,
                        revision=self.revision,
                        show_progress=show_progress,
                        seen_questions=seen,
                    )
                    tr_xy, te_xy = _carve_iid_test(
                        x_raw,
                        y_raw,
                        test_size=self.test_size_per_difficulty,
                        seed=self.test_seed,
                    )
                    task_diff_train[task][diff] = tr_xy
                    task_diff_test[task][diff] = te_xy

                if self.eval_split == "interpolate" or not self.include_extrapolate:
                    task_interp[task] = _load_or_build_tensors(
                        self.root,
                        [("interpolate", task)],
                        self.vocab,
                        max_question_len=self.max_question_len,
                        max_answer_len=self.max_answer_len,
                        max_samples_per_file=self.max_samples_per_file,
                        deduplicate=self.deduplicate,
                        download=self.download,
                        repo_id=self.repo_id,
                        revision=self.revision,
                        show_progress=show_progress,
                        seen_questions=seen,
                    )

                if self.include_extrapolate:
                    extrap_pairs = _resolve_split_files(
                        [task],
                        ["extrapolate"],
                        warn_missing_extrapolate=(self.eval_split == "extrapolate"),
                    )
                    if extrap_pairs:
                        task_extrap[task] = _load_or_build_tensors(
                            self.root,
                            extrap_pairs,
                            self.vocab,
                            max_question_len=self.max_question_len,
                            max_answer_len=self.max_answer_len,
                            max_samples_per_file=self.max_samples_per_file,
                            deduplicate=self.deduplicate,
                            download=self.download,
                            repo_id=self.repo_id,
                            revision=self.revision,
                            show_progress=show_progress,
                            seen_questions=seen,
                        )

            comp_cache.parent.mkdir(parents=True, exist_ok=True)
            tmp_cache = comp_cache.with_suffix(".pt.tmp")
            torch.save(
                {
                    "task_diff_train": task_diff_train,
                    "task_diff_test": task_diff_test,
                    "task_interp": task_interp,
                    "task_extrap": task_extrap,
                },
                tmp_cache,
            )
            tmp_cache.rename(comp_cache)

        def _make_ds(xy: tuple[torch.Tensor, torch.Tensor]) -> BaseTensorDataset:
            return BaseTensorDataset(xy[0], xy[1], transforms=self.transforms)

        # Combine across all requested tasks & difficulties
        full_train_xy = _concat_tensors(
            [task_diff_train[t][d] for t in resolved_tasks for d in self.difficulties]
        )
        full_iid_xy = _concat_tensors(
            [task_diff_test[t][d] for t in resolved_tasks for d in self.difficulties]
        )
        full_interp_xy = (
            _concat_tensors([task_interp[t] for t in resolved_tasks if t in task_interp])
            if task_interp
            else None
        )
        full_extrap_xy = (
            _concat_tensors([task_extrap[t] for t in resolved_tasks if t in task_extrap])
            if task_extrap
            else None
        )

        full_train_ds = _make_ds(full_train_xy)
        full_iid_ds = _make_ds(full_iid_xy)
        full_interp_ds = _make_ds(full_interp_xy) if full_interp_xy is not None and len(full_interp_xy[0]) > 0 else None
        full_extrap_ds = _make_ds(full_extrap_xy) if full_extrap_xy is not None and len(full_extrap_xy[0]) > 0 else None

        self.num_features = int(full_train_xy[0].shape[-1]) if full_train_xy[0].ndim == 2 else self.max_question_len
        self.num_classes = self.vocab_size
        self.x_shape = (self.max_question_len,)
        self.y_shape = (self.max_answer_len,)

        split_kw = self._split_kwargs()

        # Choose primary eval test split
        primary_test_ds = (
            full_extrap_ds
            if (self.eval_split == "extrapolate" and full_extrap_ds is not None)
            else (full_interp_ds if full_interp_ds is not None else full_iid_ds)
        )
        primary_val_ds = full_iid_ds if len(full_iid_ds) > 0 else primary_test_ds

        if len(self.dataset_split) == 3:
            self.dataset = full_train_ds
            self._split_dataset(self.dataset)
            self.subtasks["full"] = SubtaskSplit(
                name="full",
                dataset_train=self.dataset_train,
                dataset_val=self.dataset_val,
                dataset_test=self.dataset_test,
                **split_kw,
            )
        else:
            self.subtasks["full"] = SubtaskSplit(
                name="full",
                dataset_train=full_train_ds,
                dataset_val=primary_val_ds,
                dataset_test=primary_test_ds,
                **split_kw,
            )
            self._sync_base_split()

        # Evaluation subtasks
        eval_targets: list[str] = []
        if len(full_iid_ds) > 0:
            self.subtasks["iid"] = SubtaskSplit(
                name="iid",
                dataset_train=None,
                dataset_val=full_iid_ds,
                dataset_test=full_iid_ds,
                **split_kw,
            )
            eval_targets.append("iid")

        if full_interp_ds is not None:
            self.subtasks["interpolate"] = SubtaskSplit(
                name="interpolate",
                dataset_train=None,
                dataset_val=full_interp_ds,
                dataset_test=full_interp_ds,
                **split_kw,
            )
            eval_targets.append("interpolate")

        if full_extrap_ds is not None:
            self.subtasks["extrapolate"] = SubtaskSplit(
                name="extrapolate",
                dataset_train=None,
                dataset_val=full_extrap_ds,
                dataset_test=full_extrap_ds,
                **split_kw,
            )
            eval_targets.append("extrapolate")

        # Per-difficulty subtasks (matches Csordas masks_splits: easy vs medium vs hard)
        for diff in self.difficulties:
            d_train = _make_ds(_concat_tensors([task_diff_train[t][diff] for t in resolved_tasks]))
            d_test = _make_ds(_concat_tensors([task_diff_test[t][diff] for t in resolved_tasks]))
            self.subtasks[diff] = SubtaskSplit(
                name=diff,
                dataset_train=d_train,
                dataset_val=d_test,
                dataset_test=d_test,
                **split_kw,
            )

        # Per-task subtasks when training on multiple modules simultaneously
        if len(resolved_tasks) > 1:
            for task in resolved_tasks:
                t_train = _make_ds(_concat_tensors([task_diff_train[task][d] for d in self.difficulties]))
                t_iid = _make_ds(_concat_tensors([task_diff_test[task][d] for d in self.difficulties]))
                t_eval = (
                    _make_ds(task_interp[task])
                    if task in task_interp and len(task_interp[task][0]) > 0
                    else t_iid
                )
                self.subtasks[task] = SubtaskSplit(
                    name=task,
                    dataset_train=t_train,
                    dataset_val=t_iid,
                    dataset_test=t_eval,
                    **split_kw,
                )
            self._eval_matrix = {t: list(resolved_tasks) + eval_targets for t in resolved_tasks}
        else:
            diff_eval = list(self.difficulties) + eval_targets
            self._eval_matrix = {d: diff_eval for d in self.difficulties}
