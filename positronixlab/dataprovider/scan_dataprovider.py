"""SCAN sequence-to-sequence dataset provider with GitHub auto-download and local caching.

Downloads the official Lake & Baroni (2018) SCAN splits on demand from
``https://raw.githubusercontent.com/brendenlake/SCAN/master/`` (matching
Csordas et al., ICLR 2021: ``simple``, ``length``, ``jump``, ``turn_left``),
word-tokenizes commands and action sequences into padded ``LongTensor`` pairs,
and caches raw ``.txt`` files under ``<root>/scan/raw/`` and tokenized tensors
under ``<root>/scan/cache/``.

Legacy preprocessed ``<root>/scan/{train,test}.pt`` files continue to be loaded
automatically when present and ``split_type`` is not explicitly overridden.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Unpack

import numpy as np
import requests
import torch
from torch import nn

from positronixlab.dataprovider.dataprovider import (
    BaseTensorDataset,
    DataProvider,
    DataProviderOptions,
    TransformLike,
    resolve_data_root,
)

SCANSplitType = Literal["simple", "length", "jump", "turn_left"]

SCAN_BASE_URL: str = "https://raw.githubusercontent.com/brendenlake/SCAN/master"

SCAN_URLS: dict[str, dict[str, str]] = {
    "simple": {
        "train": f"{SCAN_BASE_URL}/simple_split/tasks_train_simple.txt",
        "test": f"{SCAN_BASE_URL}/simple_split/tasks_test_simple.txt",
    },
    "length": {
        "train": f"{SCAN_BASE_URL}/length_split/tasks_train_length.txt",
        "test": f"{SCAN_BASE_URL}/length_split/tasks_test_length.txt",
    },
    "jump": {
        "train": f"{SCAN_BASE_URL}/add_prim_split/tasks_train_addprim_jump.txt",
        "test": f"{SCAN_BASE_URL}/add_prim_split/tasks_test_addprim_jump.txt",
    },
    "turn_left": {
        "train": f"{SCAN_BASE_URL}/add_prim_split/tasks_train_addprim_turn_left.txt",
        "test": f"{SCAN_BASE_URL}/add_prim_split/tasks_test_addprim_turn_left.txt",
    },
}

# Fixed canonical SCAN vocabularies (invariant across splits).
SCAN_INPUT_WORDS: tuple[str, ...] = (
    "after",
    "and",
    "around",
    "jump",
    "left",
    "look",
    "opposite",
    "right",
    "run",
    "thrice",
    "turn",
    "twice",
    "walk",
)

SCAN_OUTPUT_WORDS: tuple[str, ...] = (
    "I_JUMP",
    "I_LOOK",
    "I_RUN",
    "I_TURN_LEFT",
    "I_TURN_RIGHT",
    "I_WALK",
)


class SCANVocab:
    """Deterministic word vocabulary for SCAN inputs or outputs."""

    PAD_IDX: int = 0
    SOS_IDX: int = 1
    EOS_IDX: int = 2
    UNK_IDX: int = 3
    SPECIAL_TOKENS: tuple[str, ...] = ("<pad>", "<sos>", "<eos>", "<unk>")

    def __init__(self, words: Sequence[str]) -> None:
        self.itos: tuple[str, ...] = self.SPECIAL_TOKENS + tuple(words)
        self.stoi: dict[str, int] = {tok: idx for idx, tok in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(
        self,
        sentence: str,
        *,
        max_len: int,
        add_sos: bool = False,
        add_eos: bool = True,
    ) -> np.ndarray:
        tokens = sentence.strip().split()
        ids = [self.stoi.get(w, self.UNK_IDX) for w in tokens]
        reserve = int(add_sos) + int(add_eos)
        max_body = max(0, max_len - reserve)
        ids = ids[:max_body]

        out = np.full(max_len, self.PAD_IDX, dtype=np.int64)
        pos = 0
        if add_sos and pos < max_len:
            out[pos] = self.SOS_IDX
            pos += 1
        end = pos + len(ids)
        if ids:
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
        if isinstance(tokens, torch.Tensor):
            seq = tokens.detach().cpu().tolist()
        elif isinstance(tokens, np.ndarray):
            seq = tokens.tolist()
        else:
            seq = list(tokens)

        words: list[str] = []
        for idx in seq:
            i = int(idx)
            if skip_special:
                if i == self.EOS_IDX:
                    break
                if i in (self.PAD_IDX, self.SOS_IDX):
                    continue
            words.append(self.itos[i] if 0 <= i < len(self.itos) else "<unk>")
        return " ".join(words)


def download_scan_file(
    root: str | Path,
    split_type: SCANSplitType,
    subset: Literal["train", "test"],
    *,
    timeout: int = 60,
    force: bool = False,
) -> Path:
    """Download a single SCAN ``.txt`` file from GitHub into ``<root>/scan/raw/`` (atomic)."""
    if split_type not in SCAN_URLS:
        raise ValueError(
            f"Unknown SCAN split_type {split_type!r}. Available: {sorted(SCAN_URLS.keys())}"
        )
    if subset not in SCAN_URLS[split_type]:
        raise ValueError(f"subset must be 'train' or 'test', got {subset!r}")

    url = SCAN_URLS[split_type][subset]
    filename = url.rsplit("/", 1)[-1]
    dest = resolve_data_root(root) / "scan" / "raw" / split_type / filename
    if dest.exists() and not force:
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".txt.tmp")
    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in response.iter_content(chunk_size=1 << 16):
                    if chunk:
                        f.write(chunk)
        tmp.rename(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise

    return dest


def _parse_scan_file(path: Path) -> tuple[list[str], list[str]]:
    """Parse ``IN: <cmd> OUT: <actions>`` lines from a SCAN ``.txt`` file."""
    inputs: list[str] = []
    outputs: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("OUT:")
            if len(parts) != 2:
                raise ValueError(f"Malformed SCAN line in {path}: {line!r}")
            inp = parts[0].replace("IN:", "", 1).strip()
            out = parts[1].strip()
            inputs.append(inp)
            outputs.append(out)
    return inputs, outputs


class SCANDataset(BaseTensorDataset):
    """SCAN sequence-to-sequence dataset with GitHub auto-download and local caching."""

    def __init__(
        self,
        train: bool = True,
        indices=None,
        transforms: TransformLike | None = None,
        root: str | Path | None = None,
        *,
        split_type: SCANSplitType = "simple",
        max_input_len: int = 16,
        max_output_len: int = 64,
        download: bool = True,
    ) -> None:
        self.train = train
        self.root = resolve_data_root(root)
        self.split_type = split_type
        self.in_vocab = SCANVocab(SCAN_INPUT_WORDS)
        self.out_vocab = SCANVocab(SCAN_OUTPUT_WORDS)
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

        subset: Literal["train", "test"] = "train" if train else "test"
        legacy_pt = self.root / "scan" / f"{subset}.pt"
        if split_type == "simple" and legacy_pt.exists():
            data = torch.load(legacy_pt, weights_only=True)
            super().__init__(data["inputs"], data["targets"], indices=indices, transforms=transforms)
            return

        if split_type not in SCAN_URLS:
            raise ValueError(
                f"Unknown SCAN split_type {split_type!r}. Available: {sorted(SCAN_URLS.keys())}"
            )

        sig = f"{split_type}|{subset}|in={max_input_len}|out={max_output_len}|v1"
        digest = hashlib.sha256(sig.encode("utf-8")).hexdigest()[:16]
        cache_path = self.root / "scan" / "cache" / f"scan_{digest}.pt"

        if cache_path.exists():
            cached = torch.load(cache_path, weights_only=True)
            super().__init__(cached["inputs"], cached["targets"], indices=indices, transforms=transforms)
            return

        filename = SCAN_URLS[split_type][subset].rsplit("/", 1)[-1]
        raw_path = self.root / "scan" / "raw" / split_type / filename
        if not raw_path.exists():
            if download:
                raw_path = download_scan_file(self.root, split_type, subset)
            else:
                raise FileNotFoundError(
                    f"SCAN data file not found at {raw_path} (or legacy {legacy_pt}). "
                    "Pass download=True to auto-download from GitHub."
                )

        in_lines, out_lines = _parse_scan_file(raw_path)
        n = len(in_lines)
        x_np = np.full((n, max_input_len), SCANVocab.PAD_IDX, dtype=np.int64)
        y_np = np.full((n, max_output_len), SCANVocab.PAD_IDX, dtype=np.int64)
        for i, (inp, out) in enumerate(zip(in_lines, out_lines, strict=False)):
            x_np[i] = self.in_vocab.encode(inp, max_len=max_input_len, add_sos=False, add_eos=True)
            y_np[i] = self.out_vocab.encode(out, max_len=max_output_len, add_sos=True, add_eos=True)

        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_cache = cache_path.with_suffix(".pt.tmp")
        torch.save({"inputs": x, "targets": y}, tmp_cache)
        tmp_cache.rename(cache_path)

        super().__init__(x, y, indices=indices, transforms=transforms)


class SCANDataProvider(
    DataProvider,
    plugin_name="scan",
    aliases=("csordas:scan",),
):
    """DataProvider for SCAN (Lake & Baroni 2018 / Csordas et al. ICLR 2021) with GitHub auto-download."""

    def __init__(
        self,
        *,
        split_type: SCANSplitType = "simple",
        max_input_len: int = 16,
        max_output_len: int = 64,
        download: bool = True,
        **kwargs: Unpack[DataProviderOptions],
    ) -> None:
        super().__init__(**kwargs)
        self.name = "scan"
        self.split_type = split_type
        self.download = download

        ds_kw: dict[str, Any] = dict(
            split_type=split_type,
            max_input_len=max_input_len,
            max_output_len=max_output_len,
            download=download,
        )
        self.dataset = SCANDataset(train=True, transforms=self.transforms, root=self.root, **ds_kw)
        self.in_vocab: SCANVocab = self.dataset.in_vocab
        self.out_vocab: SCANVocab = self.dataset.out_vocab
        self.num_features = int(self.dataset._full_x.shape[-1])
        self.num_classes = len(self.out_vocab)
        self.x_shape = tuple(self.dataset._full_x.shape[1:])
        self.y_shape = tuple(self.dataset._full_y.shape[1:])
        self.criterion = nn.CrossEntropyLoss(ignore_index=SCANVocab.PAD_IDX)

        test_ds = (
            SCANDataset(train=False, transforms=self.transforms, root=self.root, **ds_kw)
            if len(self.dataset_split) == 2
            else None
        )
        self._split_dataset(self.dataset, test_dataset=test_ds)
