"""Crash-safe, multiprocess-safe baseline checkpoint store and artifact loader."""

from __future__ import annotations

import fcntl
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from positronixlab.baselines.loader import ModelLoader
from positronixlab.tasks.masks import SubnetworkMask
from positronixlab.utils.utils import atomic_write_json


class CheckpointStore:
    """Crash-safe, multiprocess-safe stage & per-subtask checkpoint manager.

    Directory layout inside ``work_dir``:
    - ``checkpoint_state.json``: stage & subtask completion flags + timings
    - ``model_bundle.pt``: ``ModelLoader.save(...)`` bundle
    - ``masks/<method>/<subtask>.pt``: ``SubnetworkMask.save(...)`` artifacts
    - ``baseline_manifest.json``: canonical index consumed by downstream eval/HPO

    All paths inside ``work_dir`` are stored as portable POSIX paths relative to
    ``work_dir`` so directories can be moved or mounted at a different root.
    """

    STATE_FILENAME = "checkpoint_state.json"
    LOCK_FILENAME = ".checkpoint_state.lock"
    MANIFEST_FILENAME = "baseline_manifest.json"

    def __init__(self, work_dir: str | Path, export_filename: str = "model_bundle.pt") -> None:
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.work_dir / self.STATE_FILENAME
        self.lock_path = self.work_dir / self.LOCK_FILENAME
        self.manifest_path = self.work_dir / self.MANIFEST_FILENAME
        self.model_bundle_path = self.work_dir / export_filename
        self.masks_dir = self.work_dir / "masks"

    def _relativize(self, path: str | Path) -> str:
        resolved = Path(path).resolve()
        try:
            return resolved.relative_to(self.work_dir).as_posix()
        except ValueError:
            return str(resolved)

    def _resolve(self, stored: str | Path) -> Path:
        p = Path(stored)
        return p if p.is_absolute() else (self.work_dir / p).resolve()

    def _default_state(self) -> dict[str, Any]:
        return {
            "format": "baseline_train_checkpoint_v1",
            "updated_at": time.time(),
            "model": {
                "completed": False,
                "bundle_path": self._relativize(self.model_bundle_path),
                "metrics": {},
                "duration_sec": 0.0,
            },
            "methods": {},
        }

    def _read_state_unlocked(self) -> dict[str, Any]:
        if not self.state_path.is_file():
            return self._default_state()
        try:
            with self.state_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return self._default_state()

    def read_state(self) -> dict[str, Any]:
        with self.lock_path.open("a+", encoding="utf-8") as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_SH)
            try:
                return self._read_state_unlocked()
            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    def _update_state(self, mutator: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        with self.lock_path.open("a+", encoding="utf-8") as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            try:
                state = self._read_state_unlocked()
                mutator(state)
                state["updated_at"] = time.time()
                atomic_write_json(self.state_path, state)
                return state
            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    # ------------------------------------------------------------------
    # Stage 1: Base Model
    # ------------------------------------------------------------------
    def is_model_done(self) -> bool:
        state = self.read_state()
        model_info = state.get("model", {})
        bundle = self._resolve(model_info.get("bundle_path", self._relativize(self.model_bundle_path)))
        return bool(model_info.get("completed")) and bundle.is_file()

    def mark_model_done(
        self,
        bundle_path: str | Path,
        metrics: dict[str, Any],
        duration_sec: float,
    ) -> None:
        stored_path = self._relativize(bundle_path)

        def _mutate(state: dict[str, Any]) -> None:
            state["model"] = {
                "completed": True,
                "bundle_path": stored_path,
                "metrics": dict(metrics),
                "duration_sec": float(duration_sec),
                "completed_at": time.time(),
            }

        self._update_state(_mutate)

    def invalidate_model(self) -> None:
        def _mutate(state: dict[str, Any]) -> None:
            state["model"]["completed"] = False
            state["methods"] = {}

        self._update_state(_mutate)

    # ------------------------------------------------------------------
    # Stage 2+: Weightmasks (Csordas, Lepori, ...)
    # ------------------------------------------------------------------
    def mask_path(self, method: str, subtask: str) -> Path:
        safe_subtask = subtask.replace("/", "_").replace(":", "_")
        return self.masks_dir / method / f"{safe_subtask}.pt"

    def is_subtask_mask_done(self, method: str, subtask: str) -> bool:
        state = self.read_state()
        m_info = state.get("methods", {}).get(method, {})
        sub_info = m_info.get("subtasks", {}).get(subtask)
        if not sub_info or not sub_info.get("completed"):
            return False
        path = self._resolve(sub_info.get("path", self._relativize(self.mask_path(method, subtask))))
        return path.is_file()

    def is_method_done(self, method: str, subtasks: Sequence[str]) -> bool:
        state = self.read_state()
        m_info = state.get("methods", {}).get(method, {})
        if not m_info:
            return False
        sub_map = m_info.get("subtasks", {})
        for name in subtasks:
            entry = sub_map.get(name)
            if not entry or not entry.get("completed"):
                return False
            path = self._resolve(entry.get("path", self._relativize(self.mask_path(method, name))))
            if not path.is_file():
                return False
        return bool(subtasks)

    def save_subtask_mask(
        self,
        method: str,
        subtask: str,
        mask: SubnetworkMask,
        duration_sec: float,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        out_path = self.mask_path(method, subtask)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mask.save(out_path)

        entry: dict[str, Any] = {
            "completed": True,
            "path": self._relativize(out_path),
            "granularity": mask.granularity,
            "l0": mask.l0(),
            "max_l0": mask.max_l0(),
            "sparsity": mask.sparsity(),
            "duration_sec": float(duration_sec),
            "completed_at": time.time(),
        }
        if extra:
            entry.update(extra)

        def _mutate(state: dict[str, Any]) -> None:
            methods = state.setdefault("methods", {})
            m_entry = methods.setdefault(method, {"completed": False, "subtasks": {}})
            m_entry.setdefault("subtasks", {})[subtask] = entry

        self._update_state(_mutate)
        return out_path

    def mark_method_done(self, method: str, total_duration_sec: float) -> None:
        def _mutate(state: dict[str, Any]) -> None:
            methods = state.setdefault("methods", {})
            m_entry = methods.setdefault(method, {"subtasks": {}})
            m_entry["completed"] = True
            m_entry["total_duration_sec"] = float(total_duration_sec)
            m_entry["completed_at"] = time.time()

        self._update_state(_mutate)

    def invalidate_method(self, method: str) -> None:
        def _mutate(state: dict[str, Any]) -> None:
            state.get("methods", {}).pop(method, None)

        self._update_state(_mutate)

    def load_method_masks(
        self,
        method: str,
        subtasks: Sequence[str] | None = None,
    ) -> dict[str, SubnetworkMask]:
        state = self.read_state()
        sub_map = state.get("methods", {}).get(method, {}).get("subtasks", {})
        names = list(subtasks) if subtasks is not None else sorted(sub_map.keys())
        loaded: dict[str, SubnetworkMask] = {}
        for name in names:
            entry = sub_map.get(name)
            path = (
                self._resolve(entry["path"])
                if entry and "path" in entry
                else self.mask_path(method, name)
            )
            if path.is_file():
                loaded[name] = SubnetworkMask.load(path)
        return loaded

    # ------------------------------------------------------------------
    # Canonical Manifest for Downstream Eval / HPO
    # ------------------------------------------------------------------
    def write_manifest(
        self,
        *,
        config_dict: dict[str, Any],
        subtasks: Sequence[str],
        report_json_path: str | Path | None = None,
        report_csv_path: str | Path | None = None,
        summary_metrics: dict[str, Any] | None = None,
    ) -> Path:
        state = self.read_state()
        model_info = dict(state.get("model", {}))
        if "bundle_path" in model_info:
            model_info["bundle_path"] = self._relativize(self._resolve(model_info["bundle_path"]))

        methods_info: dict[str, Any] = {}
        for m_name, m_entry in state.get("methods", {}).items():
            m_copy = dict(m_entry)
            sub_copy: dict[str, Any] = {}
            for s_name, s_entry in m_entry.get("subtasks", {}).items():
                s_item = dict(s_entry)
                if "path" in s_item:
                    s_item["path"] = self._relativize(self._resolve(s_item["path"]))
                sub_copy[s_name] = s_item
            m_copy["subtasks"] = sub_copy
            methods_info[m_name] = m_copy

        manifest: dict[str, Any] = {
            "format": "baseline_artifact_manifest_v1",
            "created_at": time.time(),
            "work_dir": ".",
            "config": config_dict,
            "subtasks": list(subtasks),
            "model": model_info,
            "methods": methods_info,
            "report_json_path": self._relativize(report_json_path) if report_json_path else None,
            "report_csv_path": self._relativize(report_csv_path) if report_csv_path else None,
            "summary_metrics": dict(summary_metrics or {}),
        }
        atomic_write_json(self.manifest_path, manifest)
        return self.manifest_path


@dataclass
class BaselineArtifact:
    """Loaded baseline bundle (base model + discovered masks) for downstream eval/HPO.

    Relative paths stored in ``manifest`` are resolved against ``root_dir``
    (defaulting to ``work_dir``, the directory containing ``baseline_manifest.json``).
    """

    work_dir: Path
    manifest: dict[str, Any]
    root_dir: Path | None = None

    @property
    def effective_root(self) -> Path:
        return self.root_dir if self.root_dir is not None else self.work_dir

    def resolve_path(self, stored: str | Path) -> Path:
        p = Path(stored)
        if not p.is_absolute():
            return (self.effective_root / p).resolve()
        if p.exists():
            return p
        # Backwards-compatible fallback when a legacy absolute-path manifest is moved
        legacy_root = self.manifest.get("work_dir")
        if legacy_root and legacy_root != ".":
            try:
                rel = p.relative_to(Path(legacy_root))
                return (self.effective_root / rel).resolve()
            except ValueError:
                pass
        return p

    @property
    def config(self) -> dict[str, Any]:
        return dict(self.manifest.get("config", {}))

    @property
    def subtasks(self) -> list[str]:
        return list(self.manifest.get("subtasks", []))

    @property
    def model_bundle_path(self) -> Path:
        return self.resolve_path(self.manifest["model"]["bundle_path"])

    @property
    def report_json_path(self) -> Path | None:
        raw = self.manifest.get("report_json_path")
        return self.resolve_path(raw) if raw else None

    @property
    def report_csv_path(self) -> Path | None:
        raw = self.manifest.get("report_csv_path")
        return self.resolve_path(raw) if raw else None

    def available_methods(self) -> list[str]:
        return sorted(self.manifest.get("methods", {}).keys())

    def load_model(self, device: torch.device | str | None = None) -> tuple[Any, dict[str, Any]]:
        return ModelLoader.load_bundle(self.model_bundle_path, device=device)

    def load_masks(self, method: str) -> dict[str, SubnetworkMask]:
        sub_map = self.manifest.get("methods", {}).get(method, {}).get("subtasks", {})
        loaded: dict[str, SubnetworkMask] = {}
        for subtask, entry in sub_map.items():
            path = self.resolve_path(entry["path"])
            if path.is_file():
                loaded[subtask] = SubnetworkMask.load(path)
        return loaded

    def load_all_masks(self) -> dict[str, dict[str, SubnetworkMask]]:
        return {method: self.load_masks(method) for method in self.available_methods()}


class BaselineArtifactLoader:
    """Entry point for loading saved baseline model + mask artifacts.

    Can be used directly via :meth:`load` (defaulting ``root_dir`` to the
    manifest directory) or instantiated with a shared ``root_dir`` to resolve
    relative artifact directories from a central root.
    """

    DEFAULT_ROOT: Path | None = None

    def __init__(self, root_dir: str | Path | None = None) -> None:
        base = root_dir if root_dir is not None else self.DEFAULT_ROOT
        self.root_dir: Path = Path(base).resolve() if base is not None else Path.cwd().resolve()

    def load_artifact(
        self,
        path: str | Path = ".",
        *,
        artifact_root: str | Path | None = None,
    ) -> BaselineArtifact:
        """Load an artifact relative to ``self.root_dir`` (or an explicit path)."""
        return self.load(path, root_dir=self.root_dir, artifact_root=artifact_root)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        root_dir: str | Path | None = None,
        artifact_root: str | Path | None = None,
    ) -> BaselineArtifact:
        """Load a :class:`BaselineArtifact` from a manifest file or directory.

        Args:
            path: Directory containing ``baseline_manifest.json`` or path to the
                manifest file itself. If relative and not found in the current
                working directory, it is resolved under ``root_dir`` (or
                ``cls.DEFAULT_ROOT``).
            root_dir: Optional base directory used to locate relative ``path``
                arguments. Defaults to ``cls.DEFAULT_ROOT``.
            artifact_root: Optional override root against which relative file
                paths inside the manifest are resolved. Defaults to the
                directory containing ``baseline_manifest.json``.
        """
        raw_path = Path(path)
        base_root = (
            Path(root_dir).resolve()
            if root_dir is not None
            else (cls.DEFAULT_ROOT.resolve() if cls.DEFAULT_ROOT is not None else None)
        )
        if not raw_path.is_absolute() and base_root is not None and not raw_path.exists():
            p = (base_root / raw_path).resolve()
        else:
            p = raw_path.resolve()

        manifest_path = p / CheckpointStore.MANIFEST_FILENAME if p.is_dir() else p
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Baseline manifest not found at {manifest_path}."
            )
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        resolved_artifact_root = (
            Path(artifact_root).resolve() if artifact_root is not None else manifest_path.parent
        )
        return BaselineArtifact(
            work_dir=manifest_path.parent,
            manifest=manifest,
            root_dir=resolved_artifact_root,
        )
