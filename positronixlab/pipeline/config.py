from __future__ import annotations

import hashlib
import json
import os
import types
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, TypeVar, Union, get_args, get_origin

import yaml

TConfig = TypeVar("TConfig", bound="BaseConfig")


def _yaml_safe(value: Any) -> Any:
    """Convert tuples recursively to lists so ``yaml.safe_dump`` emits plain YAML."""
    if isinstance(value, tuple):
        return [_yaml_safe(v) for v in value]
    if isinstance(value, list):
        return [_yaml_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _yaml_safe(v) for k, v in value.items()}
    return value


def _unwrap_optional_type(expected_type: Any) -> Any:
    """Unwrap ``Optional[T]`` / ``T | None`` for both runtime types and string annotations."""
    if isinstance(expected_type, str):
        norm = expected_type.replace(" ", "")
        if norm.startswith("Optional[") and norm.endswith("]"):
            return norm[len("Optional[") : -1]
        if "|" in norm:
            parts = [p for p in norm.split("|") if p and p != "None"]
            if len(parts) == 1:
                return parts[0]
        return norm

    origin = get_origin(expected_type)
    if origin in (Union, types.UnionType):
        non_none = [arg for arg in get_args(expected_type) if arg is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return expected_type


def _expects_tuple(expected_type: Any) -> bool:
    inner = _unwrap_optional_type(expected_type)
    if inner is tuple or get_origin(inner) is tuple:
        return True
    if isinstance(inner, str):
        return inner in ("tuple", "Tuple") or inner.startswith(("tuple[", "Tuple["))
    return False


@dataclass
class BaseConfig:
    """
    Minimal, framework-independent dataclass configuration for hyperparameters
    with deterministic hashing, immutable updates, and dict/JSON/YAML serialization.
    """

    name: str = "default_config"

    def signature(self) -> str:
        content = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.md5(content.encode("utf-8"), usedforsecurity=False).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(_yaml_safe(self.to_dict()), sort_keys=False)

    def with_updates(self: TConfig, **overrides: Any) -> TConfig:
        """Return a new config instance with valid field overrides applied."""
        valid = {f.name for f in fields(self)}
        filtered = {k: v for k, v in overrides.items() if k in valid}
        return replace(self, **filtered)

    def save(self, filepath: str | Path) -> Path:
        path = Path(filepath)
        os.makedirs(path.parent.resolve(), exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            if path.suffix.lower() == ".json":
                json.dump(self.to_dict(), f, indent=2, sort_keys=True)
            else:
                yaml.safe_dump(_yaml_safe(self.to_dict()), f, sort_keys=False)
        return path

    @classmethod
    def load(cls: type[TConfig], filepath: str | Path) -> TConfig:
        path = Path(filepath)
        with path.open(encoding="utf-8") as f:
            if path.suffix.lower() == ".json":
                data = json.load(f)
            else:
                data = yaml.safe_load(f)
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError(f"Expected mapping in {path}, got {type(data).__name__}")
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls: type[TConfig], data: dict[str, Any]) -> TConfig:
        valid_fields = {f.name: f.type for f in fields(cls)}
        filtered: dict[str, Any] = {}
        for k, v in data.items():
            if k not in valid_fields:
                continue
            if v is None:
                filtered[k] = None
                continue
            expected_type = valid_fields[k]
            inner_type = _unwrap_optional_type(expected_type)
            try:
                if _expects_tuple(expected_type) and isinstance(v, list):
                    filtered[k] = tuple(v)
                elif inner_type in (bool, "bool"):
                    if isinstance(v, bool):
                        filtered[k] = v
                    elif isinstance(v, str):
                        filtered[k] = v.strip().lower() in ("true", "1", "yes")
                    else:
                        filtered[k] = bool(v)
                elif inner_type in (int, "int"):
                    filtered[k] = int(v)
                elif inner_type in (float, "float"):
                    filtered[k] = float(v)
                elif inner_type in (str, "str"):
                    filtered[k] = str(v)
                else:
                    filtered[k] = v
            except (ValueError, TypeError):
                filtered[k] = v
        return cls(**filtered)
