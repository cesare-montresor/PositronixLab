from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
import os
from typing import Any, Dict, Type, TypeVar
import yaml

TConfig = TypeVar("TConfig", bound="BaseConfig")


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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_yaml(self) -> str:
        return yaml.dump(self.to_dict(), sort_keys=False)

    def with_updates(self: TConfig, **overrides: Any) -> TConfig:
        """Return a new config instance with valid field overrides applied."""
        valid = {f.name for f in fields(self)}
        filtered = {k: v for k, v in overrides.items() if k in valid}
        return replace(self, **filtered)

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            if filepath.endswith(".json"):
                json.dump(self.to_dict(), f, indent=2, sort_keys=True)
            else:
                yaml.dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def load(cls: Type[TConfig], filepath: str) -> TConfig:
        with open(filepath, "r", encoding="utf-8") as f:
            if filepath.endswith(".json"):
                data = json.load(f)
            else:
                data = yaml.safe_load(f)
        return cls.from_dict(data or {})

    @classmethod
    def from_dict(cls: Type[TConfig], data: Dict[str, Any]) -> TConfig:
        valid_fields = {f.name: f.type for f in fields(cls)}
        filtered: Dict[str, Any] = {}
        for k, v in data.items():
            if k not in valid_fields:
                continue
            if v is None:
                filtered[k] = None
                continue
            expected_type = valid_fields[k]
            try:
                if expected_type in (bool, "bool", "Optional[bool]"):
                    if isinstance(v, bool):
                        filtered[k] = v
                    elif isinstance(v, str):
                        filtered[k] = v.strip().lower() in ("true", "1", "yes")
                    else:
                        filtered[k] = bool(v)
                elif expected_type in (int, "int", "Optional[int]"):
                    filtered[k] = int(v)
                elif expected_type in (float, "float", "Optional[float]"):
                    filtered[k] = float(v)
                elif expected_type in (str, "str", "Optional[str]"):
                    filtered[k] = str(v)
                else:
                    filtered[k] = v
            except (ValueError, TypeError):
                filtered[k] = v
        return cls(**filtered)
