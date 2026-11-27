"""
Runnable example demonstrating composable `BaseConfig` mixins, primitive coercion
via `from_dict()`, immutable `with_updates()`, deterministic `signature()` hashing,
and YAML/JSON round-trip persistence.

Run:
    .venv/bin/python positronixlab/pipeline/examples/example_pipeline_config.py
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

from positronixlab.pipeline import BaseConfig


@dataclass
class DataConfig(BaseConfig):
    dataset_name: str = "fashion_mnist"
    batch_size: int = 32


@dataclass
class TrainConfig(BaseConfig):
    learning_rate: float = 0.001
    train_epochs: int = 10
    use_amp: bool = False


@dataclass
class ExperimentConfig(TrainConfig, DataConfig):
    seed: int | None = None
    device: str | None = None


def main() -> None:
    # 1. Construct from an untyped dictionary (e.g. CLI/YAML/Ax params)
    raw_params = {
        "name": "fmnist_demo",
        "dataset_name": "fashion_mnist",
        "batch_size": "64",
        "learning_rate": "0.002",
        "train_epochs": "20",
        "use_amp": "true",
        "seed": "42",
        "device": "cuda:0",
        "unused_extra_key": "silently_ignored",
    }
    cfg = ExperimentConfig.from_dict(raw_params)
    print("Constructed config:", cfg)
    print("Signature (MD5):   ", cfg.signature())

    # 2. Immutable override (e.g. recording early-stopped effective epochs)
    effective_cfg = cfg.with_updates(train_epochs=12, unknown_key="ignored")
    assert cfg.train_epochs == 20
    assert effective_cfg.train_epochs == 12
    assert cfg.signature() != effective_cfg.signature()
    print("Original epochs:   ", cfg.train_epochs, "| Effective epochs:", effective_cfg.train_epochs)

    # 3. YAML & JSON round-trip persistence
    with tempfile.TemporaryDirectory() as tmp_dir:
        yaml_path = os.path.join(tmp_dir, "config.yaml")
        json_path = os.path.join(tmp_dir, "config.json")

        cfg.save(yaml_path)
        cfg.save(json_path)

        from_yaml = ExperimentConfig.load(yaml_path)
        from_json = ExperimentConfig.load(json_path)

        assert from_yaml == cfg
        assert from_json == cfg
        assert from_yaml.signature() == cfg.signature()
        assert from_json.signature() == cfg.signature()
        print("YAML & JSON round-trip verified successfully.")

    # 4. Optional PipelineOptions demo (requires `parallelazy` in environment)
    try:
        from positronixlab.pipeline import PipelineOptions

        opts = PipelineOptions(timeout_sec=120.0, poll_interval=0.5, verbose=True)
        print("PipelineOptions manager kwargs:", opts.to_manager_kwargs())
    except ImportError as exc:
        print(f"Skipping PipelineOptions demo (parallelazy unavailable: {exc})")


if __name__ == "__main__":
    main()
