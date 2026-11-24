"""
Minimal runnable example of `positronixlab.nas` (`TrialConfig`, `TemplateBase`, `Experiment`).

Runs a fast, self-contained multi-objective NAS dry-run in a temporary directory
using a lightweight mock `eval_fn` (no GPU or dataset download required).

Run:
    .venv/bin/python positronixlab/nas/examples/example_nas_trial.py
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import tempfile

from positronixlab.nas import Experiment, TrialConfig, TrialResult
from positronixlab.template import TemplateBase


@dataclass
class DemoTrialConfig(TrialConfig):
    """Flat Ax-aware config combining search parameters and fixed constants."""

    model_type: str = "fc"
    cnn_kernel_size: int = 3
    hidden_dim: int = 64
    learning_rate: float = 1e-3
    train_epochs: int = 50


def build_template() -> TemplateBase:
    """Define a small hierarchical search space with fixed constants."""
    tpl = TemplateBase("demo_nas")
    tpl.add_choice(
        "model_type",
        ["fc", "cnn"],
        "str",
        is_ordered=False,
        dependent_parameters={"cnn": ["cnn_kernel_size"]},
    )
    tpl.add_choice("cnn_kernel_size", [3, 5], "int", is_ordered=True)
    tpl.add_range("hidden_dim", [32, 256], "int")
    tpl.add_range("learning_rate", [1e-4, 1e-2], "float", scaling="log")
    tpl.add_fixed("train_epochs", 50)
    return tpl


def mock_eval(cfg: DemoTrialConfig) -> TrialResult:
    """Deterministic synthetic objective simulating accuracy vs. model size."""
    arch_bonus = 0.04 if cfg.model_type == "cnn" else 0.0
    kernel_bonus = 0.01 if (cfg.model_type == "cnn" and cfg.cnn_kernel_size == 5) else 0.0
    lr_penalty = abs(cfg.learning_rate - 2e-3) * 10.0
    val_acc = min(0.99, 0.72 + arch_bonus + kernel_bonus + 0.0005 * cfg.hidden_dim - lr_penalty)
    model_size = float(cfg.hidden_dim * (cfg.cnn_kernel_size if cfg.model_type == "cnn" else 1))
    best_epoch = max(10, min(cfg.train_epochs, 20 + cfg.hidden_dim // 16))

    return TrialResult(
        status="completed",
        metrics={"val_acc": val_acc, "model_size": model_size},
        metadata={"best_epoch": best_epoch},
        effective_overrides={"train_epochs": best_epoch},
    )


def main() -> None:
    template = build_template()

    with tempfile.TemporaryDirectory(prefix="positronixlab_nas_demo_") as tmp_dir:
        exp = Experiment(
            name="demo_moo",
            template=template,
            objective="val_acc, -model_size",
            outcome_constraints=["val_acc >= 0.70", "model_size <= 2000.0"],
            eval_fn=mock_eval,
            config_cls=DemoTrialConfig,
            output_dir=tmp_dir,
            initialization_budget=4,
            parallel_init=2,
            parallel_bo=1,
            random_seed=42,
        )

        frontier = exp.run(n_trials=6)

        print(f"Experiment dir: {exp.exp_dir}")
        print(f"Completed trials: {len(exp.client._experiment.trials)}")
        print(f"Pareto frontier size: {len(frontier)}")
        for cfg, pred, t_idx in frontier:
            print(
                f"  trial_{t_idx:03d}: model={cfg.model_type}, hidden_dim={cfg.hidden_dim}, "
                f"lr={cfg.learning_rate:.5f} -> {pred}"
            )

        best_summary_path = os.path.join(exp.best_dir, "best_summary.json")
        with open(best_summary_path, encoding="utf-8") as f:
            summary = json.load(f)
        print(f"Exported best_summary.json with {len(summary['best_trials'])} entries.")
        print(f"summary.csv exists: {os.path.exists(os.path.join(exp.exp_dir, 'summary.csv'))}")


if __name__ == "__main__":
    main()
