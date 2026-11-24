"""Runnable CPU example for LayerSpace + presets (+ optional ClusterTracker).

Demonstrates:
1. Listing and loading a preset into `LayerSpaceConfig`.
2. Instantiating `LayerSpace` for a single synthetic linear layer.
3. Generating activations from co-firing synthetic modules and running
   contrastive optimization steps on CPU.
4. Measuring within-module vs. between-module cosine similarity before and after.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from positronixlab.modularspace import (
    LayerSpace,
    LayerSpaceConfig,
    get_preset,
    list_presets,
)


def make_synthetic_activations(
    batch_size: int,
    groups: torch.Tensor,
    num_modules: int,
    active_modules_per_sample: int = 2,
) -> torch.Tensor:
    """Create synthetic activations where CUs in the same module tend to co-fire."""
    num_units = int(groups.numel())
    active_mods = torch.stack(
        [torch.randperm(num_modules)[:active_modules_per_sample] for _ in range(batch_size)]
    )
    fires = (groups[None, :, None] == active_mods[:, None, :]).any(dim=-1).float()
    return fires * (1.0 + 0.5 * torch.rand(batch_size, num_units)) - (1.0 - fires) * torch.rand(
        batch_size, num_units
    )


def module_similarity_stats(positions: torch.Tensor, groups: torch.Tensor) -> tuple[float, float]:
    """Return (mean_within_module_cos_sim, mean_between_module_cos_sim)."""
    normed = F.normalize(positions.detach(), dim=-1)
    sim = normed @ normed.T
    same = (groups[:, None] == groups[None, :]).float()
    eye = torch.eye(positions.shape[0], device=positions.device)
    within_mask = (same - eye).clamp_min(0.0)
    between_mask = 1.0 - same
    within = float((sim * within_mask).sum() / within_mask.sum().clamp_min(1.0))
    between = float((sim * between_mask).sum() / between_mask.sum().clamp_min(1.0))
    return within, between


def main() -> None:
    torch.manual_seed(42)
    device = torch.device("cpu")

    num_modules = 4
    units_per_module = 8
    num_units = num_modules * units_per_module
    groups = torch.arange(num_units) // units_per_module

    print("Available presets:", list_presets())
    preset_name = "softmax_ce"
    cfg = LayerSpaceConfig.from_dict(
        get_preset(
            preset_name,
            modular_space_size=6,
            modular_space_lr=0.03,
            modular_space_loss_sigreg=0.1,
        )
    )

    observed_layers = {
        "fc_latent": {
            "out_units": num_units,
            "layer_type": "linear",
            "activation_type": "relu",
        }
    }
    space = LayerSpace(cfg, observed_layers, device=device, stats_enabled=True)

    init_within, init_between = module_similarity_stats(space.positions["fc_latent"], groups)
    print(f"Preset: {preset_name}")
    print(f"Initial cosine sim -> within: {init_within:.3f}, between: {init_between:.3f}")

    num_steps = 40
    for step in range(1, num_steps + 1):
        activations = make_synthetic_activations(32, groups, num_modules=num_modules)
        losses, positions, stats = space.process_activations(
            "fc_latent",
            activations,
            clusters_labels=groups,
        )
        if step in (1, 20, num_steps):
            within, between = module_similarity_stats(positions, groups)
            print(
                f"Step {step:2d} | loss={float(losses['loss']):.4f} | "
                f"within={within:.3f}, between={between:.3f}"
            )

    assert stats is not None and stats.shape == (num_units, 4)
    print("Done. Final positions shape:", tuple(positions.shape), "stats shape:", stats.shape)


if __name__ == "__main__":
    main()
