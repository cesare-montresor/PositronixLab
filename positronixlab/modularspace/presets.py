from __future__ import annotations

from typing import Any, Dict, List

from positronixlab.modularspace.space_step import SpaceStepSpec


PRESETS: Dict[str, Dict[str, Any]] = {
    # Reproduces ModularSpace v1: global distance_log + centers repulsion
    "v1_centers": {
        "modular_space_distance": "cosine",
        "modular_space_scoring": "quantile",
        "modular_space_weighting": "raw",
        "modular_space_contrast": "distance_log",
        "modular_space_loss_global": 1.0,
        "modular_space_loss_cluster": 0.0,
        "modular_space_loss_centers": 1.0,
        "modular_space_loss_batch": 0.0,
        "modular_space_loss_sigreg": 0.0,
    },
    # Reproduces layerspace_remixed.py: vectorized distance_log (global + cluster) + SIGReg on active centers
    "distance_sigreg_centers": {
        "modular_space_distance": "cosine",
        "modular_space_scoring": "quantile",
        "modular_space_weighting": "raw",
        "modular_space_contrast": "distance_log",
        "modular_space_loss_global": 1.0,
        "modular_space_loss_cluster": 1.0,
        "modular_space_loss_centers": 0.0,
        "modular_space_loss_batch": 0.0,
        "modular_space_loss_sigreg": 1.0,
        "modular_space_sigreg_method": "weak",
        "modular_space_sigreg_target": "centers",
    },
    # Reproduces layerspace_process_scores_simple.py: distance_log (global + cluster) + SIGReg on positions
    "distance_sigreg_positions": {
        "modular_space_distance": "cosine",
        "modular_space_scoring": "quantile",
        "modular_space_weighting": "raw",
        "modular_space_contrast": "distance_log",
        "modular_space_loss_global": 1.0,
        "modular_space_loss_cluster": 1.0,
        "modular_space_loss_centers": 0.0,
        "modular_space_loss_batch": 0.0,
        "modular_space_loss_sigreg": 1.0,
        "modular_space_sigreg_method": "weak",
        "modular_space_sigreg_target": "positions",
    },
    # Reproduces layerspace_process_scores.py: softmax CE + inverse-frequency weighting + SIGReg on positions
    "softmax_ce": {
        "modular_space_distance": "cosine",
        "modular_space_scoring": "quantile",
        "modular_space_weighting": "inv_freq",
        "modular_space_freq_alpha": 0.5,
        "modular_space_contrast": "softmax_ce",
        "modular_space_tau": 0.1,
        "modular_space_loss_global": 1.0,
        "modular_space_loss_cluster": 1.0,
        "modular_space_loss_centers": 0.0,
        "modular_space_loss_batch": 0.0,
        "modular_space_loss_sigreg": 1.0,
        "modular_space_sigreg_method": "weak",
        "modular_space_sigreg_target": "positions",
    },
}


def list_presets() -> List[str]:
    """Return sorted names of all registered space loss presets."""
    return sorted(PRESETS.keys())


def get_preset(name: str, **overrides: Any) -> Dict[str, Any]:
    """Return a copy of a preset config dict with optional overrides (ideal for ablations)."""
    key = str(name).strip().lower()
    if key not in PRESETS:
        available = ", ".join(list_presets())
        raise KeyError(f"Unknown preset '{name}'. Available: [{available}]")
    cfg = dict(PRESETS[key])
    cfg.update(overrides)
    return cfg


def get_preset_spec(name: str, **overrides: Any) -> SpaceStepSpec:
    """Return a `SpaceStepSpec` instantiated from a preset + optional overrides."""
    return SpaceStepSpec.from_dict(get_preset(name, **overrides))
