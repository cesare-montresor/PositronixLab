from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Mapping, Optional, Tuple
import torch
from torch import Tensor, nn

from positronixlab.spacemath.contrast import ContrastResult, get_contrast
from positronixlab.spacemath.losses import (
    loss_batch_repulsion,
    loss_centers_repulsion,
    loss_sigreg,
    loss_sigreg_centers,
)
from positronixlab.spacemath.scoring import activations_to_scores
from positronixlab.spacemath.space_type import SpaceType
from positronixlab.spacemath.weighting import (
    get_weighter,
    mask_cluster_membership,
    mask_global_scope,
)


@dataclass
class SpaceStepSpec:
    """
    Flat, Ax-friendly specification of how a single space optimization step
    scores activations, weights units, runs contrast, and combines losses.
    Any loss weight set to 0.0 skips that term completely (clean ablations).
    """

    # Space geometry
    space_type: str = SpaceType.COSINE

    # Activation -> score stage
    score_method: str = "quantile"
    arch_reducer: Optional[str] = None
    apply_activation: bool = False
    active_q: float = 0.80
    cnn_locality_weight: float = 0.5
    spatial_radius: int = 3

    # Weighting stage
    weight_method: str = "raw"
    freq_alpha: float = 0.0

    # Contrast stage
    contrast_method: str = "distance_log"
    loss_on: float = 1.0
    loss_off: float = 1.0
    tau: float = 0.1

    # Scope & auxiliary loss weights (0.0 = disabled)
    loss_global: float = 1.0
    loss_cluster: float = 1.0
    loss_centers: float = 0.0
    loss_batch: float = 0.0
    loss_sigreg: float = 1.0

    # SIGReg options
    sigreg_method: str = "weak"
    sigreg_target: str = "centers"  # "centers" | "positions"
    sigreg_dim: int = 64

    eps: float = 1e-8

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this spec to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SpaceStepSpec":
        """Build from a dict using either short names or `modular_space_*` config keys."""
        valid = {f.name for f in fields(cls)}
        kwargs: Dict[str, Any] = {}
        for k, v in data.items():
            key = _CONFIG_KEY_MAP.get(k, k)
            if key in valid and v is not None:
                kwargs[key] = v
        return cls(**kwargs)

    @classmethod
    def from_config(cls, config: Any) -> "SpaceStepSpec":
        """Extract SpaceStepSpec fields from any PipelineConfig-like object or dict."""
        if isinstance(config, cls):
            return config
        if isinstance(config, Mapping):
            return cls.from_dict(config)
        if hasattr(config, "to_dict"):
            return cls.from_dict(config.to_dict())
        data = {k: getattr(config, k) for k in dir(config) if not k.startswith("_")}
        return cls.from_dict(data)


# Maps flat PipelineConfigV2 field names -> SpaceStepSpec field names
_CONFIG_KEY_MAP: Dict[str, str] = {
    "modular_space_distance": "space_type",
    "modular_space_type": "space_type",
    "modular_space_scoring": "score_method",
    "modular_space_reducer": "arch_reducer",
    "modular_space_apply_activation": "apply_activation",
    "modular_space_active_q": "active_q",
    "modular_cnn_locality": "cnn_locality_weight",
    "modular_space_weighting": "weight_method",
    "modular_space_freq_alpha": "freq_alpha",
    "modular_space_contrast": "contrast_method",
    "modular_space_loss_on": "loss_on",
    "modular_space_loss_off": "loss_off",
    "modular_space_tau": "tau",
    "modular_space_loss_global": "loss_global",
    "modular_space_loss_cluster": "loss_cluster",
    "modular_space_loss_centers": "loss_centers",
    "modular_space_loss_batch": "loss_batch",
    "modular_space_loss_sigreg": "loss_sigreg",
    "modular_space_sigreg_method": "sigreg_method",
    "modular_space_sigreg_target": "sigreg_target",
    "modular_space_sigreg_dim": "sigreg_dim",
}


class SpaceStep:
    """
    Composable, stateless runner that wires together registered `spacemath`
    primitives according to a `SpaceStepSpec`.

    Works on generic unit tensors `positions: (U, D)` and `scores: (bs, U)`,
    so it can be reused across `LayerSpace` (neurons) and recursive `OverSpace` (clusters).
    """

    def __init__(self, spec: Optional[SpaceStepSpec] = None) -> None:
        self.spec = spec if spec is not None else SpaceStepSpec()
        self.space_type = SpaceType.normalize_name(self.spec.space_type)
        self._weight_fn = get_weighter(self.spec.weight_method)
        self._contrast_fn = get_contrast(self.spec.contrast_method)

    def score_activations(
        self,
        activations: Tensor,
        *,
        layer_type: str = "linear",
        activation_fn: Optional[nn.Module] = None,
    ) -> Tensor:
        """Convert raw layer activations into [-1, 1] unit scores using this spec."""
        s = self.spec
        return activations_to_scores(
            activations,
            layer_type=layer_type,
            score_method=s.score_method,
            reducer=s.arch_reducer,
            activation_fn=activation_fn,
            apply_activation=s.apply_activation,
            active_q=s.active_q,
            cnn_locality_weight=s.cnn_locality_weight,
            spatial_radius=s.spatial_radius,
            eps=s.eps,
        )

    def compute_loss(
        self,
        positions: Tensor,
        scores: Tensor,
        *,
        stats: Optional[Tensor] = None,
        clusters_labels: Optional[Tensor] = None,
        unit_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute the composite loss and per-component metrics dict for one batch.

        Args:
            positions: (U, D) unit embeddings.
            scores: (bs, U) unit activation scores in [-1, 1].
            stats: Optional (U, 4) running stats for frequency-based weighting.
            clusters_labels: Optional (U,) integer cluster IDs (-1 = noise).
            unit_mask: Optional (U,) binary mask of active unit slots (for dynamic OverSpace).

        Returns:
            (total_loss, metrics) where `metrics` maps component names to scalar tensors.
        """
        s = self.spec
        num_units = positions.shape[0]
        device, dtype = positions.device, positions.dtype
        zero = positions.new_zeros(())

        w_on, w_off = self._weight_fn(
            scores,
            stats=stats,
            freq_alpha=s.freq_alpha,
        )

        total_loss = zero
        metrics: Dict[str, Tensor] = {}

        contrast_kwargs = dict(
            space_type=self.space_type,
            weight_on=s.loss_on,
            weight_off=s.loss_off,
            tau=s.tau,
            eps=s.eps,
        )

        global_res: Optional[ContrastResult] = None
        cluster_res: Optional[ContrastResult] = None

        # 1) Global contrast
        if s.loss_global > 0.0:
            m_global = mask_global_scope(num_units, device, dtype, unit_mask=unit_mask)
            global_res = self._contrast_fn(positions, w_on, w_off, m_global, **contrast_kwargs)
            total_loss = total_loss + float(s.loss_global) * global_res.loss
            metrics["loss_global"] = global_res.loss
            metrics["loss_on"] = global_res.loss_on
            metrics["loss_off"] = global_res.loss_off
            metrics["frac_mixed_global"] = global_res.mixed.mean().detach()

        # 2) Cluster contrast
        if s.loss_cluster > 0.0:
            m_cluster = mask_cluster_membership(
                clusters_labels,
                num_units=num_units,
                device=device,
                dtype=dtype,
                unit_mask=unit_mask,
            )
            if m_cluster is not None:
                cluster_res = self._contrast_fn(positions, w_on, w_off, m_cluster, **contrast_kwargs)
                total_loss = total_loss + float(s.loss_cluster) * cluster_res.loss
                metrics["loss_cluster"] = cluster_res.loss
                metrics["frac_mixed_cluster"] = cluster_res.mixed.mean().detach()
            else:
                metrics["loss_cluster"] = zero

        # Primary contrast result for center-based auxiliary losses
        primary = cluster_res if cluster_res is not None else global_res

        # 3) Center separation (on vs off center repulsion)
        if s.loss_centers > 0.0 and global_res is not None:
            l_centers = loss_centers_repulsion(
                global_res.centers_on,
                global_res.centers_off,
                mixed=global_res.mixed,
                space_type=self.space_type,
                eps=s.eps,
            )
            total_loss = total_loss + float(s.loss_centers) * l_centers
            metrics["loss_centers"] = l_centers

        # 4) Batch separation (inter-sample center repulsion)
        if s.loss_batch > 0.0 and primary is not None:
            l_batch = loss_batch_repulsion(
                primary.centers_on,
                mixed=primary.mixed,
                space_type=self.space_type,
                eps=s.eps,
            )
            total_loss = total_loss + float(s.loss_batch) * l_batch
            metrics["loss_batch"] = l_batch

        # 5) SIGReg regularization (eager when dynamic masking is needed)
        if s.loss_sigreg > 0.0:
            target = str(s.sigreg_target).strip().lower()
            if target == "positions":
                pts = positions[unit_mask > 0] if unit_mask is not None else positions
                l_sigreg = loss_sigreg(
                    pts,
                    space_type=self.space_type,
                    method=s.sigreg_method,
                    sketch_dim=s.sigreg_dim,
                    eps=s.eps,
                )
            elif target == "centers":
                if primary is not None:
                    l_sigreg = loss_sigreg_centers(
                        primary.centers_on,
                        primary.mixed,
                        space_type=self.space_type,
                        method=s.sigreg_method,
                        sketch_dim=s.sigreg_dim,
                        eps=s.eps,
                    )
                else:
                    l_sigreg = zero
            else:
                raise ValueError(
                    f"Unknown sigreg_target '{s.sigreg_target}'. Expected 'centers' or 'positions'."
                )
            total_loss = total_loss + float(s.loss_sigreg) * l_sigreg
            metrics["loss_sigreg"] = l_sigreg

        metrics["loss"] = total_loss
        return total_loss, metrics
