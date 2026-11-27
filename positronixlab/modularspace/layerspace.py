from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from positronixlab.modularspace.config import LayerSpaceConfig
from positronixlab.spacemath.contrast import ContrastResult, get_contrast
from positronixlab.spacemath.losses import (
    loss_batch_repulsion,
    loss_centers_repulsion,
    loss_sigreg,
    loss_sigreg_centers,
)
from positronixlab.spacemath.scoring import activations_to_scores
from positronixlab.spacemath.space_type import SpaceType
from positronixlab.spacemath.spacemath import ActivationType
from positronixlab.spacemath.weighting import (
    get_weighter,
    mask_cluster_membership,
    mask_global_scope,
    stats_update_,
)


class LayerSpace:
    """
    Per-layer decoupled embedding space for computation units (CUs).

    Directly parametrizes and conditionally enables each stage of the update
    (activation scoring, unit weighting, global/cluster contrast, and auxiliary
    losses) from `LayerSpaceConfig` (or any compatible config) plus optional
    explicit keyword overrides.
    """

    def __init__(
        self,
        config: LayerSpaceConfig,
        observed_layers: dict[str, dict[str, Any]],
        device: torch.device | None = None,
        stats_enabled: bool = False,
        *,
        # Space geometry
        space_type: str | None = None,
        # Activation -> scores options
        score_method: str | None = None,
        arch_reducer: str | None = None,
        apply_activation: bool | None = None,
        active_q: float | None = None,
        cnn_locality_weight: float | None = None,
        spatial_radius: int = 3,
        # Weighting options
        weight_method: str | None = None,
        freq_alpha: float | None = None,
        # Contrast options
        contrast_method: str | None = None,
        loss_on: float | None = None,
        loss_off: float | None = None,
        tau: float | None = None,
        # Conditional loss weights (0.0 = disabled / skipped)
        loss_global: float | None = None,
        loss_cluster: float | None = None,
        loss_centers: float | None = None,
        loss_batch: float | None = None,
        loss_sigreg: float | None = None,
        # SIGReg options
        sigreg_method: str | None = None,
        sigreg_target: str | None = None,
        sigreg_dim: int | None = None,
        eps: float = 1e-8,
    ) -> None:
        self.config = config
        self.observed_layers = observed_layers
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stats_enabled = stats_enabled
        self.eps = float(eps)

        # Resolve parameters: explicit kwargs override config fields, falling back to defaults.
        def _pick(override: Any, attr: str, default: Any) -> Any:
            if override is not None:
                return override
            val = getattr(config, attr, default)
            return default if val is None else val

        self.space_type = SpaceType.normalize_name(_pick(space_type, "modular_space_distance", SpaceType.COSINE))

        # Scoring stage
        self.score_method = str(_pick(score_method, "modular_space_scoring", "quantile"))
        self.arch_reducer = arch_reducer if arch_reducer is not None else getattr(config, "modular_space_reducer", None)
        self.apply_activation = bool(_pick(apply_activation, "modular_space_apply_activation", False))
        self.active_q = float(_pick(active_q, "modular_space_active_q", 0.80))
        self.cnn_locality_weight = float(_pick(cnn_locality_weight, "modular_cnn_locality", 0.5))
        self.spatial_radius = int(spatial_radius)

        # Weighting stage
        self.weight_method = str(_pick(weight_method, "modular_space_weighting", "raw"))
        self.freq_alpha = float(_pick(freq_alpha, "modular_space_freq_alpha", 0.0))
        self._weight_fn = get_weighter(self.weight_method)

        # Contrast stage
        self.contrast_method = str(_pick(contrast_method, "modular_space_contrast", "distance_log"))
        self.loss_on = float(_pick(loss_on, "modular_space_loss_on", 1.0))
        self.loss_off = float(_pick(loss_off, "modular_space_loss_off", 1.0))
        self.tau = float(_pick(tau, "modular_space_tau", 0.1))
        self._contrast_fn = get_contrast(self.contrast_method)

        # Conditional loss weights (0.0 disables the term completely)
        self.loss_global = float(_pick(loss_global, "modular_space_loss_global", 1.0))
        self.loss_cluster = float(_pick(loss_cluster, "modular_space_loss_cluster", 1.0))
        self.loss_centers = float(_pick(loss_centers, "modular_space_loss_centers", 0.0))
        self.loss_batch = float(_pick(loss_batch, "modular_space_loss_batch", 0.0))
        self.loss_sigreg = float(_pick(loss_sigreg, "modular_space_loss_sigreg", 1.0))

        # SIGReg stage
        self.sigreg_method = str(_pick(sigreg_method, "modular_space_sigreg_method", "weak"))
        self.sigreg_target = str(_pick(sigreg_target, "modular_space_sigreg_target", "centers")).strip().lower()
        self.sigreg_dim = int(_pick(sigreg_dim, "modular_space_sigreg_dim", 64))
        if self.sigreg_target not in ("centers", "positions"):
            raise ValueError(f"Unknown sigreg_target '{self.sigreg_target}'. Expected 'centers' or 'positions'.")

        self.positions: dict[str, Tensor] = {}
        self.stats: dict[str, Tensor] = {}
        self.optimizers: dict[str, torch.optim.Optimizer] = {}

        self._init_layers()
        self._init_space()

    def _init_layers(self) -> None:
        """Instantiate optional activation modules for each observed layer."""
        for layer_id, layer_info in self.observed_layers.items():
            act_type = layer_info.get("activation_type", None)
            act_fn = None if act_type is None else ActivationType.get_function(act_type)
            self.observed_layers[layer_id]["activation"] = act_fn

    def _init_space(self) -> None:
        """Allocate per-layer CU positions, optional running stats, and Adam optimizers."""
        for layer_id, layer_info in self.observed_layers.items():
            num_cus = int(layer_info["out_units"])

            positions = torch.randn(
                num_cus,
                self.config.modular_space_size,
                device=self.device,
                requires_grad=True,
            )
            torch._dynamo.mark_static_address(positions)
            self.positions[layer_id] = positions

            if self.stats_enabled:
                stats = torch.zeros(num_cus, 4, device=self.device)
                torch._dynamo.mark_static_address(stats)
                self.stats[layer_id] = stats

            self.optimizers[layer_id] = torch.optim.Adam(
                (positions,),
                lr=self.config.modular_space_lr,
            )

    def get_space(self, layer_id: str):
        """Return (layer_info, positions, optimizer, stats) for `layer_id`."""
        layer_info = self.observed_layers.get(layer_id, None)
        positions = self.positions.get(layer_id, None)
        optimizer = self.optimizers.get(layer_id, None)
        stats = self.stats.get(layer_id, None)
        if layer_info is None or positions is None or optimizer is None:
            raise ValueError(f"Layer {layer_id} not found in LayerSpace.")
        return layer_info, positions, optimizer, stats

    def score_activations(self, layer_id: str, activations: Tensor) -> Tensor:
        """Convert raw activations for `layer_id` into detached [-1, 1] unit scores."""
        layer_info = self.get_space(layer_id)[0]
        scores = activations_to_scores(
            activations,
            layer_type=layer_info.get("layer_type", "linear"),
            score_method=self.score_method,
            reducer=self.arch_reducer,
            activation_fn=layer_info.get("activation", None),
            apply_activation=self.apply_activation,
            active_q=self.active_q,
            cnn_locality_weight=self.cnn_locality_weight,
            spatial_radius=self.spatial_radius,
            eps=self.eps,
        )
        return scores.detach().clone()

    def process_activations(
        self,
        layer_id: str,
        activations: Tensor,
        clusters_labels: Tensor | None = None,
        unit_mask: Tensor | None = None,
    ) -> tuple[dict[str, Any], Tensor, Any | None]:
        """Score raw layer activations and run one space optimization step."""
        scores = self.score_activations(layer_id, activations)
        return self.process_scores(layer_id, scores, clusters_labels=clusters_labels, unit_mask=unit_mask)

    def compute_loss(
        self,
        positions: Tensor,
        scores: Tensor,
        *,
        stats: Tensor | None = None,
        clusters_labels: Tensor | None = None,
        unit_mask: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Conditionally compute all enabled contrast and auxiliary loss terms."""
        num_units = positions.shape[0]
        device, dtype = positions.device, positions.dtype
        zero = positions.new_zeros(())

        # 1) Weighting
        w_on, w_off = self._weight_fn(
            scores,
            stats=stats,
            freq_alpha=self.freq_alpha,
        )

        total_loss = zero
        metrics: dict[str, Tensor] = {}

        contrast_kwargs = dict(
            space_type=self.space_type,
            weight_on=self.loss_on,
            weight_off=self.loss_off,
            tau=self.tau,
            eps=self.eps,
        )

        global_res: ContrastResult | None = None
        cluster_res: ContrastResult | None = None

        # 2) Global contrast
        if self.loss_global > 0.0:
            m_global = mask_global_scope(num_units, device, dtype, unit_mask=unit_mask)
            global_res = self._contrast_fn(positions, w_on, w_off, m_global, **contrast_kwargs)
            total_loss = total_loss + self.loss_global * global_res.loss
            metrics["loss_global"] = global_res.loss
            metrics["loss_on"] = global_res.loss_on
            metrics["loss_off"] = global_res.loss_off
            metrics["frac_mixed_global"] = global_res.mixed.mean().detach()

        # 3) Cluster contrast
        if self.loss_cluster > 0.0:
            m_cluster = mask_cluster_membership(
                clusters_labels,
                num_units=num_units,
                device=device,
                dtype=dtype,
                unit_mask=unit_mask,
            )
            if m_cluster is not None:
                cluster_res = self._contrast_fn(positions, w_on, w_off, m_cluster, **contrast_kwargs)
                total_loss = total_loss + self.loss_cluster * cluster_res.loss
                metrics["loss_cluster"] = cluster_res.loss
                metrics["frac_mixed_cluster"] = cluster_res.mixed.mean().detach()
            else:
                metrics["loss_cluster"] = zero

        primary = cluster_res if cluster_res is not None else global_res

        # 4) Center repulsion (on vs off center separation)
        if self.loss_centers > 0.0 and global_res is not None:
            l_centers = loss_centers_repulsion(
                global_res.centers_on,
                global_res.centers_off,
                mixed=global_res.mixed,
                space_type=self.space_type,
                eps=self.eps,
            )
            total_loss = total_loss + self.loss_centers * l_centers
            metrics["loss_centers"] = l_centers

        # 5) Batch repulsion (inter-sample center separation)
        if self.loss_batch > 0.0 and primary is not None:
            l_batch = loss_batch_repulsion(
                primary.centers_on,
                mixed=primary.mixed,
                space_type=self.space_type,
                eps=self.eps,
            )
            total_loss = total_loss + self.loss_batch * l_batch
            metrics["loss_batch"] = l_batch

        # 6) SIGReg regularization
        if self.loss_sigreg > 0.0:
            if self.sigreg_target == "positions":
                pts = positions[unit_mask > 0] if unit_mask is not None else positions
                l_sigreg = loss_sigreg(
                    pts,
                    space_type=self.space_type,
                    method=self.sigreg_method,
                    sketch_dim=self.sigreg_dim,
                    eps=self.eps,
                )
            else:
                if primary is not None:
                    l_sigreg = loss_sigreg_centers(
                        primary.centers_on,
                        primary.mixed,
                        space_type=self.space_type,
                        method=self.sigreg_method,
                        sketch_dim=self.sigreg_dim,
                        eps=self.eps,
                    )
                else:
                    l_sigreg = zero
            total_loss = total_loss + self.loss_sigreg * l_sigreg
            metrics["loss_sigreg"] = l_sigreg

        metrics["loss"] = total_loss
        return total_loss, metrics

    def process_scores(
        self,
        layer_id: str,
        scores: Tensor,
        clusters_labels: Tensor | None = None,
        unit_mask: Tensor | None = None,
    ) -> tuple[dict[str, Any], Tensor, Any | None]:
        """Update stats, compute composite loss, step optimizer, and return (losses, positions, stats)."""
        _, positions, optimizer, stats = self.get_space(layer_id)
        if not self.stats_enabled:
            stats = None

        if stats is not None:
            stats_update_(stats, scores)

        loss, metrics = self.compute_loss(
            positions,
            scores,
            stats=stats,
            clusters_labels=clusters_labels,
            unit_mask=unit_mask,
        )

        if loss.requires_grad:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        losses_dict = {name: val.detach().cpu().numpy() for name, val in metrics.items()}
        stats_copy = None if stats is None else stats.detach().cpu().numpy()
        return losses_dict, positions.detach(), stats_copy
