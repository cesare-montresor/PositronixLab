"""PositronixLab Atlas subnetwork discoverer implementing :class:`SubnetworkDiscoverer`.

Connects ``LayerSpace`` (contrastive CU embedding) + ``ModularClusterKmeans``
(adaptive functional clustering) + activation-based cluster attribution to the
canonical :class:`SubnetworkMask` contract so PositronixLab's Atlas method can
be benchmarked head-to-head against ``NeuroSurgeonDiscoverer`` and
``CsordasGumbelDiscoverer`` on identical models and datasets.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

import torch
import torch.nn as nn

from positronixlab.baselines.base import get_maskable_layer_names
from positronixlab.modularcluster.modularcluster_kmeans import ModularClusterKmeans
from positronixlab.modularobserver.activation_cache import (
    infer_layer_info,
    reduce_activation_2d as _reduce_activation_to_2d,
)
from positronixlab.modularspace.config import LayerSpaceConfig
from positronixlab.modularspace.layerspace import LayerSpace
from positronixlab.tasks.discoverer import SubnetworkDiscoverer
from positronixlab.tasks.discoverer import unpack_batch as _unpack_batch
from positronixlab.tasks.masks import SubnetworkMask

AttributionMode = Literal["selectivity", "contrastive", "raw"]
EnergyMode = Literal["abs", "relu"]


def _model_device(model: nn.Module) -> torch.device:
    first_param = next(model.parameters(), None)
    return first_param.device if first_param is not None else torch.device("cpu")


def _ensure_reiterable(dataloader: Iterable[Any], name: str = "dataloader") -> None:
    if iter(dataloader) is dataloader:
        raise ValueError(
            f"{name} must be re-iterable (not a one-shot iterator/generator), "
            f"as AtlasDiscoverer iterates it multiple times."
        )


def _infer_out_units(module: nn.Module) -> int | None:
    out_units = infer_layer_info("", module)["out_units"]
    return out_units if out_units > 0 else None


class AtlasDiscoverer(SubnetworkDiscoverer):
    """Unsupervised functional-modularity discoverer powered by PositronixLab Atlas.

    Workflow:
    1. Attaches leak-free forward hooks (``try/finally``) to ``target_layers``.
    2. Projects each layer's computation units (CUs) into a decoupled embedding
       space via :class:`LayerSpace` over ``atlas_epochs`` passes.
    3. Clusters CU embeddings into functional groups via :class:`ModularClusterKmeans`.
    4. Attributes clusters to a target dataloader (``discover``) or to multiple
       classes / subtasks in one shared Atlas pass (``discover_per_class``,
       ``discover_subtasks``), emitting canonical ``SubnetworkMask(granularity="unit")``
       objects.
    """

    name = "atlas"

    def __init__(
        self,
        layer_space_config: LayerSpaceConfig | None = None,
        atlas_epochs: int = 3,
        space_dim: int = 8,
        space_lr: float = 5e-2,
        active_q: float = 0.75,
        contrast_method: str = "softmax_ce",
        cluster_metric: str = "cosine",
        k_min: int | None = None,
        k_max: int | None = None,
        top_cluster_fraction: float = 0.5,
        min_active_clusters: int = 1,
        attribution_mode: AttributionMode = "selectivity",
        energy_mode: EnergyMode = "abs",
    ) -> None:
        if attribution_mode not in ("selectivity", "contrastive", "raw"):
            raise ValueError(f"Unsupported attribution_mode: {attribution_mode!r}")
        if energy_mode not in ("abs", "relu"):
            raise ValueError(f"Unsupported energy_mode: {energy_mode!r}")
        self.layer_space_config = layer_space_config
        self.atlas_epochs = atlas_epochs
        self.space_dim = space_dim
        self.space_lr = space_lr
        self.active_q = active_q
        self.contrast_method = contrast_method
        self.cluster_metric = cluster_metric
        self.k_min = k_min
        self.k_max = k_max
        self.top_cluster_fraction = top_cluster_fraction
        self.min_active_clusters = min_active_clusters
        self.attribution_mode: AttributionMode = attribution_mode
        self.energy_mode: EnergyMode = energy_mode

        # Populated after fit_atlas() for inspection / reuse
        self.last_positions: dict[str, torch.Tensor] = {}
        self.last_clusters: dict[str, torch.Tensor] = {}

    def _build_config(self) -> LayerSpaceConfig:
        if self.layer_space_config is not None:
            return self.layer_space_config
        return LayerSpaceConfig(
            modular_space_size=self.space_dim,
            modular_space_lr=self.space_lr,
            modular_space_active_q=self.active_q,
            modular_space_contrast=self.contrast_method,
            modular_space_loss_global=1.0,
            modular_space_loss_cluster=0.0,
            modular_space_loss_centers=0.0,
            modular_space_loss_batch=0.0,
            modular_space_loss_sigreg=0.0,
        )

    def fit_atlas(
        self,
        model: nn.Module,
        dataloader: Iterable[Any],
        target_layers: list[str] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Fit per-layer ``LayerSpace`` embeddings and ``ModularClusterKmeans`` clusters.

        Returns:
            ``(positions_by_layer, cluster_labels_by_layer)`` where every cluster
            label tensor has ``dtype=torch.long`` and shape ``(out_units,)``.
        """
        layers = (
            list(target_layers)
            if target_layers is not None
            else get_maskable_layer_names(model)
        )
        modules = dict(model.named_modules())

        observed_layers: dict[str, dict[str, Any]] = {}
        valid_layers: list[str] = []
        for layer_name in layers:
            mod = modules.get(layer_name)
            if mod is None:
                continue
            out_units = _infer_out_units(mod)
            if out_units is None or out_units < 2:
                continue
            valid_layers.append(layer_name)
            observed_layers[layer_name] = {
                "out_units": out_units,
                "activation": "linear",
                "layer_type": "conv2d" if isinstance(mod, nn.Conv2d) else "linear",
            }

        if not valid_layers:
            raise ValueError("No observable target_layers with >= 2 units found.")

        if self.atlas_epochs > 1:
            _ensure_reiterable(dataloader, "dataloader")

        device = _model_device(model)
        cfg = self._build_config()
        layer_space = LayerSpace(
            config=cfg,
            observed_layers=observed_layers,
            device=device,
            stats_enabled=False,
        )

        # Register temporary forward hooks with guaranteed cleanup
        captured: dict[str, torch.Tensor] = {}
        handles = []
        for layer_name in valid_layers:
            mod = modules[layer_name]

            def _make_hook(name: str):
                def _hook(mod_ref: nn.Module, _inp: Any, output: Any) -> None:
                    out_t = output[0] if isinstance(output, (tuple, list)) else output
                    if isinstance(out_t, torch.Tensor):
                        captured[name] = _reduce_activation_to_2d(out_t.detach(), module=mod_ref)

                return _hook

            handles.append(mod.register_forward_hook(_make_hook(layer_name)))

        was_training = model.training
        model.eval()
        try:
            for _ in range(max(1, self.atlas_epochs)):
                for batch in dataloader:
                    x, _ = _unpack_batch(batch)
                    x = x.to(device)
                    captured.clear()
                    with torch.no_grad():
                        model(x)
                    for layer_name, act in captured.items():
                        if act.shape[0] < 2:
                            continue
                        layer_space.process_activations(layer_name, act)
        finally:
            for h in handles:
                h.remove()
            model.train(was_training)

        # Cluster each layer's learned CU positions on CPU
        positions_by_layer: dict[str, torch.Tensor] = {}
        clusters_by_layer: dict[str, torch.Tensor] = {}

        for layer_name in valid_layers:
            pos = layer_space.positions[layer_name].detach().cpu()
            positions_by_layer[layer_name] = pos
            n_units = pos.shape[0]
            if n_units < 4:
                clusters_by_layer[layer_name] = torch.zeros(n_units, dtype=torch.long)
                continue

            k_min = self.k_min if self.k_min is not None else 2
            k_max = self.k_max if self.k_max is not None else min(max(2, n_units // 2), 8)
            k_max = min(k_max, n_units - 1)
            k_min = min(k_min, k_max)
            if k_min < 2 or k_max < 2:
                clusters_by_layer[layer_name] = torch.zeros(n_units, dtype=torch.long)
                continue

            kmeans = ModularClusterKmeans(
                metric=self.cluster_metric,  # type: ignore[arg-type]
                k_min=k_min,
                k_max=k_max,
                device=torch.device("cpu"),
                verbose=False,
            )
            kmeans.fit(pos)
            if kmeans.labels is None:
                clusters_by_layer[layer_name] = torch.zeros(n_units, dtype=torch.long)
            else:
                clusters_by_layer[layer_name] = kmeans.labels.detach().cpu().long()

        self.last_positions = positions_by_layer
        self.last_clusters = clusters_by_layer
        return positions_by_layer, clusters_by_layer

    def _collect_mean_unit_energy(
        self,
        model: nn.Module,
        dataloader: Iterable[Any],
        target_layers: list[str],
    ) -> dict[str, torch.Tensor]:
        """Compute per-unit mean absolute activation energy across *dataloader*."""
        modules = dict(model.named_modules())
        device = _model_device(model)
        sums: dict[str, torch.Tensor] = {}
        counts: dict[str, int] = {k: 0 for k in target_layers}

        handles = []
        for layer_name in target_layers:
            mod = modules[layer_name]

            def _make_hook(name: str):
                def _hook(mod_ref: nn.Module, _inp: Any, output: Any) -> None:
                    out_t = output[0] if isinstance(output, (tuple, list)) else output
                    if not isinstance(out_t, torch.Tensor):
                        return
                    act2d = _reduce_activation_to_2d(out_t.detach(), module=mod_ref)
                    if self.energy_mode == "relu":
                        batch_sum = act2d.clamp_min(0.0).sum(dim=0)
                    else:
                        batch_sum = act2d.abs().sum(dim=0)
                    if name not in sums:
                        sums[name] = batch_sum
                    else:
                        sums[name] = sums[name] + batch_sum
                    counts[name] += int(act2d.shape[0])

                return _hook

            handles.append(mod.register_forward_hook(_make_hook(layer_name)))

        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for batch in dataloader:
                    x, _ = _unpack_batch(batch)
                    model(x.to(device))
        finally:
            for h in handles:
                h.remove()
            model.train(was_training)

        result = {
            k: (sums[k] / counts[k]).cpu() for k in target_layers if k in sums and counts[k] > 0
        }
        if not result:
            raise ValueError(
                "Dataloader yielded no samples for any target layer "
                "(empty or exhausted iterator)."
            )
        return result

    def _clusters_to_unit_mask(
        self,
        clusters_by_layer: dict[str, torch.Tensor],
        unit_energy: dict[str, torch.Tensor],
        mask_name: str,
        extra_metadata: dict[str, Any] | None = None,
        top_cluster_fraction: float | None = None,
    ) -> SubnetworkMask:
        """Select top-attributed functional clusters per layer and build a ``SubnetworkMask``."""
        frac = self.top_cluster_fraction if top_cluster_fraction is None else float(top_cluster_fraction)
        masks: dict[str, torch.Tensor] = {}
        scores: dict[str, torch.Tensor] = {}
        selected_clusters: dict[str, list[int]] = {}

        for layer_name, labels in clusters_by_layer.items():
            energy = unit_energy.get(layer_name)
            if energy is None:
                continue
            labels_long = labels.detach().cpu().long()
            energy_cpu = energy.detach().cpu()
            unique_c = torch.unique(labels_long)
            cluster_scores = []
            for cid in unique_c:
                c_mask = labels_long == cid
                c_score = float(energy_cpu[c_mask].mean().item())
                cluster_scores.append((int(cid.item()), c_score))

            cluster_scores.sort(key=lambda item: item[1], reverse=True)
            k_select = max(
                self.min_active_clusters,
                int(round(len(cluster_scores) * frac)),
            )
            k_select = min(k_select, len(cluster_scores))
            chosen = {cid for cid, _ in cluster_scores[:k_select]}
            selected_clusters[layer_name] = sorted(chosen)

            unit_mask = torch.zeros_like(labels_long, dtype=torch.bool)
            for cid in chosen:
                unit_mask |= labels_long == cid
            masks[layer_name] = unit_mask
            scores[layer_name] = energy_cpu

        if not masks:
            raise ValueError(
                "No target layers matched unit_energy; cannot construct an empty SubnetworkMask."
            )

        meta = {
            "method": self.name,
            "top_cluster_fraction": frac,
            "selected_clusters": selected_clusters,
            "num_clusters": {
                k: int(torch.unique(v).numel()) for k, v in clusters_by_layer.items()
            },
        }
        if extra_metadata:
            meta.update(extra_metadata)

        return SubnetworkMask(
            name=mask_name,
            granularity="unit",
            masks=masks,
            scores=scores,
            metadata=meta,
        )

    def _compute_attribution_scores(
        self,
        energy_by_subtask: dict[str, dict[str, torch.Tensor]],
        reference_energy: dict[str, torch.Tensor] | None = None,
        eps: float = 1e-8,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Transform raw per-subtask unit energies according to ``self.attribution_mode``."""
        if self.attribution_mode == "raw":
            return energy_by_subtask

        subtasks = list(energy_by_subtask.keys())
        if len(subtasks) <= 1 and reference_energy is None:
            return energy_by_subtask

        scored: dict[str, dict[str, torch.Tensor]] = {}
        if self.attribution_mode == "contrastive" and len(subtasks) > 1:
            for name, e_sub in energy_by_subtask.items():
                scored[name] = {}
                for layer, val in e_sub.items():
                    others = [
                        energy_by_subtask[o][layer]
                        for o in subtasks
                        if o != name and layer in energy_by_subtask[o]
                    ]
                    if others:
                        ref = torch.stack(others, dim=0).mean(dim=0)
                        scored[name][layer] = (val - ref) / (ref.abs() + eps)
                    else:
                        scored[name][layer] = val
            return scored

        # Default: "selectivity" relative to reference_energy (or mean across subtasks)
        for name, e_sub in energy_by_subtask.items():
            scored[name] = {}
            for layer, val in e_sub.items():
                if reference_energy is not None and layer in reference_energy:
                    ref = reference_energy[layer]
                    scored[name][layer] = (val - ref) / (ref.abs() + eps)
                elif len(subtasks) > 1:
                    all_vals = [
                        energy_by_subtask[o][layer]
                        for o in subtasks
                        if layer in energy_by_subtask[o]
                    ]
                    ref = torch.stack(all_vals, dim=0).mean(dim=0)
                    scored[name][layer] = (val - ref) / (ref.abs() + eps)
                else:
                    scored[name][layer] = val
        return scored

    def attribute_from_unit_energy(
        self,
        clusters_by_layer: dict[str, torch.Tensor],
        energy_by_subtask: dict[str, dict[str, torch.Tensor]],
        *,
        reference_energy: dict[str, torch.Tensor] | None = None,
        top_cluster_fraction: float | None = None,
    ) -> dict[str, SubnetworkMask]:
        """Build per-subtask ``SubnetworkMask``s from pre-computed clusters and unit energies (zero model passes)."""
        scored_by_subtask = self._compute_attribution_scores(
            energy_by_subtask, reference_energy=reference_energy
        )
        results: dict[str, SubnetworkMask] = {}
        for subtask_name, energy in scored_by_subtask.items():
            results[subtask_name] = self._clusters_to_unit_mask(
                clusters_by_layer,
                energy,
                mask_name=f"atlas:{subtask_name}",
                extra_metadata={
                    "subtask": subtask_name,
                    "attribution_mode": self.attribution_mode,
                    "energy_mode": self.energy_mode,
                },
                top_cluster_fraction=top_cluster_fraction,
            )

        # Record pairwise IoU across subtasks to expose any mask collapse immediately
        if len(results) > 1:
            names = list(results.keys())
            for name in names:
                pairwise: dict[str, dict[str, Any]] = {}
                for other in names:
                    if other == name:
                        continue
                    pairwise[other] = {
                        "overall": results[name].iou(results[other]),
                        "per_layer": results[name].per_layer_iou(results[other]),
                    }
                results[name].metadata["pairwise_iou"] = pairwise

        return results

    def attribute_from_clusters(
        self,
        model: nn.Module,
        clusters_by_layer: dict[str, torch.Tensor],
        subtask_dataloaders: dict[str, Iterable[Any]],
        *,
        reference_dataloader: Iterable[Any] | None = None,
        top_cluster_fraction: float | None = None,
    ) -> dict[str, SubnetworkMask]:
        """Attribute pre-computed clusters (e.g. from ``ClusterPlugin``) to subtasks without re-fitting the Atlas."""
        layers = list(clusters_by_layer.keys())
        ref_energy = (
            self._collect_mean_unit_energy(model, reference_dataloader, layers)
            if reference_dataloader is not None
            else None
        )
        energy_by_subtask = {
            subtask_name: self._collect_mean_unit_energy(model, loader, layers)
            for subtask_name, loader in subtask_dataloaders.items()
        }
        return self.attribute_from_unit_energy(
            clusters_by_layer,
            energy_by_subtask,
            reference_energy=ref_energy,
            top_cluster_fraction=top_cluster_fraction,
        )

    def sweep_fractions(
        self,
        clusters_by_layer: dict[str, torch.Tensor],
        unit_energy: dict[str, torch.Tensor],
        fractions: Iterable[float],
        mask_name: str = "atlas",
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[SubnetworkMask]:
        """Generate a sparsity sweep of ``SubnetworkMask``s from one attribution pass (zero retraining)."""
        return [
            self._clusters_to_unit_mask(
                clusters_by_layer,
                unit_energy,
                mask_name=f"{mask_name}@{float(f):.2f}",
                extra_metadata=extra_metadata,
                top_cluster_fraction=float(f),
            )
            for f in fractions
        ]

    def discover(
        self,
        model: nn.Module,
        dataloader: Iterable[Any],
        target_layers: list[str] | None = None,
        mask_name: str = "atlas",
        **kwargs: Any,
    ) -> SubnetworkMask:
        """Fit Atlas on *dataloader* and return the top-attributed unit subnetwork."""
        _ensure_reiterable(dataloader, "dataloader")
        _, clusters_by_layer = self.fit_atlas(model, dataloader, target_layers=target_layers)
        layers = list(clusters_by_layer.keys())
        energy = self._collect_mean_unit_energy(model, dataloader, layers)
        return self._clusters_to_unit_mask(clusters_by_layer, energy, mask_name=mask_name)

    def discover_subtasks(
        self,
        model: nn.Module,
        reference_dataloader: Iterable[Any],
        subtask_dataloaders: dict[str, Iterable[Any]],
        target_layers: list[str] | None = None,
    ) -> dict[str, SubnetworkMask]:
        """Fit ONE shared Atlas on *reference_dataloader* and attribute all subtasks at once."""
        if self.attribution_mode == "selectivity":
            _ensure_reiterable(reference_dataloader, "reference_dataloader")
        _, clusters_by_layer = self.fit_atlas(
            model, reference_dataloader, target_layers=target_layers
        )
        layers = list(clusters_by_layer.keys())
        ref_energy = (
            self._collect_mean_unit_energy(model, reference_dataloader, layers)
            if self.attribution_mode == "selectivity"
            else None
        )
        energy_by_subtask = {
            subtask_name: self._collect_mean_unit_energy(model, loader, layers)
            for subtask_name, loader in subtask_dataloaders.items()
        }
        return self.attribute_from_unit_energy(
            clusters_by_layer,
            energy_by_subtask,
            reference_energy=ref_energy,
        )
