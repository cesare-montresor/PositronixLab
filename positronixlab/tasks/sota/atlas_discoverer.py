"""PositronixLab Atlas subnetwork discoverer implementing :class:`SubnetworkDiscoverer`.

Connects ``LayerSpace`` (contrastive CU embedding) + ``ModularClusterKmeans``
(adaptive functional clustering) + activation-based cluster attribution to the
canonical :class:`SubnetworkMask` contract so PositronixLab's Atlas method can
be benchmarked head-to-head against ``NeuroSurgeonDiscoverer`` and
``CsordasGumbelDiscoverer`` on identical models and datasets.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from positronixlab.metamodel.sota.base import get_maskable_layer_names
from positronixlab.modularcluster.modularcluster_kmeans import ModularClusterKmeans
from positronixlab.modularspace.config import LayerSpaceConfig
from positronixlab.modularspace.layerspace import LayerSpace
from positronixlab.tasks.sota.discoverers import SubnetworkDiscoverer, _unpack_batch
from positronixlab.tasks.sota.masks import SubnetworkMask


def _reduce_activation_to_2d(act: torch.Tensor) -> torch.Tensor:
    """Reduce arbitrary layer output tensor to ``(B, U)`` unit activations.

    Handles:
    - 2D Linear: ``(B, U)`` -> unchanged
    - 3D Sequence (Transformer / RNN / ViT): ``(B, T, U)`` -> mean over ``T``
    - 4D Conv2d: ``(B, C, H, W)`` -> spatial mean over ``(H, W)``
    - 3D Conv1d: ``(B, C, L)`` -> handled via mean over last dim if needed
    """
    if act.ndim == 2:
        return act
    if act.ndim == 4:
        return act.mean(dim=(2, 3))
    if act.ndim == 3:
        return act.mean(dim=1)
    return act.reshape(act.shape[0], -1)


def _infer_out_units(module: nn.Module) -> Optional[int]:
    if isinstance(module, nn.Linear):
        return int(module.out_features)
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return int(module.out_channels)
    if type(module).__name__ == "Conv1D" and hasattr(module, "nf"):
        return int(module.nf)
    w = getattr(module, "weight", None)
    if isinstance(w, torch.Tensor) and w.ndim >= 1:
        return int(w.shape[0])
    return None


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
        layer_space_config: Optional[LayerSpaceConfig] = None,
        atlas_epochs: int = 3,
        space_dim: int = 8,
        space_lr: float = 5e-2,
        active_q: float = 0.75,
        contrast_method: str = "softmax_ce",
        cluster_metric: str = "cosine",
        k_min: Optional[int] = None,
        k_max: Optional[int] = None,
        top_cluster_fraction: float = 0.5,
        min_active_clusters: int = 1,
    ) -> None:
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

        # Populated after fit_atlas() for inspection / reuse
        self.last_positions: Dict[str, torch.Tensor] = {}
        self.last_clusters: Dict[str, torch.Tensor] = {}

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
        target_layers: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
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

        observed_layers: Dict[str, Dict[str, Any]] = {}
        valid_layers: List[str] = []
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

        device = next(model.parameters()).device
        cfg = self._build_config()
        layer_space = LayerSpace(
            config=cfg,
            observed_layers=observed_layers,
            device=device,
            stats_enabled=False,
        )

        # Register temporary forward hooks with guaranteed cleanup
        captured: Dict[str, torch.Tensor] = {}
        handles = []
        for layer_name in valid_layers:
            mod = modules[layer_name]

            def _make_hook(name: str):
                def _hook(_mod: nn.Module, _inp: Any, output: Any) -> None:
                    out_t = output[0] if isinstance(output, (tuple, list)) else output
                    if isinstance(out_t, torch.Tensor):
                        captured[name] = _reduce_activation_to_2d(out_t.detach())

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
        positions_by_layer: Dict[str, torch.Tensor] = {}
        clusters_by_layer: Dict[str, torch.Tensor] = {}

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
                # Cast to long explicitly (guards against ModularClusterKmeans float32 label bug)
                clusters_by_layer[layer_name] = kmeans.labels.detach().cpu().long()

        self.last_positions = positions_by_layer
        self.last_clusters = clusters_by_layer
        return positions_by_layer, clusters_by_layer

    def _collect_mean_unit_energy(
        self,
        model: nn.Module,
        dataloader: Iterable[Any],
        target_layers: List[str],
    ) -> Dict[str, torch.Tensor]:
        """Compute per-unit mean absolute activation energy across *dataloader*."""
        modules = dict(model.named_modules())
        device = next(model.parameters()).device
        sums: Dict[str, torch.Tensor] = {}
        counts: Dict[str, int] = {k: 0 for k in target_layers}

        handles = []
        for layer_name in target_layers:
            mod = modules[layer_name]

            def _make_hook(name: str):
                def _hook(_mod: nn.Module, _inp: Any, output: Any) -> None:
                    out_t = output[0] if isinstance(output, (tuple, list)) else output
                    if not isinstance(out_t, torch.Tensor):
                        return
                    act2d = _reduce_activation_to_2d(out_t.detach()).abs().cpu()
                    batch_sum = act2d.sum(dim=0)
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

        return {
            k: (sums[k] / max(1, counts[k])) for k in target_layers if k in sums
        }

    def _clusters_to_unit_mask(
        self,
        clusters_by_layer: Dict[str, torch.Tensor],
        unit_energy: Dict[str, torch.Tensor],
        mask_name: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> SubnetworkMask:
        """Select top-attributed functional clusters per layer and build a ``SubnetworkMask``."""
        masks: Dict[str, torch.Tensor] = {}
        scores: Dict[str, torch.Tensor] = {}
        selected_clusters: Dict[str, List[int]] = {}

        for layer_name, labels in clusters_by_layer.items():
            energy = unit_energy.get(layer_name)
            if energy is None:
                continue
            unique_c = torch.unique(labels)
            cluster_scores = []
            for cid in unique_c:
                c_mask = labels == cid
                c_score = float(energy[c_mask].mean().item())
                cluster_scores.append((int(cid.item()), c_score))

            cluster_scores.sort(key=lambda item: item[1], reverse=True)
            k_select = max(
                self.min_active_clusters,
                int(round(len(cluster_scores) * self.top_cluster_fraction)),
            )
            k_select = min(k_select, len(cluster_scores))
            chosen = {cid for cid, _ in cluster_scores[:k_select]}
            selected_clusters[layer_name] = sorted(chosen)

            unit_mask = torch.zeros_like(labels, dtype=torch.bool)
            for cid in chosen:
                unit_mask |= labels == cid
            masks[layer_name] = unit_mask
            scores[layer_name] = energy

        meta = {
            "method": self.name,
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

    def discover(
        self,
        model: nn.Module,
        dataloader: Iterable[Any],
        target_layers: Optional[List[str]] = None,
        mask_name: str = "atlas",
        **kwargs: Any,
    ) -> SubnetworkMask:
        """Fit Atlas on *dataloader* and return the top-attributed unit subnetwork."""
        _, clusters_by_layer = self.fit_atlas(model, dataloader, target_layers=target_layers)
        layers = list(clusters_by_layer.keys())
        energy = self._collect_mean_unit_energy(model, dataloader, layers)
        return self._clusters_to_unit_mask(clusters_by_layer, energy, mask_name=mask_name)

    def discover_subtasks(
        self,
        model: nn.Module,
        reference_dataloader: Iterable[Any],
        subtask_dataloaders: Dict[str, Iterable[Any]],
        target_layers: Optional[List[str]] = None,
    ) -> Dict[str, SubnetworkMask]:
        """Fit ONE shared Atlas on *reference_dataloader* and attribute all subtasks at once."""
        _, clusters_by_layer = self.fit_atlas(
            model, reference_dataloader, target_layers=target_layers
        )
        layers = list(clusters_by_layer.keys())
        results: Dict[str, SubnetworkMask] = {}
        for subtask_name, loader in subtask_dataloaders.items():
            energy = self._collect_mean_unit_energy(model, loader, layers)
            results[subtask_name] = self._clusters_to_unit_mask(
                clusters_by_layer,
                energy,
                mask_name=f"atlas:{subtask_name}",
                extra_metadata={"subtask": subtask_name},
            )
        return results
