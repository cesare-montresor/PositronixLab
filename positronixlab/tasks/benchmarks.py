"""Unified compositionality benchmark & accuracy-vs-sparsity Pareto evaluator.

Evaluates discovered :class:`SubnetworkMask` objects (from Atlas, Csordas
Gumbel-Sigmoid, NeuroSurgeon, or random controls) under identical ablation
modes and produces structured JSON/CSV reports plus scalar summary metrics for
Ax HPO.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn

from positronixlab.tasks.discoverer import unpack_batch as _unpack_batch
from positronixlab.tasks.masks import AblationMode, MaskApplier, SubnetworkMask
from positronixlab.utils.utils import save_data


def default_eval_accuracy(model: nn.Module, dataloader: Iterable[Any]) -> float:
    """Compute top-1 accuracy on ``dataloader`` (supports 2- and 3-tuple batches)."""
    was_training = model.training
    model.eval()
    first_param = next(model.parameters(), None)
    device = first_param.device if first_param is not None else torch.device("cpu")
    correct, total = 0, 0
    try:
        with torch.no_grad():
            for batch in dataloader:
                x, y = _unpack_batch(batch)
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(dim=-1)
                correct += int((preds == y).sum().item())
                total += int(y.numel())
    finally:
        model.train(was_training)
    return correct / max(1, total)


def _mask_counts(mask: SubnetworkMask, model: nn.Module) -> Dict[str, Any]:
    """Compute unit-level and weight-level active counts & sparsities."""
    unit_mask = mask.to_unit_mask()
    weight_mask = mask.to_weight_mask(model)
    active_units = int(unit_mask.l0())
    total_units = int(unit_mask.max_l0())
    active_weights = int(weight_mask.l0())
    total_weights = int(weight_mask.max_l0())
    return {
        "active_units": active_units,
        "total_units": total_units,
        "masked_units": total_units - active_units,
        "unit_sparsity": float(unit_mask.sparsity()),
        "active_weights": active_weights,
        "total_weights": total_weights,
        "masked_weights": total_weights - active_weights,
        "weight_sparsity": float(weight_mask.sparsity()),
    }


@dataclass
class SubnetworkEvalResult:
    """Single-mask evaluation record (accuracy vs. masked neurons/weights)."""

    method: str
    subtask: str
    mask_name: str
    granularity: str
    baseline_accuracy: float
    retain_accuracy: Optional[float] = None
    zero_ablate_accuracy: Optional[float] = None
    random_ablate_accuracy: Optional[float] = None
    randomly_sampled_accuracy: Optional[float] = None
    complement_sampled_accuracy: Optional[float] = None
    off_target_zero_ablate: Dict[str, float] = field(default_factory=dict)
    active_units: int = 0
    total_units: int = 0
    masked_units: int = 0
    unit_sparsity: float = 0.0
    active_weights: int = 0
    total_weights: int = 0
    masked_weights: int = 0
    weight_sparsity: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def selectivity(self) -> Optional[float]:
        """Double-dissociation selectivity: off-target preservation minus target drop under zero_ablate."""
        if self.zero_ablate_accuracy is None or not self.off_target_zero_ablate:
            return None
        mean_off = sum(self.off_target_zero_ablate.values()) / len(self.off_target_zero_ablate)
        return mean_off - self.zero_ablate_accuracy

    def to_flat_dict(self) -> Dict[str, Any]:
        row = asdict(self)
        row["selectivity"] = self.selectivity
        row["off_target_zero_ablate"] = json.dumps(self.off_target_zero_ablate, sort_keys=True)
        row["metadata"] = json.dumps(self.metadata, sort_keys=True, default=str)
        return row


@dataclass
class BenchmarkReport:
    """Structured output of :class:`ModularityBenchmark`."""

    dataset_name: str
    model_arch: str
    baselines: Dict[str, float]
    results: List[SubnetworkEvalResult] = field(default_factory=list)
    alignments: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "model_arch": self.model_arch,
            "baselines": dict(self.baselines),
            "results": [asdict(r) for r in self.results],
            "alignments": list(self.alignments),
            "metadata": dict(self.metadata),
        }

    def summary_metrics(self) -> Dict[str, float]:
        """Aggregate scalar metrics per method (directly usable by Ax HPO)."""
        by_method: Dict[str, List[SubnetworkEvalResult]] = {}
        for r in self.results:
            by_method.setdefault(r.method, []).append(r)

        summary: Dict[str, float] = {}
        for method, rows in by_method.items():
            ret = [r.retain_accuracy for r in rows if r.retain_accuracy is not None]
            abl = [r.zero_ablate_accuracy for r in rows if r.zero_ablate_accuracy is not None]
            sel = [r.selectivity for r in rows if r.selectivity is not None]
            if ret:
                summary[f"{method}_retain_acc"] = sum(ret) / len(ret)
            if abl:
                summary[f"{method}_zero_ablate_acc"] = sum(abl) / len(abl)
            if sel:
                summary[f"{method}_selectivity"] = sum(sel) / len(sel)
            summary[f"{method}_unit_sparsity"] = sum(r.unit_sparsity for r in rows) / len(rows)
            summary[f"{method}_weight_sparsity"] = sum(r.weight_sparsity for r in rows) / len(rows)
        return summary

    def save_json(self, path: str | Path, **kwargs: Any) -> Path:
        out = Path(path)
        kwargs.setdefault("indent", 2)
        kwargs.setdefault("sort_keys", True)
        kwargs.setdefault("default", str)
        save_data(str(out), self.to_dict(), **kwargs)
        return out

    def save_csv(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        rows = [r.to_flat_dict() for r in self.results]
        if not rows:
            out.write_text("", encoding="utf-8")
            return out
        fieldnames = list(rows[0].keys())
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return out


class ModularityBenchmark:
    """Head-to-head compositionality evaluator for Atlas, Csordas, and NeuroSurgeon masks."""

    def __init__(
        self,
        eval_fn: Callable[[nn.Module, Iterable[Any]], float] = default_eval_accuracy,
        modes: Sequence[AblationMode] = ("retain", "zero_ablate"),
        seed: int = 42,
    ) -> None:
        self.eval_fn = eval_fn
        self.modes = tuple(modes)
        self.seed = seed

    def evaluate_mask(
        self,
        model: nn.Module,
        mask: SubnetworkMask,
        target_loader: Iterable[Any],
        *,
        method: str,
        subtask: str,
        baseline_accuracy: Optional[float] = None,
        off_target_loaders: Optional[Dict[str, Iterable[Any]]] = None,
    ) -> SubnetworkEvalResult:
        """Evaluate a single ``SubnetworkMask`` on ``target_loader`` (and optional off-target splits)."""
        base_acc = (
            float(baseline_accuracy)
            if baseline_accuracy is not None
            else float(self.eval_fn(model, target_loader))
        )
        counts = _mask_counts(mask, model)

        mode_scores: Dict[str, Optional[float]] = {
            "retain": None,
            "zero_ablate": None,
            "random_ablate": None,
            "randomly_sampled": None,
            "complement_sampled": None,
        }
        off_target_scores: Dict[str, float] = {}

        for mode in self.modes:
            with MaskApplier(model, mask, mode=mode, seed=self.seed):
                mode_scores[mode] = float(self.eval_fn(model, target_loader))
                if mode == "zero_ablate" and off_target_loaders:
                    for other_name, other_loader in off_target_loaders.items():
                        if other_name == subtask:
                            continue
                        off_target_scores[other_name] = float(self.eval_fn(model, other_loader))

        return SubnetworkEvalResult(
            method=method,
            subtask=subtask,
            mask_name=mask.name,
            granularity=mask.granularity,
            baseline_accuracy=base_acc,
            retain_accuracy=mode_scores["retain"],
            zero_ablate_accuracy=mode_scores["zero_ablate"],
            random_ablate_accuracy=mode_scores["random_ablate"],
            randomly_sampled_accuracy=mode_scores["randomly_sampled"],
            complement_sampled_accuracy=mode_scores["complement_sampled"],
            off_target_zero_ablate=off_target_scores,
            metadata=dict(mask.metadata),
            **counts,
        )

    def compare_methods(
        self,
        model: nn.Module,
        masks_by_method: Dict[str, Dict[str, SubnetworkMask | Sequence[SubnetworkMask]]],
        subtask_loaders: Dict[str, Iterable[Any]],
        *,
        dataset_name: str = "unknown",
        model_arch: str = "unknown",
        reference_method: str = "atlas",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkReport:
        """Compare multiple discovery methods (supporting single masks or sparsity-sweep sequences)."""
        baselines = {
            subtask: float(self.eval_fn(model, loader))
            for subtask, loader in subtask_loaders.items()
        }

        results: List[SubnetworkEvalResult] = []
        primary_masks: Dict[str, Dict[str, SubnetworkMask]] = {}

        for method, per_subtask in masks_by_method.items():
            primary_masks[method] = {}
            for subtask, mask_or_seq in per_subtask.items():
                if subtask not in subtask_loaders:
                    continue
                mask_list: List[SubnetworkMask] = (
                    [mask_or_seq]
                    if isinstance(mask_or_seq, SubnetworkMask)
                    else list(mask_or_seq)
                )
                if mask_list:
                    primary_masks[method][subtask] = mask_list[0]
                for mask in mask_list:
                    results.append(
                        self.evaluate_mask(
                            model,
                            mask,
                            subtask_loaders[subtask],
                            method=method,
                            subtask=subtask,
                            baseline_accuracy=baselines[subtask],
                            off_target_loaders=subtask_loaders,
                        )
                    )

        # Cross-method alignment against reference_method (unit & weight IoU)
        alignments: List[Dict[str, Any]] = []
        ref_map = primary_masks.get(reference_method, {})
        for method, per_subtask in primary_masks.items():
            if method == reference_method:
                continue
            for subtask, other_mask in per_subtask.items():
                ref_mask = ref_map.get(subtask)
                if ref_mask is None:
                    continue
                ref_u, oth_u = ref_mask.to_unit_mask(), other_mask.to_unit_mask()
                ref_w, oth_w = ref_mask.to_weight_mask(model), other_mask.to_weight_mask(model)
                alignments.append(
                    {
                        "reference_method": reference_method,
                        "other_method": method,
                        "subtask": subtask,
                        "unit_iou": float(ref_u.iou(oth_u)),
                        "weight_iou": float(ref_w.iou(oth_w)),
                        "unit_overlap": float(ref_u.overlap_fraction(oth_u)),
                        "weight_overlap": float(ref_w.overlap_fraction(oth_w)),
                    }
                )

        return BenchmarkReport(
            dataset_name=dataset_name,
            model_arch=model_arch,
            baselines=baselines,
            results=results,
            alignments=alignments,
            metadata=dict(metadata or {}),
        )
