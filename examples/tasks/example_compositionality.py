"""
Self-contained CPU example demonstrating downstream modularity and
compositionality evaluation without requiring external .pkl or .pth fixtures.

Run from the repository root:
    .venv/bin/python positronixlab/tasks/examples/example_compositionality.py
"""

from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from positronixlab.pipeline import PipelineConfigV1
from positronixlab.tasks.compositionality.atlas_check import AtlasCheck
from positronixlab.tasks.compositionality.compositionality import CompositionalityCheck
from positronixlab.tasks.compositionality.weightmask_check import (
    WeightmaskCheck,
    WeightMasksMetrics,
)


class SyntheticModularMLP(nn.Module):
    """3-class block-modular MLP with dedicated hidden neurons per class."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(num_classes * 2, num_classes * 2)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(num_classes * 2, num_classes)
        self.device = torch.device("cpu")

        with torch.no_grad():
            self.fc1.weight.zero_()
            self.fc1.bias.zero_()
            self.fc2.weight.zero_()
            self.fc2.bias.zero_()
            for c in range(num_classes):
                self.fc1.weight[2 * c : 2 * c + 2, 2 * c : 2 * c + 2] = torch.eye(2)
                self.fc2.weight[c, 2 * c : 2 * c + 2] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def build_synthetic_dataprovider(num_classes: int = 3, per_class: int = 24, batch_size: int = 18):
    torch.manual_seed(42)
    xs, ys = [], []
    for c in range(num_classes):
        x_c = torch.randn(per_class, num_classes * 2) * 0.15
        x_c[:, 2 * c : 2 * c + 2] += 3.0
        xs.append(x_c)
        ys.append(torch.full((per_class,), c, dtype=torch.long))

    x_all = torch.cat(xs, dim=0)
    y_all = torch.cat(ys, dim=0)
    # Interleave so every batch contains all classes
    perm = torch.arange(num_classes * per_class).reshape(num_classes, per_class).T.reshape(-1)
    ds = TensorDataset(x_all[perm], y_all[perm])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return SimpleNamespace(num_classes=num_classes, train=loader, val=loader, test=loader)


def adapt_atlas_masks(raw_masks):
    """Converts AtlasCheck 1D neuron masks (dotted keys) to StaticMaskApplier format."""
    return {
        cls_id: {
            layer_name.replace(".", "_"): mask.float().unsqueeze(1)
            for layer_name, mask in layer_masks.items()
        }
        for cls_id, layer_masks in raw_masks.items()
    }


def main():
    torch.manual_seed(0)
    num_classes = 3
    model = SyntheticModularMLP(num_classes=num_classes)
    dp = build_synthetic_dataprovider(num_classes=num_classes)
    config = PipelineConfigV1()

    # Simulated cluster assignments discovered in ModularSpace (1 cluster per class)
    clusters = {
        "fc1": torch.tensor([0, 0, 1, 1, 2, 2], device=model.device),
        "fc2": torch.tensor([0, 1, 2], device=model.device),
    }

    # 1. Atlas modularity & cluster-based masks
    atlas = AtlasCheck(config=config, model=model, dataprovider=dp, clusters=clusters)
    atlas.run()
    modularity = atlas.modularity_score(kld_target="y")
    raw_atlas_masks = atlas.extract_weightmasks(threshold=0.05)
    atlas.observer.on_forward_layer(None)  # Clean up observer hooks
    atlas_masks = adapt_atlas_masks(raw_atlas_masks)

    print("Atlas modularity (model):", {k: round(v, 4) for k, v in modularity["model"].items()})

    # 2. Csordas differentiable weightmask baseline
    orig_req_grad = {p: p.requires_grad for p in model.parameters()}
    target_layers = {"fc1": model.fc1, "fc2": model.fc2}
    csordas = WeightmaskCheck(model=model, dataprovider=dp)
    csordas_masks = csordas.run(target_layers=target_layers, num_epochs=3, lr=0.1)
    for p, req in orig_req_grad.items():
        p.requires_grad = req

    overlap_01 = WeightMasksMetrics.evaluate_network_modularity(csordas_masks[0], csordas_masks[1])
    print("Csordas mask overlap (class 0 vs 1):", {k: round(v, 4) for k, v in overlap_01.items()})

    # 3. Compositionality & cross-class interference evaluation
    checker = CompositionalityCheck(model=model, dataloader=dp.test, num_classes=num_classes)
    results = checker.run(atlas_masks=atlas_masks, csordas_masks=csordas_masks)

    print(f"Untainted accuracy: {results['untainted']['accuracy']:.4f}")
    for c, comp in results["comparison"].items():
        print(
            f"  Class {c}: "
            f"Atlas recall={comp['atlas_target_recall']:.3f}, "
            f"Csordas recall={comp['csordas_target_recall']:.3f}, "
            f"CM Fro diff={comp['cm_frobenius_diff']:.3f}"
        )


if __name__ == "__main__":
    main()
