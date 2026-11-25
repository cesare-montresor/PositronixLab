"""End-to-end CPU example: Unified SOTA ModelLoader + Mask Discovery + Ablation.

Demonstrates loading Csordas & Lepori architectures through ``ModelLoader``,
discovering subnetwork masks via Csordas Gumbel-Sigmoid (and NeuroSurgeon when
available), comparing masks with ``SubnetworkMask`` set algebra, and running
leak-free ablations via ``MaskApplier``.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from positronixlab.baselines import (
    CsordasGumbelDiscoverer,
    ModelLoader,
    NeuroSurgeonDiscoverer,
    is_neurosurgeon_available,
)
from positronixlab.tasks import MaskApplier, RandomMaskDiscoverer, SubnetworkMask


def evaluate_acc(model: torch.nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            preds = model(x).argmax(dim=-1)
            correct += int((preds == y).sum().item())
            total += int(y.numel())
    return correct / max(1, total)


def main() -> None:
    torch.manual_seed(42)
    print("Registered SOTA models:", ModelLoader.list_models())

    # 1. Create a Csordas MLP via the unified ModelLoader
    model = ModelLoader.create(
        "csordas:mlp", n_inputs=12, n_outputs=3, layer_sizes=[24, 16], device="cpu"
    )
    x = torch.randn(128, 12)
    y = ((x[:, 0] + x[:, 1]) > 0).long() + (x[:, 2] > 0).long()
    loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=False)

    # Quick base training
    opt = torch.optim.Adam(model.parameters(), lr=3e-2)
    for _ in range(15):
        for bx, by in loader:
            opt.zero_grad()
            loss = model.loss(bx, by, model(bx))
            loss.backward()
            opt.step()

    base_acc = evaluate_acc(model, loader)
    print(f"Base MLP accuracy: {base_acc:.3f}")

    # 2. Discover a Csordas Gumbel-Sigmoid weight mask
    csordas_mask = CsordasGumbelDiscoverer(
        mask_loss_weight=5e-3, lr=5e-2, epochs=5, init_logit=1.5
    ).discover(model, loader, mask_name="csordas")
    print(
        f"Csordas mask: L0={csordas_mask.l0()}/{csordas_mask.max_l0()} "
        f"(sparsity={csordas_mask.sparsity():.2%})"
    )

    # 3. Compare with a synthetic unit-level Atlas mask & NeuroSurgeon (if available)
    atlas_unit_mask = SubnetworkMask(
        name="atlas_mock",
        granularity="unit",
        masks={
            "layers.0": torch.arange(24) < 12,
            "layers.1": torch.arange(16) < 8,
            "layers.2": torch.ones(3, dtype=torch.bool),
        },
    )
    atlas_weight_mask = atlas_unit_mask.to_weight_mask(model)
    print(f"Atlas vs Csordas weight IoU: {atlas_weight_mask.iou(csordas_mask):.3f}")

    if is_neurosurgeon_available():
        ns_mask = NeuroSurgeonDiscoverer(
            mask_method="continuous_sparsification",
            mask_unit="weight",
            l0_lambda=1e-3,
            lr=5e-2,
            epochs=5,
            max_temp=20.0,
        ).discover(model, loader, mask_name="neurosurgeon")
        print(
            f"NeuroSurgeon mask: L0={ns_mask.l0()}/{ns_mask.max_l0()} "
            f"(sparsity={ns_mask.sparsity():.2%}), "
            f"IoU vs Csordas={ns_mask.iou(csordas_mask):.3f}"
        )
    else:
        print("NeuroSurgeon not accessible in current environment; skipping NS discoverer.")

    # 4. Run unified ablation modes via MaskApplier
    rand_mask = RandomMaskDiscoverer(density=0.4, seed=0).discover(model)
    for mode in ("retain", "zero_ablate", "random_ablate", "randomly_sampled"):
        with MaskApplier(model, csordas_mask, mode=mode, seed=0):
            acc = evaluate_acc(model, loader)
        print(f"  MaskApplier(csordas, mode={mode:<17s}) -> acc={acc:.3f}")

    # Verify Lepori CVR ResNet creation through the exact same loader
    cvr = ModelLoader.create("lepori:cvr_resnet18", mlp_hidden_dim=64, mlp_out_dim=16, device="cpu")
    cvr_out = cvr(torch.randn(1, 4, 3, 64, 64))
    print(f"Lepori CVR ResNet18 output shape: {tuple(cvr_out.shape)} (maskable layers: {len(cvr.maskable_layers())})")
    assert evaluate_acc(model, loader) == base_acc, "Base model weights must be restored!"
    print("All checks passed cleanly.")


if __name__ == "__main__":
    main()
