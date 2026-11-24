"""Head-to-head CPU benchmark: PositronixLab Atlas vs. NeuroSurgeon vs. Csordas.

Uses the unified ``DataProviderLoader`` (``csordas:add_mul``), unified
``ModelLoader`` (``csordas:mlp``), and unified ``SubnetworkMask`` / ``MaskApplier``
contract to compare all three discovery methods on the exact same trained model.
"""

from __future__ import annotations

import torch

from positronixlab.dataprovider.sota import DataProviderLoader
from positronixlab.metamodel.sota import ModelLoader, is_neurosurgeon_available
from positronixlab.tasks.sota import (
    AtlasDiscoverer,
    CsordasGumbelDiscoverer,
    MaskApplier,
    NeuroSurgeonDiscoverer,
)


def eval_acc(model: torch.nn.Module, loader) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            preds = model(x).argmax(dim=-1)
            correct += int((preds == y).sum().item())
            total += int(y.numel())
    return correct / max(1, total)


def main() -> None:
    torch.manual_seed(0)

    # 1. Load Csordas Add/Mul compositional benchmark
    dp = DataProviderLoader.create(
        "csordas:add_mul", n_train=512, n_val=128, n_test=128, n_digits=1, batch_size=64
    )
    model = ModelLoader.create(
        "csordas:mlp",
        n_inputs=dp.num_features,
        n_outputs=dp.num_classes,
        layer_sizes=[48, 32],
        device="cpu",
    )

    # 2. Quick base model training on full (add + mul) task
    opt = torch.optim.Adam(model.parameters(), lr=2e-2)
    for _ in range(12):
        for x, y in dp.train_loader():
            opt.zero_grad()
            loss = model.loss(x, y, model(x))
            loss.backward()
            opt.step()

    val_add = dp.get_subtask("add").val_loader()
    val_mul = dp.get_subtask("mul").val_loader()
    print(
        f"Base accuracy -> full={eval_acc(model, dp.val_loader()):.3f}, "
        f"add={eval_acc(model, val_add):.3f}, mul={eval_acc(model, val_mul):.3f}"
    )

    # 3. Atlas: ONE unsupervised pass -> masks for BOTH subtasks
    atlas = AtlasDiscoverer(atlas_epochs=2, space_dim=8, k_min=2, k_max=4)
    atlas_masks = atlas.discover_subtasks(
        model,
        reference_dataloader=dp.val_loader(),
        subtask_dataloaders={"add": val_add, "mul": val_mul},
    )

    # 4. Csordas Gumbel-Sigmoid: supervised per-subtask mask training
    csordas = CsordasGumbelDiscoverer(mask_loss_weight=1e-2, lr=5e-2, epochs=4, init_logit=1.0)
    cs_add = csordas.discover(model, val_add, mask_name="csordas:add")
    cs_mul = csordas.discover(model, val_mul, mask_name="csordas:mul")

    methods = {
        "Atlas (unit)": (atlas_masks["add"], atlas_masks["mul"]),
        "Csordas (weight)": (cs_add, cs_mul),
    }

    # 5. NeuroSurgeon (Lepori): supervised continuous sparsification (neuron mode)
    if is_neurosurgeon_available():
        ns = NeuroSurgeonDiscoverer(
            mask_method="continuous_sparsification",
            mask_unit="neuron",
            l0_lambda=1e-3,
            lr=5e-2,
            epochs=4,
            max_temp=20.0,
        )
        ns_add = ns.discover(model, val_add, mask_name="ns:add")
        ns_mul = ns.discover(model, val_mul, mask_name="ns:mul")
        methods["NeuroSurgeon (neuron)"] = (ns_add, ns_mul)

    # 6. Unified evaluation across all methods
    print("\n--- Head-to-Head Compositional Ablation Summary ---")
    for label, (m_add, m_mul) in methods.items():
        with MaskApplier(model, m_add, mode="retain"):
            ret_add = eval_acc(model, val_add)
        with MaskApplier(model, m_add, mode="zero_ablate"):
            abl_add_on_add = eval_acc(model, val_add)
            abl_add_on_mul = eval_acc(model, val_mul)
        print(
            f"{label:<22s} | sparsity={m_add.sparsity():5.1%} | "
            f"IoU(add,mul)={m_add.iou(m_mul):.3f} | "
            f"retain(add)={ret_add:.3f} | "
            f"ablate_add -> add={abl_add_on_add:.3f}, mul={abl_add_on_mul:.3f}"
        )

    # Cross-method alignment (Atlas vs Csordas weight-level IoU)
    atlas_w_add = atlas_masks["add"].to_weight_mask(model)
    print(f"\nAtlas vs Csordas (add) weight IoU: {atlas_w_add.iou(cs_add):.3f}")
    if "NeuroSurgeon (neuron)" in methods:
        print(f"Atlas vs NeuroSurgeon (add) unit IoU: {atlas_masks['add'].iou(methods['NeuroSurgeon (neuron)'][0]):.3f}")


if __name__ == "__main__":
    main()
