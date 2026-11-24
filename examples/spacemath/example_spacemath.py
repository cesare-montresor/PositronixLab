"""
Minimal CPU example demonstrating the `positronixlab.spacemath` v2 pipeline:
  1. Scoring linear and convolutional activations
  2. Tracking per-unit stats and inverse-frequency weighting
  3. Global and cluster-scoped contrastive optimization
  4. Auxiliary repulsion + SIGReg regularization
"""

import torch
from torch import nn

from positronixlab.spacemath import (
    SpaceType,
    activations_to_scores,
    get_contrast,
    get_weighter,
    loss_batch_repulsion,
    loss_centers_repulsion,
    loss_sigreg,
    loss_sigreg_centers,
    mask_cluster_membership,
    mask_global_scope,
    stats_activation_frequency,
    stats_update_,
)


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float32

    # --- 1. Synthetic modular activations (2 latent modules of 16 units each) ---
    bs, num_units, emb_dim = 32, 32, 4
    positions = torch.randn(num_units, emb_dim, device=device, dtype=dtype, requires_grad=True)
    optimizer = torch.optim.Adam([positions], lr=0.05)
    stats = torch.zeros(num_units, 4, device=device, dtype=dtype)

    scope_global = mask_global_scope(num_units, device, dtype)
    # Two clusters (units 0..15 and 16..31)
    labels = torch.cat([torch.zeros(16, dtype=torch.long), torch.ones(16, dtype=torch.long)])
    scope_cluster = mask_cluster_membership(labels, num_units=num_units, device=device, dtype=dtype)

    contrast_fn = get_contrast("softmax_ce")
    weight_fn = get_weighter("inv_freq")

    print("=== Optimization Loop (Cosine Space, softmax_ce + SIGReg) ===")
    for step in range(1, 21):
        # Half the batch activates group 0, half activates group 1
        act = torch.rand(bs, num_units, device=device, dtype=dtype) * 0.2
        act[: bs // 2, :16] += 1.0
        act[bs // 2 :, 16:] += 1.0

        scores = activations_to_scores(act, layer_type="linear", score_method="top_k", active_q=0.5)
        stats_update_(stats, scores)
        w_on, w_off = weight_fn(scores, stats=stats, freq_alpha=0.5)

        res_global = contrast_fn(positions, w_on, w_off, scope_global, space_type=SpaceType.COSINE, tau=0.2)
        l_centers = loss_centers_repulsion(
            res_global.centers_on, res_global.centers_off, res_global.mixed, space_type=SpaceType.COSINE
        )
        l_batch = loss_batch_repulsion(res_global.centers_on, res_global.mixed, space_type=SpaceType.COSINE)
        l_sigreg = loss_sigreg_centers(
            res_global.centers_on, res_global.mixed, space_type=SpaceType.COSINE, method="weak"
        )

        loss = res_global.loss + 0.1 * l_centers + 0.05 * l_batch + 0.1 * l_sigreg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step in (1, 10, 20):
            print(
                f"step={step:02d} | total={loss.item():.4f} | contrast={res_global.loss.item():.4f} "
                f"| centers_rep={l_centers.item():.4f} | sigreg={l_sigreg.item():.4f}"
            )

    freq = stats_activation_frequency(stats)
    print(f"Activation freq range: [{freq.min().item():.2f}, {freq.max().item():.2f}]")

    # --- 2. Cluster-scoped contrast + direct SIGReg ---
    res_cluster = get_contrast("distance_log")(
        positions, w_on, w_off, scope_cluster, space_type=SpaceType.EUCLIDEAN
    )
    l_pos_sigreg = loss_sigreg(positions.detach(), space_type=SpaceType.COSINE, method="strong", sketch_dim=32)
    print(
        f"Cluster contrast (distance_log, euclidean): {res_cluster.loss.item():.4f} "
        f"| Strong SIGReg on positions: {l_pos_sigreg.item():.4f}"
    )

    # --- 3. Convolutional feature-map scoring ---
    fmap = torch.randn(8, 12, 7, 7)
    conv_scores = activations_to_scores(
        fmap,
        layer_type="conv",
        reducer="conv_l2",
        score_method="quantile",
        activation_fn=nn.ReLU(),
        apply_activation=True,
        active_q=0.75,
    )
    print(f"Conv scores shape: {tuple(conv_scores.shape)}, range: [{conv_scores.min():.2f}, {conv_scores.max():.2f}]")


if __name__ == "__main__":
    main()
