"""Headless example for positronixlab.modularui.

Populates a ModularUI store with synthetic experiment data (training metrics,
per-layer & overall space loss, 2D PCA positions, cluster snapshots, model DAG,
config, and dynamic vars), builds the Dash component tree and Plotly figures
headlessly without starting a live server, and prints a concise summary.
"""

import numpy as np

from positronixlab.modularui.modularui import ModularUI
from positronixlab.modularui.settings import SPACELOSS_FAMILY


def build_demo_ui() -> ModularUI:
    rng = np.random.default_rng(42)
    ui = ModularUI(open_browser=False, positions_maxlen=50, metrics_maxlen=100, cluster_maxlen=20)

    # 1. Experiment metadata, config, and model DAG
    ui.set_config({
        "dataset": "mnist",
        "lr": 1e-3,
        "batch_size": 64,
        "modular_space_size": 8,
    })
    ui.set_var("epoch", (5, 20))
    ui.set_var("device", "cpu")
    ui.set_observed_layers({
        "fc1": {"class_name": "Linear", "in_units": 784, "out_units": 32, "activation_fn": "ReLU"},
        "fc2": {"class_name": "Linear", "in_units": 32, "out_units": 16, "activation_fn": "ReLU"},
    })
    ui.set_model_graph({
        "nodes": {
            "fc1": {"class_name": "Linear"},
            "fc2": {"class_name": "Linear"},
            "out": {"class_name": "Linear"},
        },
        "edges": [
            {"source": "fc1", "target": "fc2"},
            {"source": "fc2", "target": "out"},
        ],
    })
    ui.set_space_stats({
        "fc1": np.column_stack([rng.integers(5, 50, size=32), rng.integers(5, 50, size=32)]),
        "fc2": np.column_stack([rng.integers(5, 50, size=16), rng.integers(5, 50, size=16)]),
    })

    # 2. Training & space-loss scalar series
    for step in range(1, 6):
        ui.add_training_point("train.loss", round(1.0 / step, 4))
        ui.add_training_point("val.loss", round(1.1 / step, 4))
        ui.add_training_point("train.accuracy", round(0.5 + 0.08 * step, 4))
        ui.add_training_point("val.accuracy", round(0.48 + 0.07 * step, 4))

        ui.add_spaceloss_point("fc1", ["loss", "loss_on"], [round(0.6 / step, 4), round(0.2 / step, 4)])
        ui.add_spaceloss_point("fc2", ["loss", "loss_on"], [round(0.5 / step, 4), round(0.15 / step, 4)])
        ui.add_overall_spaceloss_point(["loss", "loss_on"], [round(0.55 / step, 4), round(0.175 / step, 4)])

    # 3. 2D PCA positions & synced cluster snapshots
    for idx in range(5):
        pts_fc1 = rng.normal(loc=0.0, scale=1.0, size=(32, 2))
        pts_fc2 = rng.normal(loc=0.0, scale=1.0, size=(16, 2))
        ui.add_position_step("fc1", pts_fc1, index=idx)
        ui.add_position_step("fc2", pts_fc2, index=idx)

        if idx in (2, 4):
            labels = rng.integers(0, 3, size=32)
            centroids = np.array([pts_fc1[labels == k].mean(axis=0) for k in range(3)])
            ui.add_cluster("fc1", idx, labels, best_k=3, best_score=0.62, centroids=centroids)

    return ui


def main() -> None:
    ui = build_demo_ui()

    # Render components & layout headlessly (no server started)
    page_tree = ui.home_layout.page_layout()
    training_tree = ui.home_layout.training_metrics.render()
    spaceloss_tree = ui.home_layout.overall_spaceloss.render()
    dag_elements = ui.home_layout.dag._get_elements()
    board_tree = ui.home_layout.layer_board.render(visible_layers=["fc1", "fc2"])

    print("=== ModularUI Headless Summary ===")
    print(f"Registered Dash callbacks : {len(ui.app.callback_map)}")
    print(f"Debug state               : {ui.debug_state()}")
    print(f"Training groups           : {list(ui.get_training_groups().keys())}")
    print(f"Spaceloss labels          : {ui.get_labels(SPACELOSS_FAMILY)}")
    print(f"DAG elements              : {len(dag_elements)} (nodes + edges)")
    print(f"Page layout root type     : {type(page_tree).__name__}")
    print(f"Training charts signature : {training_tree.children[0].data}")
    print(f"Spaceloss charts signature: {spaceloss_tree.children[0].data}")
    print(f"Layer board cards rendered: {len(board_tree.children[1].children)}")
    print(f"fc1 cluster @ step 4      : best_k={ui.get_cluster_for_index('fc1', 4)['best_k']}")


if __name__ == "__main__":
    main()
