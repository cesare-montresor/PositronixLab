import threading
from types import SimpleNamespace

import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import numpy as np
import plotly.graph_objects as go
import pytest
import torch
from dash import dcc, html

from positronixlab.modularui.components import (
    ConfigOverlay,
    Container,
    ExpandableSection,
    ModelDAGDisplay,
    Title,
    VarsDisplay,
    _format_config_value,
)
from positronixlab.modularui.docks import DashboardLayout, Panel
from positronixlab.modularui.layouts import HomeLayout, _BoundWidget
from positronixlab.modularui.modularui import ModularUI, UILocks, _to_numpy
from positronixlab.modularui.plots import (
    GroupedChartsDisplay,
    LayerBoardDisplay,
    MetricRow,
    SpaceLossRow,
    _line_figure,
    loss_then_accuracy_first,
)
from positronixlab.modularui.settings import GLOBAL_TRACE_NAME, SPACELOSS_FAMILY


@pytest.fixture
def ui() -> ModularUI:
    return ModularUI(
        open_browser=False,
        positions_maxlen=5,
        metrics_maxlen=10,
        cluster_maxlen=5,
    )


# ---------------------------------------------------------------------------
# UILocks & helpers
# ---------------------------------------------------------------------------


class TestUILocksAndHelpers:
    def test_locks_are_frozen_and_distinct(self):
        locks = UILocks()
        with pytest.raises(AttributeError):
            locks.vars = threading.Lock()  # type: ignore[misc]

        lock_objs = [
            locks.vars,
            locks.config,
            locks.model_graph,
            locks.stats,
            locks.metrics,
            locks.spaceloss,
            locks.positions,
            locks.observed_layers,
        ]
        assert len({id(l) for l in lock_objs}) == len(lock_objs)

    def test_to_numpy_handles_tensor_and_array_like(self):
        t = torch.tensor([[1.0, 2.0]], requires_grad=True)
        arr = _to_numpy(t)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (1, 2)
        np.testing.assert_allclose(arr, [[1.0, 2.0]])

        scalar = _to_numpy(3.5)
        assert scalar.size == 1
        assert float(scalar) == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# ModularUI data store
# ---------------------------------------------------------------------------


class TestModularUIStore:
    def test_initial_state_and_callback_registration(self, ui: ModularUI):
        assert not ui.is_running()
        assert ui.get_all_vars_snapshot() == {}
        assert ui.get_config() == {}
        assert ui.get_model_graph() == {"nodes": {}, "edges": []}
        assert ui.get_model_graph_version() == 0
        assert ui.get_known_layer_ids() == []
        assert len(ui.app.callback_map) == 15
        assert ui.homedir().endswith("modularui")

    def test_dynamic_vars_and_versioning(self, ui: ModularUI):
        assert ui.get_var("epoch") is None
        assert ui.get_var("epoch", "default") == "default"
        assert ui.get_var_version("epoch") == 0

        ui.set_var("epoch", (1, 10))
        assert ui.get_var("epoch") == (1, 10)
        assert ui.get_var_version("epoch") == 1

        ui.set_var("epoch", (2, 10))
        assert ui.get_var_version("epoch") == 2

        snap = ui.get_all_vars_snapshot()
        snap["injected"] = 123
        assert ui.get_var("injected") is None

    def test_config_dict_and_to_dict_object(self, ui: ModularUI):
        ui.set_config({"lr": 1e-3, "batch_size": 32})
        cfg = ui.get_config()
        assert cfg == {"lr": 1e-3, "batch_size": 32}
        cfg["lr"] = 999
        assert ui.get_config()["lr"] == 1e-3

        obj = SimpleNamespace(to_dict=lambda: {"lr": 5e-4, "epochs": 20})
        ui.set_config(obj)
        assert ui.get_config() == {"lr": 5e-4, "epochs": 20}

    def test_model_graph_version_increments(self, ui: ModularUI):
        graph = {
            "nodes": {"fc1": {"class_name": "Linear"}, "fc2": {"class_name": "Linear"}},
            "edges": [{"source": "fc1", "target": "fc2"}],
        }
        ui.set_model_graph(graph)
        assert ui.get_model_graph_version() == 1
        assert ui.get_model_graph()["nodes"]["fc1"]["class_name"] == "Linear"

    def test_observed_layers_and_stats(self, ui: ModularUI):
        ui.set_observed_layers({
            "fc1": {"class_name": "Linear", "in_units": 16, "out_units": 8},
            "fc2": "raw_meta",
        })
        assert ui.get_observed_layer_ids() == ["fc1", "fc2"]
        assert ui.get_observed_layer_info("fc1")["class_name"] == "Linear"
        assert ui.get_observed_layer_info("fc2") == {"value": "raw_meta"}
        assert ui.get_observed_layer_info("missing") == {}

        ui.set_observed_layers(["l1", "l2"])
        assert ui.get_observed_layer_ids() == ["l1", "l2"]
        assert ui.get_observed_layer_info("l1") == {}

        stats = {"fc1": np.array([[10.0, 2.0], [5.0, 5.0]])}
        ui.set_space_stats(stats)
        np.testing.assert_array_equal(ui.get_layer_stats("fc1"), stats["fc1"])
        assert ui.get_layer_stats("missing") is None

    def test_scalar_metrics_and_rolling_window(self, ui: ModularUI):
        for i in range(15):
            ui.add_point("train", "global", "loss", float(i))
        # metrics_maxlen=10 -> keeps last 10
        assert ui.get_series("train", "global")["loss"] == [float(i) for i in range(5, 15)]

        # Non-scalar arrays are silently ignored
        ui.add_point("train", "global", "loss", np.array([1.0, 2.0]))
        assert len(ui.get_series("train", "global")["loss"]) == 10

        # Tensor scalar works
        ui.add_point("val", "global", "loss", torch.tensor(0.25))
        assert ui.get_series("val", "global")["loss"] == [pytest.approx(0.25)]

        # ingest_stat
        ui.ingest_stat("train", {"layer_id": "global", "label": "acc", "value": 0.9, "seq": 1, "t": 1.0})
        assert ui.get_series("train", "global")["acc"] == [pytest.approx(0.9)]

    def test_training_and_spaceloss_shims_and_groups(self, ui: ModularUI):
        ui.add_training_point("train.loss", 0.8)
        ui.add_training_point("val.loss", 0.9)
        ui.add_training_point("accuracy", 0.75)  # no dot -> family="metrics"

        groups = ui.get_training_groups()
        assert groups["loss"] == {"train": [0.8], "val": [0.9]}
        assert groups["accuracy"] == {"metrics": [0.75]}
        assert ui.get_training_metrics() == {
            "train.loss": [0.8],
            "val.loss": [0.9],
            "metrics.accuracy": [0.75],
        }

        ui.add_spaceloss_point("fc1", ["loss", "loss_on"], [0.5, 0.2])
        ui.add_spaceloss_point("fc2", None, {"loss": 0.3, "loss_on": 0.1})
        ui.add_overall_spaceloss_point(["loss", "loss_on"], [0.4, 0.15])

        assert ui.get_labels(SPACELOSS_FAMILY) == ["loss", "loss_on"]
        assert ui.overall_spaceloss == {"loss": [0.4], "loss_on": [0.15]}
        assert ui.get_layer_spaceloss("fc1") == {"loss": [0.5], "loss_on": [0.2]}

        fam = ui.get_family_groups(SPACELOSS_FAMILY)
        assert fam["loss"]["fc1"] == [0.5]
        assert fam["loss"]["fc2"] == [0.3]
        assert fam["loss"][GLOBAL_TRACE_NAME] == [0.4]

    def test_training_groups_merges_metrics_history_fallback(self, ui: ModularUI):
        ui.set_var("metrics_history", {
            "train": {"loss": [[1.0, 0.9], [2.0, 0.7], [3.0, 0.5]]},
            "val": {"loss": [0.95, 0.8]},
            SPACELOSS_FAMILY: {"loss": [[1.0, 0.1]]},  # skipped by _training_history_groups
        })
        groups = ui.get_training_groups()
        assert groups["loss"]["train"] == [0.9, 0.7, 0.5]
        assert groups["loss"]["val"] == [0.95, 0.8]
        assert SPACELOSS_FAMILY not in groups["loss"]

        # Shorter live series does not truncate longer metrics_history
        ui.add_training_point("train.loss", 0.4)
        assert ui.get_training_groups()["loss"]["train"] == [0.9, 0.7, 0.5]

        # Longer live series wins over metrics_history
        for v in (0.3, 0.2, 0.1):
            ui.add_training_point("train.loss", v)
        assert ui.get_training_groups()["loss"]["train"] == [0.4, 0.3, 0.2, 0.1]

    def test_positions_rolling_window_and_cluster_lookup(self, ui: ModularUI):
        # PCA progress before positions exist
        ui.set_position_progress({"required": 100, "layers": {"fc1": 40}})
        assert ui.get_pca_progress("fc1") == {"collected": 40, "required": 100}
        assert ui.get_known_layer_ids() == ["fc1"]

        # Add 8 contiguous steps into positions_maxlen=5 -> retained abs indices: [3, 4, 5, 6, 7]
        for idx in range(8):
            pts = np.full((4, 2), float(idx))
            ui.add_position_step("fc1", pts, index=idx)

        assert ui.get_pca_progress("fc1") is None
        assert ui.get_position_steps("fc1") == 5
        assert ui.get_position_offset("fc1") == 3
        np.testing.assert_allclose(ui.get_position_at("fc1", 0), np.full((4, 2), 3.0))
        np.testing.assert_allclose(ui.get_position_at("fc1", 4), np.full((4, 2), 7.0))
        assert ui.get_position_at("fc1", -1) is None
        assert ui.get_position_at("fc1", 5) is None

        # Add clusters at absolute indices 4 and 6
        ui.add_cluster("fc1", 4, np.array([0, 0, 1, 1]), 2, 0.6, np.array([[0.0, 0.0], [1.0, 1.0]]))
        ui.add_cluster("fc1", 6, np.array([0, 1, 1, 0]), 2, 0.8, np.array([[0.2, 0.2], [0.8, 0.8]]))

        # relative 0 -> abs 3 (before first cluster at 4)
        assert ui.get_cluster_for_index("fc1", 0) is None
        # relative 1 -> abs 4 -> cluster@4
        assert ui.get_cluster_for_index("fc1", 1)["position_index"] == 4
        # relative 2 -> abs 5 -> cluster@4
        assert ui.get_cluster_for_index("fc1", 2)["position_index"] == 4
        # relative 3 -> abs 6 -> cluster@6
        assert ui.get_cluster_for_index("fc1", 3)["position_index"] == 6
        # relative 4 -> abs 7 -> cluster@6
        assert ui.get_cluster_for_index("fc1", 4)["position_index"] == 6

    def test_reset_and_debug_state(self, ui: ModularUI):
        ui.set_var("epoch", (1, 5))
        ui.set_config({"lr": 0.01})
        ui.add_training_point("train.loss", 0.5)
        ui.add_position_step("fc1", np.zeros((3, 2)), index=0)

        dbg = ui.debug_state()
        assert dbg["series"]["train"]["global"]["loss"] == 1
        assert "epoch" in dbg["vars"]
        assert dbg["layers"] == ["fc1"]

        ui.reset()
        assert ui.get_all_vars_snapshot() == {}
        assert ui.get_config() == {}
        assert ui.get_training_groups() == {}
        assert ui.get_known_layer_ids() == []

    def test_concurrent_reads_and_writes(self, ui: ModularUI):
        errors = []

        def writer():
            try:
                for i in range(100):
                    ui.add_training_point("train.loss", float(i))
                    ui.add_position_step("fc1", np.full((3, 2), float(i)), index=i)
                    ui.set_var("step", i)
            except Exception as exc:
                errors.append(exc)

        def reader():
            try:
                for _ in range(100):
                    ui.get_training_groups()
                    ui.get_position_steps("fc1")
                    ui.get_position_at("fc1", 0)
                    ui.get_all_vars_snapshot()
                    ui.debug_state()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        assert errors == []


# ---------------------------------------------------------------------------
# Components, Docks, Plots & Layouts (Headless Rendering)
# ---------------------------------------------------------------------------


class TestWidgetsAndFigures:
    def test_basic_components(self):
        title = Title("My Title")
        assert isinstance(title(), html.H1)
        assert title().children == "My Title"

        sec = ExpandableSection(expanded=True).render("Section", "body")
        assert isinstance(sec, html.Details)
        assert sec.open is True

        container = Container("cid", title).render()
        assert isinstance(container, html.Div)
        assert container.id == "cid"

        vd = VarsDisplay()
        empty = vd.render({})
        assert empty.children == "no other vars"
        populated = vd.render({"b": 2, "a": {"k": "v"}})
        assert len(populated.children) == 2

    def test_config_overlay_rendering(self):
        assert _format_config_value("short") == "short"
        long_val = "x" * 100
        formatted = _format_config_value(long_val, max_len=10)
        assert len(formatted) == 10 and formatted.endswith("…")

        overlay = ConfigOverlay()
        assert isinstance(overlay.render_button(), dbc.Button)
        assert isinstance(overlay.render_overlay(), dbc.Modal)

        none_pinned = overlay.render_selected({"lr": 1e-3}, [])
        assert none_pinned.children == "config: none pinned"

        pinned = overlay.render_selected({"lr": 1e-3, "bs": 64}, ["lr", "missing"])
        assert len(pinned.children) == 1
        assert "lr: 0.001" in pinned.children[0].children

    def test_model_dag_display_elements_and_render(self, ui: ModularUI):
        ui.set_model_graph({
            "nodes": {"fc1": {"class_name": "Linear"}, "fc2": {}},
            "edges": [{"source": "fc1", "target": "fc2"}],
        })
        dag = ModelDAGDisplay(ui=ui)
        elements = dag._get_elements()
        assert len(elements) == 3
        node_el = next(e for e in elements if e["data"].get("id") == "fc1")
        assert node_el["data"]["label"] == "fc1\n(Linear)"

        rendered = dag.render()
        assert isinstance(rendered, dbc.Card)
        cyto_nodes = [c for c in rendered.children.children if isinstance(c, cyto.Cytoscape)]
        assert len(cyto_nodes) == 1

    def test_panel_and_dashboard_layout(self):
        p1 = Panel("p1", "Panel 1", Title("1"))
        p2 = Panel("p2", "Panel 2", Title("2"))
        p3 = Panel("p3", "Panel 3", Title("3"))
        left = Panel("left", "Left", Title("L"))
        main = Panel("main", "Main", Title("M"))

        max_div = p1.render(maximized=True, grid_area="top-a")
        assert max_div.style["position"] == "fixed"
        assert max_div.style["gridArea"] == "top-a"

        dash_layout = DashboardLayout([p1, p2, p3], left, main)
        grid_all = dash_layout.render()
        assert len(grid_all.children) == 5

        grid_collapsed = dash_layout.render({"p2": {"collapsed": True}})
        assert len(grid_collapsed.children) == 4
        assert all(c.id != "p2" for c in grid_collapsed.children)

    def test_line_figure_and_metric_rows(self, ui: ModularUI):
        fig = _line_figure("loss", 120, ["train", "val"], legend=True)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2
        assert fig.layout.height == 120
        assert fig.layout.showlegend is True

        fig_emph = _line_figure("loss", 110, ["fc1", GLOBAL_TRACE_NAME], emphasize=GLOBAL_TRACE_NAME)
        assert fig_emph.data[0].line.width == 1
        assert fig_emph.data[1].line.width == 3

        assert loss_then_accuracy_first(["z_metric", "acc", "val_loss", "loss"]) == [
            "loss",
            "val_loss",
            "acc",
            "z_metric",
        ]

        row = MetricRow("m", per_layer=True)
        assert row.render([], layer_id="fc1").children == "no data yet"
        rendered = row.render(["loss", "loss_on"], layer_id="fc1")
        assert len(rendered.children) == 2

        ui.add_spaceloss_point("fc1", ["loss"], [0.5])
        sl_row = SpaceLossRow("sl", labels_fn=lambda: ui.get_labels(SPACELOSS_FAMILY))
        assert len(sl_row.render("fc1").children) == 1

    def test_grouped_charts_and_layer_board_render(self, ui: ModularUI):
        ui.add_training_point("train.loss", 0.5)
        ui.add_training_point("val.loss", 0.6)
        ui.set_observed_layers({"fc1": {"class_name": "Linear", "in_units": 8, "out_units": 4}})
        ui.add_spaceloss_point("fc1", ["loss"], [0.2])

        gc = GroupedChartsDisplay("gc", groups_fn=ui.get_training_groups)
        tree = gc.render()
        assert isinstance(tree.children[0], dcc.Store)
        assert tree.children[0].data == ["loss|train", "loss|val"]

        lb = LayerBoardDisplay(ui=ui)
        board = lb.render(visible_layers=["fc1"])
        assert isinstance(board.children[0], dcc.Store)
        assert board.children[0].data == {"layers": ["fc1"], "labels": ["loss"]}
        assert len(board.children[1].children) == 1

    def test_home_layout_page_and_helpers(self, ui: ModularUI):
        page = ui.home_layout.page_layout()
        assert isinstance(page, dbc.Container)

        assert HomeLayout._format_epoch(None).children == "epoch: —"
        assert HomeLayout._format_epoch((3, 10)).children == "epoch: 3 / 10"

        bw = _BoundWidget(VarsDisplay(), lambda: {"k": 1})
        rendered = bw.render()
        assert isinstance(rendered, html.Div)


# ---------------------------------------------------------------------------
# Strict xfail tests documenting confirmed bugs in library source code
# ---------------------------------------------------------------------------


class TestConfirmedBugs:
    @pytest.mark.xfail(
        strict=True,
        reason="Bug (plots.py:237-242): _filtered_layer_ids treats empty list [] as unfiltered and returns all layers.",
    )
    def test_layer_board_empty_visible_layers_hides_all(self, ui: ModularUI):
        ui.set_observed_layers(["fc1", "fc2"])
        lb = LayerBoardDisplay(ui=ui)
        assert lb._filtered_layer_ids([]) == []

    @pytest.mark.xfail(
        strict=True,
        reason="Bug (layouts.py:98-104): _format_epoch crashes with TypeError when given a scalar epoch int.",
    )
    def test_format_epoch_handles_scalar_int(self):
        div = HomeLayout._format_epoch(5)
        assert "5" in div.children

    @pytest.mark.xfail(
        strict=True,
        reason="Bug (layouts.py:34 & modularui.py:123): external_stylesheets contains nonexistent relative themes/*.css paths.",
    )
    def test_external_stylesheets_has_no_broken_relative_theme_paths(self, ui: ModularUI):
        for sheet in ui.app.config.external_stylesheets:
            assert not sheet.startswith("themes/")

    @pytest.mark.xfail(
        strict=True,
        reason="Bug (modularui.py:257-266): add_point raises TypeError on None instead of ignoring non-numeric scalar.",
    )
    def test_add_point_ignores_none_value(self, ui: ModularUI):
        ui.add_point("train", "global", "loss", None)
        assert ui.get_series("train", "global") == {}

    @pytest.mark.xfail(
        strict=True,
        reason="Bug (modularui.py:441-445): _position_offset_locked assumes contiguous +1 indices; gaps misalign cluster lookup.",
    )
    def test_non_contiguous_position_indices_align_clusters(self, ui: ModularUI):
        ui.add_position_step("fc1", np.zeros((2, 2)), index=0)
        ui.add_position_step("fc1", np.ones((2, 2)), index=5)
        ui.add_cluster("fc1", 2, np.array([0, 1]), 2, 0.5, np.zeros((2, 2)))
        # relative index 0 corresponds to absolute index 0 (before cluster at 2), so should be None
        assert ui.get_cluster_for_index("fc1", 0) is None

    @pytest.mark.xfail(
        strict=True,
        reason="Bug (components.py:70): VarsDisplay.render crashes with TypeError on nested dict with mixed key types.",
    )
    def test_vars_display_handles_mixed_key_types(self):
        vd = VarsDisplay()
        out = vd.render({"mixed": {1: "one", "a": "alpha"}})
        assert isinstance(out, html.Div)

    @pytest.mark.xfail(
        strict=True,
        reason="Bug (modularui.py:232-234): get_model_graph returns internal mutable dict reference instead of a copy.",
    )
    def test_get_model_graph_returns_defensive_copy(self, ui: ModularUI):
        ui.set_model_graph({"nodes": {"fc1": {}}, "edges": []})
        g = ui.get_model_graph()
        g["nodes"]["injected"] = {}
        assert "injected" not in ui.get_model_graph()["nodes"]

    @pytest.mark.xfail(
        strict=True,
        reason="Bug (docks.py:95,113): DashboardLayout silently drops top_panels beyond TOP_AREAS (3).",
    )
    def test_dashboard_layout_renders_more_than_three_top_panels(self):
        tops = [Panel(f"t{i}", f"T{i}", Title(str(i))) for i in range(4)]
        layout = DashboardLayout(tops, Panel("l", "L", Title("L")), Panel("m", "M", Title("M")))
        grid = layout.render()
        assert len(grid.children) == 6
