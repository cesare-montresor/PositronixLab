"""Unit tests for `positronixlab.template`."""

from __future__ import annotations

import copy
import pytest
from ax.api.client import Client
from ax.api.configs import (
    ChoiceParameterConfig,
    DerivedParameterConfig,
    RangeParameterConfig,
)

from positronixlab.template import (
    TemplateBase,
    TemplateCovertype,
    TemplateFMnistCNN,
    TemplateFMnistFC,
    TemplateFactory,
    TemplateHiggs,
    TemplateMnistCNN,
    TemplateMnistFC,
)
from positronixlab.template.template_base import (
    ParameterType,
    TemplateAtlas,
    TemplateCNN,
    TemplateDL,
    TemplateFC,
)
from positronixlab.template.template_factory import TemplateVariantFactory
from positronixlab.template.template_variant import TemplateVariant, TemplateVariantList


# =============================================================================
# TemplateBase
# =============================================================================
class TestTemplateBase:
    def test_default_and_custom_name(self):
        t_default = TemplateBase()
        assert t_default.name == "templatebase"
        assert t_default.params == {}
        assert t_default.fixed == {}
        assert t_default.constraints == []

        t_named = TemplateBase("my_exp")
        assert t_named.name == "my_exp"

    def test_add_range_linear_log_and_step_size(self):
        t = (
            TemplateBase("ranges")
            .add_range("lr", [1e-4, 1e-2], ParameterType.FLOAT, scaling="log")
            .add_range("lr2", [1e-4, 1e-2], "float", log_scale=True)
            .add_range("hidden", [64, 512], ParameterType.INT, step_size=64)
        )
        assert t.params["lr"]["scaling"] == "log"
        assert t.params["lr2"]["scaling"] == "log"
        assert t.params["hidden"]["scaling"] == "linear"
        assert t.params["hidden"]["step_size"] == 64

        configs = {c.name: c for c in t.to_parameter_configs()}
        assert isinstance(configs["lr"], RangeParameterConfig)
        assert configs["lr"].scaling == "log"
        assert configs["lr"].bounds == (1e-4, 1e-2)
        assert configs["hidden"].parameter_type == "int"
        assert configs["hidden"].step_size == 64

    def test_add_choice_ordering_and_dependents(self):
        t = (
            TemplateBase("choices")
            .add_choice("flag", [True, False], ParameterType.BOOL)
            .add_choice("act", ["relu", "leaky_relu", "elu"], ParameterType.STRING)
            .add_choice("width", [32, 64, 128], ParameterType.INT)
            .add_choice(
                "arch",
                ["fc", "cnn"],
                "str",
                dependent_parameters={"cnn": ["kernel_size"]},
            )
            .add_choice("kernel_size", [3, 5], "int")
        )
        assert t.params["flag"]["is_ordered"] is True
        assert t.params["act"]["is_ordered"] is False
        assert t.params["width"]["is_ordered"] is True
        assert t.params["arch"]["dependent_parameters"] == {"cnn": ["kernel_size"]}

        configs = {c.name: c for c in t.to_parameter_configs()}
        assert isinstance(configs["flag"], ChoiceParameterConfig)
        assert configs["act"].is_ordered is False
        assert configs["arch"].dependent_parameters == {"cnn": ["kernel_size"]}

    @pytest.mark.xfail(
        strict=True,
        reason="Bug #1: ChoiceParameterConfig in Ax 1.3 does not accept log_scale kwarg",
    )
    def test_add_choice_with_log_scale_converts_to_ax(self):
        t = TemplateBase("choice_log").add_choice(
            "lr_grid", [1e-4, 1e-3, 1e-2], ParameterType.FLOAT, log_scale=True
        )
        configs = t.to_parameter_configs()
        assert len(configs) == 1

    def test_add_derived_and_constraints(self):
        t = (
            TemplateBase("derived")
            .add_range("x1", [1, 10], "int")
            .add_range("x2", [1, 10], "int")
            .add_derived("x_sum", "x1 + x2", "int")
            .add_constraint("x1 <= x2")
            .add_constraint("x1 <= x2")  # duplicate ignored
        )
        assert t.get_constraints() == ["x1 <= x2"]
        configs = {c.name: c for c in t.to_parameter_configs()}
        assert isinstance(configs["x_sum"], DerivedParameterConfig)
        assert configs["x_sum"].expression_str == "x1 + x2"

    def test_fixed_vs_tunable_mutual_exclusion_and_remove(self):
        t = TemplateBase("excl")
        t.add_range("dropout", [0.1, 0.5], "float")
        assert "dropout" in t.params and "dropout" not in t.fixed

        t.add_fixed("dropout", 0.25, "float")
        assert "dropout" not in t.params and t.fixed["dropout"] == 0.25

        t.add_choice("dropout", [0.1, 0.2, 0.3], "float")
        assert "dropout" in t.params and "dropout" not in t.fixed

        assert t.get_fixed() == {}
        t.add_fixed("seed", 42).add_fixed("device", "cpu")
        assert t.get_fixed(["seed"]) == {"seed": 42}

        t.remove("dropout").remove("seed").remove("nonexistent")
        assert "dropout" not in t.params
        assert "seed" not in t.fixed
        assert t.get_fixed() == {"device": "cpu"}

    def test_merge_overwrite_and_no_overwrite(self):
        a = (
            TemplateBase("a")
            .add_range("lr", [1e-4, 1e-2], "float", scaling="log")
            .add_fixed("dataset_name", "mnist")
            .add_constraint("x <= y")
        )
        b = (
            TemplateBase("b")
            .add_fixed("lr", 0.001)  # overrides tunable lr when overwrite=True
            .add_range("dataset_name", [1, 2], "int")
            .add_range("batch_size", [32, 128], "int")
            .add_constraint("y <= z")
        )

        # overwrite=False keeps a's existing entries across both params and fixed
        a_no_ow = TemplateBase.from_dict(a.to_dict())
        a_no_ow.merge(b, overwrite=False)
        assert "lr" in a_no_ow.params and "lr" not in a_no_ow.fixed
        assert a_no_ow.fixed["dataset_name"] == "mnist"
        assert "batch_size" in a_no_ow.params
        assert a_no_ow.get_constraints() == ["x <= y", "y <= z"]

        # overwrite=True replaces existing entries and cleans opposite dict
        a_ow = TemplateBase.from_dict(a.to_dict())
        a_ow.merge(b, overwrite=True)
        assert "lr" not in a_ow.params and a_ow.fixed["lr"] == 0.001
        assert "dataset_name" not in a_ow.fixed and "dataset_name" in a_ow.params

    def test_dict_and_yaml_round_trip(self, tmp_path):
        t = (
            TemplateBase("rt")
            .add_fixed("dataset_name", "mnist")
            .add_range("lr", [1e-4, 1e-2], "float", scaling="log")
            .add_choice("act", ["relu", "leaky_relu"], "str")
            .add_derived("x2", "2 * lr", "float")
            .add_constraint("lr <= 0.01")
        )
        d = t.to_dict()
        t_from_dict = TemplateBase.from_dict(d)
        assert t_from_dict.to_dict() == d

        yaml_path = tmp_path / "subdir" / "rt.yaml"
        returned_path = t.save(str(yaml_path))
        assert returned_path == str(yaml_path)
        assert yaml_path.exists()

        t_loaded = TemplateBase.load(str(yaml_path))
        assert t_loaded.to_dict() == d

    @pytest.mark.xfail(
        strict=True,
        reason="Bug #4: save() uses yaml.dump (emits !!python/tuple) while load() uses yaml.safe_load",
    )
    def test_yaml_round_trip_with_tuple_fixed_value(self, tmp_path):
        t = TemplateBase("tuple_rt").add_fixed("shape", (1, 28, 28))
        path = tmp_path / "tuple.yaml"
        t.save(str(path))
        loaded = TemplateBase.load(str(path))
        assert list(loaded.fixed["shape"]) == [1, 28, 28]

    @pytest.mark.xfail(
        strict=True,
        reason="Bug #6: to_parameter_configs() silently ignores unknown parameter types instead of raising ValueError",
    )
    def test_unknown_param_kind_raises(self):
        t = TemplateBase("bad_kind")
        t.params["oops"] = {"name": "oops", "type": "unknown_kind", "parameter_type": "int"}
        with pytest.raises(ValueError):
            t.to_parameter_configs()

    def test_ax_client_integration_with_derived_and_constraints(self):
        t = (
            TemplateBase("ax_demo")
            .add_range("input_hidden_dim", [64, 256], "int", step_size=64)
            .add_range("output_hidden_dim", [32, 128], "int", step_size=32)
            .add_choice("activation", ["relu", "leaky_relu"], "str")
            .add_derived("total_hidden_dim", "input_hidden_dim + output_hidden_dim", "int")
            .add_constraint("output_hidden_dim <= input_hidden_dim")
        )
        client = Client(random_seed=0)
        client.configure_experiment(
            name=t.name,
            parameters=t.to_parameter_configs(),
            parameter_constraints=t.get_constraints(),
        )
        client.configure_optimization(objective="acc")
        trials = client.get_next_trials(max_trials=1)
        assert len(trials) == 1
        params = next(iter(trials.values()))
        assert params["output_hidden_dim"] <= params["input_hidden_dim"]
        assert params["total_hidden_dim"] == params["input_hidden_dim"] + params["output_hidden_dim"]


# =============================================================================
# Template Hierarchy & Concrete Dataset Templates
# =============================================================================
class TestConcreteTemplates:
    def test_hierarchy_inheritance(self):
        dl = TemplateDL()
        atlas = TemplateAtlas()
        fc = TemplateFC()
        cnn = TemplateCNN()

        assert dl.fixed["model_type"] == "fc"
        assert "exp_metric_modularity" not in dl.params
        assert "exp_metric_modularity" in atlas.params
        assert fc.fixed["model_type"] == "fc"
        assert cnn.fixed["model_type"] == "cnn"
        assert "cnn_kernel_size" in cnn.params
        assert "modular_cnn_locality" in cnn.params

    @pytest.mark.parametrize(
        "key,expected_cls,expected_model_type,expected_in,expected_out",
        [
            ("mnist_fc", TemplateMnistFC, "fc", 784, 10),
            ("mnist_cnn", TemplateMnistCNN, "cnn", 28, 10),
            ("fmnist_fc", TemplateFMnistFC, "fc", 784, 10),
            ("fmnist_cnn", TemplateFMnistCNN, "cnn", 28, 10),
            ("covertype", TemplateCovertype, "fc", 54, 7),
            ("higgs", TemplateHiggs, "fc", 28, 2),
        ],
    )
    def test_factory_templates_and_ax_compatibility(
        self, key, expected_cls, expected_model_type, expected_in, expected_out
    ):
        tpl = TemplateFactory.FromName(key, name=f"test_{key}")
        assert isinstance(tpl, expected_cls)
        assert tpl.name == f"test_{key}"
        assert tpl.fixed["model_type"] == expected_model_type
        assert tpl.fixed["input_data_shape"] == expected_in
        assert tpl.fixed["output_data_shape"] == expected_out

        configs = tpl.to_parameter_configs()
        assert len(configs) == len(tpl.params)

        client = Client(random_seed=42)
        client.configure_experiment(
            name=tpl.name,
            parameters=configs,
            parameter_constraints=tpl.get_constraints() or None,
        )

    @pytest.mark.xfail(
        strict=True,
        reason="Bug #3: TemplateFMnistFC/CNN set dataset_name='fmnist' instead of DataProviderFactory's 'fashion_mnist'",
    )
    @pytest.mark.parametrize("cls", [TemplateFMnistFC, TemplateFMnistCNN])
    def test_fmnist_dataset_name_matches_dataprovider_factory(self, cls):
        tpl = cls()
        assert tpl.fixed["dataset_name"] == "fashion_mnist"

    @pytest.mark.xfail(
        strict=True,
        reason="Bug #5: TemplateFMnistCNN forgets to override nas_min_acc=0.85 (inherits 0.95 from TemplateDL)",
    )
    def test_fmnist_cnn_nas_min_acc_matches_fmnist_fc(self):
        assert TemplateFMnistCNN().fixed["nas_min_acc"] == TemplateFMnistFC().fixed["nas_min_acc"]


# =============================================================================
# TemplateFactory
# =============================================================================
class TestTemplateFactory:
    def test_unknown_experiment_raises(self):
        with pytest.raises(ValueError, match="Unknown experiment"):
            TemplateFactory.FromName("does_not_exist")

    def test_add_template_registers_custom_class(self):
        class DummyTemplate(TemplateBase):
            def __init__(self, name=None):
                super().__init__(name)
                self.add_fixed("dataset_name", "dummy")

        key = "_test_dummy_template"
        try:
            TemplateFactory.AddTemplate(key, DummyTemplate)
            tpl = TemplateFactory.FromName(key, name="my_dummy")
            assert isinstance(tpl, DummyTemplate)
            assert tpl.fixed["dataset_name"] == "dummy"
        finally:
            TemplateFactory.EXPERIMENTS.pop(key, None)


# =============================================================================
# TemplateVariant, TemplateVariantList, TemplateVariantFactory
# =============================================================================
class TestTemplateVariants:
    def test_variant_validate_and_serialization(self):
        tpl = TemplateFactory.FromName("mnist_fc")
        v = TemplateVariant(
            name="baseline",
            fixed_features={"exp_metric_modularity": False, "exp_modularizer": False},
            description="No modularity",
        )
        v.validate(tpl)
        assert v.get_fixed_features() == {
            "exp_metric_modularity": False,
            "exp_modularizer": False,
        }
        assert "baseline" in repr(v)

        restored = TemplateVariant.from_dict(copy.deepcopy(v.to_dict()))
        assert restored.name == v.name
        assert restored.fixed_features == v.fixed_features
        assert restored.description == v.description

    def test_variant_validate_rejects_unknown_key(self):
        tpl = TemplateFactory.FromName("mnist_fc")
        v = TemplateVariant("bad", fixed_features={"not_a_param": 1})
        with pytest.raises(ValueError, match="not_a_param"):
            v.validate(tpl)

    def test_variant_matches_trial_parameters_present_keys(self):
        v = TemplateVariant(
            "v_b",
            fixed_features={"exp_metric_modularity": True, "exp_modularizer": False},
        )
        assert v.matches_trial_parameters(
            {"exp_metric_modularity": True, "exp_modularizer": False, "batch_size": 64}
        )
        assert not v.matches_trial_parameters(
            {"exp_metric_modularity": False, "exp_modularizer": False}
        )

    @pytest.mark.xfail(
        strict=True,
        reason="Bug #2: matches_trial_parameters() returns True when fixed keys are missing from parameters",
    )
    def test_variant_matches_trial_parameters_missing_keys_returns_false(self):
        v = TemplateVariant(
            "v_b",
            fixed_features={"exp_metric_modularity": True, "exp_modularizer": False},
        )
        assert not v.matches_trial_parameters({})
        assert not v.matches_trial_parameters({"exp_metric_modularity": True})

    def test_variant_list_and_standard_factory(self):
        tpl = TemplateFactory.FromName("mnist_fc")
        vl = TemplateVariantFactory.create_standard_modularity_variants(base_name="exp")
        assert len(vl) == 4
        assert vl.list_variants() == ["exp_a", "exp_b", "exp_c", "exp_d"]
        vl.validate_all(tpl)

        assert vl.get_variant("exp_d").get_fixed_features() == {
            "exp_metric_modularity": True,
            "exp_modularizer": True,
        }
        assert [v.name for v in vl] == ["exp_a", "exp_b", "exp_c", "exp_d"]
        assert len(list(vl.items())) == 4
        assert len(list(vl.values())) == 4

        with pytest.raises(ValueError, match="already exists"):
            vl.add_variant(TemplateVariant("exp_a"))
        with pytest.raises(KeyError, match="does not exist"):
            vl.get_variant("exp_z")

        restored = TemplateVariantList.from_dict(copy.deepcopy(vl.to_dict()))
        assert restored.list_variants() == vl.list_variants()
        assert restored.to_dict() == vl.to_dict()
