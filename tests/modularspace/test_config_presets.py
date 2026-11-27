import types

import pytest

from positronixlab.modularspace.config import LayerSpaceConfig
from positronixlab.modularspace.presets import (
    PRESETS,
    get_preset,
    get_preset_spec,
    list_presets,
)
from positronixlab.modularspace.space_step import SpaceStepSpec


def test_layerspace_config_defaults_and_roundtrip(tmp_path):
    cfg = LayerSpaceConfig()
    assert cfg.modular_space_size == 6
    assert cfg.modular_space_distance == "cosine"
    assert cfg.modular_space_scoring == "quantile"
    assert cfg.modular_space_weighting == "raw"
    assert cfg.modular_space_contrast == "distance_log"

    updated = cfg.with_updates(modular_space_size=12, unknown_key="ignored")
    assert updated.modular_space_size == 12
    assert cfg.modular_space_size == 6
    assert updated.signature() != cfg.signature()

    path = tmp_path / "layer_cfg.json"
    updated.save(str(path))
    assert LayerSpaceConfig.load(str(path)) == updated


def test_layerspace_config_from_dict_coerces_and_drops_unknown():
    cfg = LayerSpaceConfig.from_dict(
        {
            "modular_space_size": "10",
            "modular_space_lr": "0.05",
            "modular_space_apply_activation": "true",
            "nonexistent_field": 123,
        }
    )
    assert cfg.modular_space_size == 10
    assert cfg.modular_space_lr == pytest.approx(0.05)
    assert cfg.modular_space_apply_activation is True
    assert not hasattr(cfg, "nonexistent_field")


def test_presets_list_and_get_preset_isolation():
    names = list_presets()
    assert names == sorted(PRESETS.keys())
    assert set(names) >= {
        "v1_centers",
        "distance_sigreg_centers",
        "distance_sigreg_positions",
        "softmax_ce",
    }

    p = get_preset("  SOFTMAX_CE ", modular_space_tau=0.25)
    assert p["modular_space_tau"] == pytest.approx(0.25)
    assert PRESETS["softmax_ce"]["modular_space_tau"] == pytest.approx(0.1)


def test_get_preset_unknown_raises_key_error():
    with pytest.raises(KeyError, match="Unknown preset"):
        get_preset("does_not_exist")


@pytest.mark.parametrize("preset_name", list_presets())
def test_get_preset_spec_builds_valid_spec(preset_name):
    spec = get_preset_spec(preset_name)
    assert isinstance(spec, SpaceStepSpec)
    assert spec.to_dict()["space_type"] == "cosine"


def test_space_step_spec_from_dict_and_from_config_variants():
    # Short names + flat modular_space_* keys
    spec = SpaceStepSpec.from_dict(
        {
            "modular_space_type": "euclidean",
            "modular_space_scoring": "top_k",
            "tau": 0.3,
            "ignored_key": "x",
            "arch_reducer": None,
        }
    )
    assert spec.space_type == "euclidean"
    assert spec.score_method == "top_k"
    assert spec.tau == pytest.approx(0.3)

    # Passthrough for existing SpaceStepSpec
    assert SpaceStepSpec.from_config(spec) is spec

    # Plain object without to_dict()
    obj = types.SimpleNamespace(
        modular_space_distance="euclidean",
        modular_space_contrast="distance_linear",
        modular_space_loss_sigreg=0.0,
    )
    spec_from_obj = SpaceStepSpec.from_config(obj)
    assert spec_from_obj.space_type == "euclidean"
    assert spec_from_obj.contrast_method == "distance_linear"
    assert spec_from_obj.loss_sigreg == pytest.approx(0.0)
