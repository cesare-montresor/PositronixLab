from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import pytest

import positronixlab.pipeline as pipeline_pkg
from positronixlab.pipeline import BaseConfig


def _ensure_parallelazy_stub() -> bool:
    """
    Return True if real `parallelazy` is importable.
    Otherwise install a minimal stub for `parallelazy` and `parallelazy.plugin_data`
    so `pipeline_config.py` / `V1` / `V2` can be unit-tested in offline/sandboxed envs.
    """
    try:
        import parallelazy  # noqa: F401
        from parallelazy.plugin_data import ManagerOptions  # noqa: F401
        return True
    except ImportError:
        pass

    @dataclass
    class _StubManagerOptions:
        work_dir: str | None = None
        log_dir: str | None = None
        display_logs: bool = True
        write_logs: bool = True

    pkg = types.ModuleType("parallelazy")
    pkg.PluginManager = object  # type: ignore[attr-defined]
    pkg.is_debugpy_attached = lambda: False  # type: ignore[attr-defined]

    plugin_data = types.ModuleType("parallelazy.plugin_data")
    plugin_data.ManagerOptions = _StubManagerOptions  # type: ignore[attr-defined]
    pkg.plugin_data = plugin_data  # type: ignore[attr-defined]

    sys.modules["parallelazy"] = pkg
    sys.modules["parallelazy.plugin_data"] = plugin_data
    return False


# ---------------------------------------------------------------------------
# Sample composable configs using BaseConfig (zero parallelazy dependency)
# ---------------------------------------------------------------------------

@dataclass
class _DataMixin(BaseConfig):
    dataset_name: str = "fashion_mnist"
    batch_size: int = 32


@dataclass
class _TrainMixin(BaseConfig):
    learning_rate: float = 0.001
    use_amp: bool = False
    tag: str | None = None


@dataclass
class _ComposedConfig(_TrainMixin, _DataMixin):
    seed: int | None = None


# ---------------------------------------------------------------------------
# BaseConfig tests
# ---------------------------------------------------------------------------

def test_base_config_defaults_and_dict_yaml():
    cfg = BaseConfig()
    assert cfg.name == "default_config"
    assert cfg.to_dict() == {"name": "default_config"}
    assert "name: default_config" in cfg.to_yaml()


def test_base_config_signature_is_deterministic_and_sensitive():
    c1 = _ComposedConfig(name="exp", batch_size=64, learning_rate=0.01)
    c2 = _ComposedConfig(name="exp", batch_size=64, learning_rate=0.01)
    c3 = _ComposedConfig(name="exp", batch_size=128, learning_rate=0.01)

    assert len(c1.signature()) == 32
    assert c1.signature() == c2.signature()
    assert c1.signature() != c3.signature()


def test_base_config_with_updates_is_immutable_and_ignores_unknown():
    orig = _ComposedConfig(name="base", batch_size=32, learning_rate=0.001)
    orig_sig = orig.signature()

    updated = orig.with_updates(batch_size=128, unknown_field="ignored")
    assert orig.batch_size == 32
    assert orig.signature() == orig_sig
    assert updated.batch_size == 128
    assert updated.name == "base"
    assert updated.signature() != orig_sig
    assert not hasattr(updated, "unknown_field")


def test_base_config_save_and_load_yaml_and_json(tmp_path):
    cfg = _ComposedConfig(
        name="roundtrip",
        dataset_name="mnist",
        batch_size=64,
        learning_rate=0.005,
        use_amp=True,
        tag="v1",
        seed=42,
    )

    yaml_path = str(tmp_path / "nested" / "cfg.yaml")
    json_path = str(tmp_path / "nested" / "cfg.json")

    cfg.save(yaml_path)
    cfg.save(json_path)

    loaded_yaml = _ComposedConfig.load(yaml_path)
    loaded_json = _ComposedConfig.load(json_path)

    assert loaded_yaml == cfg
    assert loaded_json == cfg
    assert loaded_yaml.signature() == cfg.signature()
    assert loaded_json.signature() == cfg.signature()


def test_base_config_load_empty_yaml_uses_defaults(tmp_path):
    empty_path = tmp_path / "empty.yaml"
    empty_path.write_text("", encoding="utf-8")

    loaded = _ComposedConfig.load(str(empty_path))
    assert loaded == _ComposedConfig()


def test_base_config_from_dict_coercion_and_filtering():
    cfg = _ComposedConfig.from_dict(
        {
            "name": 123,
            "batch_size": "64",
            "learning_rate": "0.0025",
            "use_amp": "yes",
            "tag": None,
            "seed": "7",
            "extra_ignored": "drop_me",
        }
    )
    assert cfg.name == "123"
    assert cfg.batch_size == 64 and isinstance(cfg.batch_size, int)
    assert cfg.learning_rate == pytest.approx(0.0025) and isinstance(cfg.learning_rate, float)
    assert cfg.use_amp is True
    assert cfg.tag is None
    assert cfg.seed == 7 and isinstance(cfg.seed, int)
    assert not hasattr(cfg, "extra_ignored")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("true", True),
        ("TRUE", True),
        (" 1 ", True),
        ("yes", True),
        ("false", False),
        ("0", False),
        ("no", False),
        (1, True),
        (0, False),
        (True, True),
        (False, False),
    ],
)
def test_base_config_from_dict_bool_coercion(raw, expected):
    cfg = _ComposedConfig.from_dict({"use_amp": raw})
    assert cfg.use_amp is expected


def test_lazy_init_unknown_attribute_raises():
    with pytest.raises(AttributeError, match="has no attribute 'DoesNotExist'"):
        _ = pipeline_pkg.DoesNotExist


def test_from_dict_pep604_optional_coercion_bug():
    @dataclass
    class _Pep604Config(BaseConfig):
        count: int | None = None
        modes: tuple[str, ...] | None = None

    cfg = _Pep604Config.from_dict({"count": "42", "modes": ["retain", "zero_ablate"]})
    assert cfg.count == 42 and isinstance(cfg.count, int)
    assert cfg.modes == ("retain", "zero_ablate") and isinstance(cfg.modes, tuple)


def test_from_dict_runtime_union_type_and_yaml_tuple_roundtrip(tmp_path):
    # Explicitly construct a dataclass with runtime types.UnionType (not stringified)
    from dataclasses import make_dataclass

    RuntimeCfg = make_dataclass(
        "RuntimeCfg",
        [
            ("count", int | None, None),
            ("modes", tuple[str, ...] | None, ("retain", "zero_ablate")),
        ],
        bases=(BaseConfig,),
    )
    cfg = RuntimeCfg.from_dict({"count": "7", "modes": ["retain"]})
    assert cfg.count == 7 and isinstance(cfg.count, int)
    assert cfg.modes == ("retain",) and isinstance(cfg.modes, tuple)

    yaml_path = tmp_path / "cfg.yaml"
    cfg.save(yaml_path)
    loaded = RuntimeCfg.load(yaml_path)
    assert loaded.count == 7
    assert loaded.modes == ("retain",) and isinstance(loaded.modes, tuple)


def test_load_non_dict_yaml_root_bug(tmp_path):
    bad_path = tmp_path / "list.yaml"
    bad_path.write_text("- item1\n- item2\n", encoding="utf-8")

    with pytest.raises((ValueError, TypeError)):
        BaseConfig.load(str(bad_path))


# ---------------------------------------------------------------------------
# PipelineOptions, PipelineConfig, PipelineConfigV1, PipelineConfigV2 tests
# ---------------------------------------------------------------------------

def test_pipeline_options_and_config_hierarchy():
    _ensure_parallelazy_stub()
    from positronixlab.pipeline import (
        PipelineConfig,
        PipelineConfigV1,
        PipelineConfigV2,
        PipelineOptions,
    )

    opts = PipelineOptions(
        work_dir="/tmp/work",
        display_logs=False,
        terminate_on_error=False,
        timeout_sec=30.0,
        poll_interval=0.1,
    )
    mgr_kwargs = opts.to_manager_kwargs()
    assert mgr_kwargs["work_dir"] == "/tmp/work"
    assert mgr_kwargs["display_logs"] is False
    assert "terminate_on_error" not in mgr_kwargs
    assert "timeout_sec" not in mgr_kwargs
    assert "poll_interval" not in mgr_kwargs

    opts2 = opts.with_updates(timeout_sec=60.0, nonexistent_flag=True)
    assert opts.timeout_sec == 30.0
    assert opts2.timeout_sec == 60.0
    assert not hasattr(opts2, "nonexistent_flag")

    base = PipelineConfig(name="p", seed=123, device="cpu")
    assert isinstance(base, BaseConfig)
    assert base.seed == 123
    assert base.device == "cpu"

    v1 = PipelineConfigV1.from_dict({"dataset_name": "mnist", "modular_space_size": "8"})
    assert isinstance(v1, PipelineConfig)
    assert v1.dataset_name == "mnist"
    assert v1.modular_space_size == 8
    assert v1.modular_space_loss_alignment == 1.0

    v2 = PipelineConfigV2.from_dict(
        {
            "dataset_name": "cifar10",
            "modular_space_scoring": "zscore",
            "modular_space_apply_activation": "true",
            "modular_space_loss_sigreg": "0.5",
        }
    )
    assert isinstance(v2, PipelineConfig)
    assert v2.dataset_name == "cifar10"
    assert v2.modular_space_scoring == "zscore"
    assert v2.modular_space_apply_activation is True
    assert v2.modular_space_loss_sigreg == pytest.approx(0.5)


def test_pipeline_config_v2_optional_str_coercion_bug():
    _ensure_parallelazy_stub()
    from positronixlab.pipeline import PipelineConfigV2

    v2 = PipelineConfigV2.from_dict({"modular_space_reducer": 123})
    assert v2.modular_space_reducer == "123" and isinstance(v2.modular_space_reducer, str)
