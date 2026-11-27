from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Any

import pytest
from ax.core.trial_status import TrialStatus

from positronixlab.dataprovider import DataConfig
from positronixlab.deeplearning import TrainConfig
from positronixlab.metamodel import MetaModelConfig
from positronixlab.modularspace import LayerSpaceConfig
from positronixlab.nas import (
    Experiment,
    TrialConfig,
    TrialPipeline,
    TrialResult,
    is_oom_error,
)
from positronixlab.pipeline import PipelineOptions
from positronixlab.template import TemplateBase


@dataclass
class ComposedTrialConfig(MetaModelConfig, LayerSpaceConfig, TrainConfig, DataConfig, TrialConfig):
    exp_metric_modularity: bool = True


def test_config_hierarchy_and_composable_mixins(tmp_path):
    cfg = ComposedTrialConfig.from_trial(
        params={
            "learning_rate": 0.002,
            "modular_space_size": 8,
            "model_type": "cnn",
            "cnn_kernel_size": 5,
            "unknown_key": 123,
        },
        fixed={"dataset_name": "mnist", "train_epochs": 50},
        experiment_name="exp1",
        trial_index=3,
        trial_dir=str(tmp_path / "trial_003"),
        gpu_id=1,
    )
    assert cfg.name == "trial_003"
    assert cfg.experiment_name == "exp1"
    assert cfg.trial_index == 3
    assert cfg.gpu_id == 1
    assert cfg.device == "cuda:1"
    assert cfg.learning_rate == pytest.approx(0.002)
    assert cfg.modular_space_size == 8
    assert cfg.model_type == "cnn"
    assert cfg.cnn_kernel_size == 5
    assert cfg.dataset_name == "mnist"
    assert cfg.train_epochs == 50

    # Immutable update for post-trial effective overrides does not mutate original
    orig_sig = cfg.signature()
    eff = cfg.with_updates(train_epochs=27, ignored_field="x")
    assert cfg.train_epochs == 50
    assert cfg.signature() == orig_sig
    assert eff.train_epochs == 27
    assert eff.signature() != orig_sig

    # Round-trip save/load
    path = str(tmp_path / "cfg.yaml")
    cfg.save(path)
    loaded = ComposedTrialConfig.load(path)
    assert loaded.signature() == cfg.signature()

    # PipelineOptions extends ManagerOptions and is separate from PipelineConfig
    opts = PipelineOptions(work_dir=str(tmp_path), display_logs=False, terminate_on_error=False)
    assert "terminate_on_error" not in opts.to_manager_kwargs()
    assert opts.to_manager_kwargs()["display_logs"] is False
    assert opts.with_updates(log_dir="custom_logs").log_dir == "custom_logs"


def test_template_ax13_hierarchical_and_merge(tmp_path):
    t_model = TemplateBase("model")
    t_model.add_choice(
        "model_type",
        ["fc", "cnn"],
        "str",
        is_ordered=False,
        dependent_parameters={"cnn": ["cnn_kernel_size"]},
    )
    t_model.add_choice("cnn_kernel_size", [3, 5], "int", is_ordered=True)
    t_model.add_range("learning_rate", [1e-4, 1e-2], "float", scaling="log")
    t_model.add_fixed("dataset_name", "mnist")

    t_space = TemplateBase("space")
    t_space.add_range("modular_space_loss_on", [0.0, 1.0], "float")
    t_space.add_range("modular_space_loss_off", [0.0, 1.0], "float")
    t_space.add_constraint("modular_space_loss_on + modular_space_loss_off <= 1.5")
    t_space.add_derived("modular_space_loss_sum", "modular_space_loss_on + modular_space_loss_off", "float")

    t_model.merge(t_space)
    configs = t_model.to_parameter_configs()
    names = {c.name for c in configs}
    assert {
        "model_type",
        "cnn_kernel_size",
        "learning_rate",
        "modular_space_loss_on",
        "modular_space_loss_off",
        "modular_space_loss_sum",
    } == names
    assert t_model.get_fixed() == {"dataset_name": "mnist"}
    assert t_model.get_constraints() == ["modular_space_loss_on + modular_space_loss_off <= 1.5"]

    # Overriding a tunable param with add_fixed removes it from params
    t_model.add_fixed("learning_rate", 1e-3)
    assert "learning_rate" not in t_model.params
    assert t_model.fixed["learning_rate"] == 1e-3

    # Save/load round-trip
    saved = t_model.save(str(tmp_path / "tpl.yaml"))
    reloaded = TemplateBase.load(saved)
    assert reloaded.to_dict() == t_model.to_dict()


def test_experiment_sqlite_resume_and_effective_overrides(tmp_path):
    tpl = TemplateBase("hpo")
    tpl.add_choice(
        "model_type",
        ["fc", "cnn"],
        "str",
        is_ordered=False,
        dependent_parameters={"cnn": ["cnn_kernel_size"]},
    )
    tpl.add_choice("cnn_kernel_size", [3, 5], "int", is_ordered=True)
    tpl.add_range("learning_rate", [1e-4, 1e-2], "float", scaling="log")
    tpl.add_fixed("train_epochs", 100)

    def eval_fn(cfg: ComposedTrialConfig) -> TrialResult:
        score = 0.8 + (0.05 if cfg.model_type == "cnn" else 0.0) - abs(cfg.learning_rate - 1e-3)
        return TrialResult(
            status="completed",
            metrics={"val_acc": float(score), "epochs_trained": 44.0},
            metadata={"best_epoch": 44},
            effective_overrides={"train_epochs": 44},
        )

    exp1 = Experiment(
        name="resume_test",
        template=tpl,
        objective="val_acc",
        eval_fn=eval_fn,
        config_cls=ComposedTrialConfig,
        output_dir=str(tmp_path),
        initialization_budget=3,
        random_seed=42,
    )
    exp1.run(n_trials=3)
    assert len(exp1.client._experiment.trials) == 3

    # Verify folder layout & immutable config vs effective_config
    t0_dir = tmp_path / "resume_test" / "trials" / "trial_000"
    assert (t0_dir / "config.yaml").exists()
    assert (t0_dir / "effective_config.yaml").exists()
    assert (t0_dir / "result.json").exists()
    assert ComposedTrialConfig.load(str(t0_dir / "config.yaml")).train_epochs == 100
    assert ComposedTrialConfig.load(str(t0_dir / "effective_config.yaml")).train_epochs == 44

    # Resume from SQLite to 5 trials
    exp2 = Experiment(
        name="resume_test",
        template=tpl,
        objective="val_acc",
        eval_fn=eval_fn,
        config_cls=ComposedTrialConfig,
        output_dir=str(tmp_path),
        initialization_budget=3,
        random_seed=42,
    )
    best = exp2.run(n_trials=5)
    assert len(exp2.client._experiment.trials) == 5
    assert len(best) == 1
    assert (tmp_path / "resume_test" / "summary.csv").exists()
    assert (tmp_path / "resume_test" / "best" / "best_summary.json").exists()


def test_experiment_moo_parallel_and_warm_start(tmp_path):
    tpl = TemplateBase("moo")
    tpl.add_range("modular_space_lr", [1e-4, 1e-1], "float", scaling="log")
    tpl.add_range("modular_space_size", [3, 12], "int")

    def eval_fn(cfg: ComposedTrialConfig) -> dict[str, float]:
        acc = 0.6 + 0.03 * cfg.modular_space_size
        size = float(cfg.modular_space_size)
        return {"val_acc": acc, "model_size": size}

    exp = Experiment(
        name="moo_exp",
        template=tpl,
        objective="val_acc, -model_size",
        outcome_constraints=["val_acc >= 0.5", "model_size <= 15.0"],
        eval_fn=eval_fn,
        config_cls=ComposedTrialConfig,
        output_dir=str(tmp_path),
        initialization_budget=4,
        parallel_init=2,
        parallel_bo=2,
        random_seed=7,
    )
    frontier = exp.run(n_trials=6)
    assert len(exp.client._experiment.trials) == 6
    assert len(frontier) >= 1
    with open(tmp_path / "moo_exp" / "best" / "best_summary.json", encoding="utf-8") as f:
        summary = json.load(f)
    assert len(summary["best_trials"]) == len(frontier)

    # Warm-start a second independent experiment from moo_exp's SQLite DB
    exp_warm = Experiment(
        name="moo_warm",
        template=tpl,
        objective="val_acc, -model_size",
        outcome_constraints=["val_acc >= 0.5", "model_size <= 15.0"],
        eval_fn=eval_fn,
        config_cls=ComposedTrialConfig,
        output_dir=str(tmp_path),
        initialization_budget=4,
        random_seed=9,
    )
    attached = exp_warm.warm_start_from(exp.db_path, "moo_exp")
    assert attached == 6
    assert len(exp_warm.client._experiment.trials) == 6


def test_experiment_oom_parallel_retry_and_abandon(tmp_path):
    tpl = TemplateBase("oom_test")
    tpl.add_range("modular_space_size", [4, 16], "int")

    attempts: dict[int, int] = {}
    lock = threading.Lock()

    def eval_fn(cfg: ComposedTrialConfig) -> dict[str, float]:
        assert cfg.trial_index is not None
        with lock:
            attempts[cfg.trial_index] = attempts.get(cfg.trial_index, 0) + 1
            count = attempts[cfg.trial_index]

        # Trial 0: OOMs once during parallel batch, succeeds on solo retry
        if cfg.trial_index == 0 and count == 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        # Trial 1: Always OOMs -> should be abandoned after exhausting solo retries
        if cfg.trial_index == 1:
            raise RuntimeError("CUDA out of memory. Always too big")
        return {"val_acc": 0.9}

    exp = Experiment(
        name="oom_exp",
        template=tpl,
        objective="val_acc",
        eval_fn=eval_fn,
        config_cls=ComposedTrialConfig,
        output_dir=str(tmp_path),
        initialization_budget=2,
        parallel_init=2,
        oom_retries=1,
        oom_backoff_sec=0.0,
        oom_final_status="abandoned",
        random_seed=1,
    )
    exp.run(n_trials=2)

    trials = exp.client._experiment.trials
    assert trials[0].status == TrialStatus.COMPLETED
    assert attempts[0] == 2  # 1 parallel OOM + 1 solo retry succeeded
    assert trials[1].status == TrialStatus.ABANDONED
    assert attempts[1] == 3  # 1 parallel OOM + 2 solo attempts (1 + oom_retries)
    assert is_oom_error("torch.cuda.OutOfMemoryError: CUDA out of memory")


class _MockTrialPipeline(TrialPipeline):
    """Uses a real parallelazy PluginManager without spawning heavy plugins."""

    def setup(self) -> None:
        self.plugin_manager.env_set("val_acc", 0.91)
        self.plugin_manager.env_set("best_epoch", 18)

    def is_complete(self) -> bool:
        return True

    def extract_metrics(self, env: dict[str, Any]) -> dict[str, float]:
        return {"val_acc": float(env["val_acc"])}

    def extract_effective_overrides(self, env: dict[str, Any]) -> dict[str, Any]:
        return {"train_epochs": int(env["best_epoch"])}


def test_trial_pipeline_with_parallelazy_dirs(tmp_path):
    tpl = TemplateBase("pipe_test")
    tpl.add_range("learning_rate", [1e-4, 1e-2], "float", scaling="log")
    tpl.add_fixed("train_epochs", 50)

    exp = Experiment(
        name="pipe_exp",
        template=tpl,
        objective="val_acc",
        trial_pipeline_cls=_MockTrialPipeline,
        config_cls=ComposedTrialConfig,
        pipeline_options=PipelineOptions(display_logs=False, write_logs=True),
        output_dir=str(tmp_path),
        initialization_budget=2,
        random_seed=0,
    )
    best = exp.run(n_trials=2)
    assert len(best) == 1

    t0_dir = tmp_path / "pipe_exp" / "trials" / "trial_000"
    assert (t0_dir / "logs").is_dir()
    assert (t0_dir / "config.yaml").exists()
    assert (t0_dir / "effective_config.yaml").exists()
    assert ComposedTrialConfig.load(str(t0_dir / "effective_config.yaml")).train_epochs == 18


# ---------------------------------------------------------------------------
# Unit tests: TrialConfig, TrialResult, is_oom_error, TrialPipeline edge cases
# ---------------------------------------------------------------------------


def test_trial_config_from_trial_precedence_and_defaults():
    # params override fixed; explicit name overrides trial_index default
    cfg = TrialConfig.from_trial(
        params={"seed": 99},
        fixed={"seed": 1, "device": "cpu"},
        experiment_name="exp_a",
        trial_index=7,
        name="custom_name",
    )
    assert cfg.seed == 99
    assert cfg.device == "cpu"
    assert cfg.name == "custom_name"
    assert cfg.experiment_name == "exp_a"
    assert cfg.trial_index == 7
    assert cfg.gpu_id is None

    # No trial_index and no name keeps BaseConfig default name
    cfg_default = TrialConfig.from_trial(params={})
    assert cfg_default.name == "default_config"
    assert cfg_default.device is None


def test_is_oom_error_markers():
    assert is_oom_error("RuntimeError: CUDA out of memory.")
    assert is_oom_error("cudaErrorMemoryAllocation: failed")
    assert is_oom_error("CUBLAS_STATUS_ALLOC_FAILED")
    assert is_oom_error("cudnn_status_alloc_failed")
    assert is_oom_error("torch.cuda.OutOfMemoryError")
    assert not is_oom_error("ValueError: invalid learning rate")
    assert not is_oom_error("")


def test_trial_result_properties_and_save(tmp_path):
    res_ok = TrialResult(
        status="completed",
        metrics={"val_acc": (0.92, 0.01), "loss": 0.15},
        duration_sec=1.25,
        metadata={"best_epoch": 12, "path": tmp_path},
        effective_overrides={"train_epochs": 12},
    )
    assert res_ok.is_completed is True
    assert res_ok.is_oom is False

    out_path = tmp_path / "nested" / "result.json"
    res_ok.save(str(out_path))
    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded["status"] == "completed"
    assert loaded["metrics"]["val_acc"] == [0.92, 0.01]
    assert loaded["metrics"]["loss"] == pytest.approx(0.15)
    assert loaded["metadata"]["best_epoch"] == 12

    res_oom = TrialResult(status="oom", reason="CUDA out of memory")
    assert res_oom.is_oom is True
    assert res_oom.is_completed is False


class _SimplePipeline(TrialPipeline):
    def __init__(self, config, options=None, *, complete=True, fail_exc=None):
        super().__init__(config=config, options=options)
        self._complete = complete
        self._fail_exc = fail_exc

    def setup(self) -> None:
        if self._fail_exc is not None:
            raise self._fail_exc

    def is_complete(self) -> bool:
        return self._complete


def test_trial_pipeline_extract_metrics_and_lifecycle(tmp_path):
    cfg = TrialConfig(trial_dir=str(tmp_path / "t0"))
    pipe = _SimplePipeline(cfg, PipelineOptions(display_logs=False, write_logs=False))
    try:
        # Auto-binds work_dir to config.trial_dir when options.work_dir is None
        assert pipe.options.work_dir == str(tmp_path / "t0")

        # Default extract_metrics handles scalars, 2-tuples/lists, ignores invalid/non-dict
        assert pipe.extract_metrics({}) == {}
        assert pipe.extract_metrics({"metrics": "not_a_dict"}) == {}
        parsed = pipe.extract_metrics(
            {
                "metrics": {
                    "acc": 0.9,
                    "epochs": 10,
                    "acc_sem": (0.91, 0.02),
                    "loss_sem": [0.2, 0.01],
                    "note": "ignored",
                    "triple": (1, 2, 3),
                }
            }
        )
        assert parsed == {
            "acc": 0.9,
            "epochs": 10.0,
            "acc_sem": (0.91, 0.02),
            "loss_sem": (0.2, 0.01),
        }
    finally:
        pipe.shutdown()

    # Explicit work_dir in PipelineOptions is preserved
    custom_work = str(tmp_path / "custom_work")
    pipe_custom = _SimplePipeline(
        cfg, PipelineOptions(work_dir=custom_work, display_logs=False, write_logs=False)
    )
    try:
        assert pipe_custom.options.work_dir == custom_work
    finally:
        pipe_custom.shutdown()

    # Timeout classification
    pipe_timeout = _SimplePipeline(
        cfg, PipelineOptions(display_logs=False, write_logs=False), complete=False
    )
    res_to = pipe_timeout.execute(timeout_sec=0.03, poll_interval=0.01)
    assert res_to.status == "timeout"
    assert res_to.reason and "timeout" in res_to.reason.lower()

    # Non-OOM exception classification
    pipe_err = _SimplePipeline(
        cfg,
        PipelineOptions(display_logs=False, write_logs=False),
        fail_exc=ValueError("bad hyperparam"),
    )
    res_err = pipe_err.execute()
    assert res_err.status == "error"
    assert "bad hyperparam" in (res_err.reason or "")

    # OOM exception classification
    pipe_oom = _SimplePipeline(
        cfg,
        PipelineOptions(display_logs=False, write_logs=False),
        fail_exc=RuntimeError("CUDA out of memory"),
    )
    res_oom = pipe_oom.execute()
    assert res_oom.is_oom


def test_experiment_validation_and_properties(tmp_path):
    tpl = TemplateBase("val_test")
    tpl.add_range("x", [0.0, 1.0], "float")

    with pytest.raises(ValueError, match="trial_pipeline_cls"):
        Experiment(name="bad1", template=tpl, objective="y", output_dir=str(tmp_path))

    with pytest.raises(ValueError, match="oom_final_status"):
        Experiment(
            name="bad2",
            template=tpl,
            objective="y",
            eval_fn=lambda c: {"y": 1.0},
            oom_final_status="invalid",
            output_dir=str(tmp_path),
        )

    exp_so = Experiment(
        name="so",
        template=tpl,
        objective="y",
        eval_fn=lambda c: {"y": 1.0},
        output_dir=str(tmp_path),
    )
    assert exp_so.is_multi_objective is False

    exp_mo = Experiment(
        name="mo",
        template=tpl,
        objective="y, -z",
        eval_fn=lambda c: {"y": 1.0, "z": 2.0},
        output_dir=str(tmp_path),
    )
    assert exp_mo.is_multi_objective is True


def test_experiment_reconcile_running_abandon(tmp_path):
    tpl = TemplateBase("reconcile_abandon")
    tpl.add_range("learning_rate", [1e-4, 1e-2], "float", scaling="log")

    exp1 = Experiment(
        name="crash_exp",
        template=tpl,
        objective="val_acc",
        eval_fn=lambda c: {"val_acc": 0.9},
        config_cls=ComposedTrialConfig,
        output_dir=str(tmp_path),
        initialization_budget=2,
        random_seed=0,
    )
    # Simulate a process crash leaving trial 0 in RUNNING state
    trials = exp1.client.get_next_trials(max_trials=1)
    assert 0 in trials
    assert exp1.client._experiment.trials[0].status == TrialStatus.RUNNING

    exp2 = Experiment(
        name="crash_exp",
        template=tpl,
        objective="val_acc",
        eval_fn=lambda c: {"val_acc": 0.9},
        config_cls=ComposedTrialConfig,
        output_dir=str(tmp_path),
        reconcile_running="abandon",
        initialization_budget=2,
        random_seed=0,
    )
    assert exp2.client._experiment.trials[0].status == TrialStatus.ABANDONED


def test_experiment_warm_start_filter_fn_and_fixed_parameters(tmp_path):
    tpl = TemplateBase("ws_filter")
    tpl.add_choice("model_type", ["fc", "cnn"], "str", is_ordered=False)
    tpl.add_range("learning_rate", [1e-4, 1e-2], "float", scaling="log")

    exp_src = Experiment(
        name="src_exp",
        template=tpl,
        objective="val_acc",
        eval_fn=lambda c: {"val_acc": (0.85, 0.02)},
        config_cls=ComposedTrialConfig,
        output_dir=str(tmp_path),
        initialization_budget=2,
        random_seed=1,
    )
    # Force fixed_parameters={"model_type": "cnn"} for 2 trials
    exp_src.run(n_trials=2, fixed_parameters={"model_type": "cnn"})
    for t in exp_src.client._experiment.trials.values():
        assert t.arm.parameters["model_type"] == "cnn"

    exp_dst = Experiment(
        name="dst_exp",
        template=tpl,
        objective="val_acc",
        eval_fn=lambda c: {"val_acc": 0.9},
        config_cls=ComposedTrialConfig,
        output_dir=str(tmp_path),
        initialization_budget=2,
        random_seed=2,
    )
    # Filter excluding cnn should attach 0; filter matching cnn should attach 2
    assert exp_dst.warm_start_from(exp_src.db_path, "src_exp", filter_fn=lambda p: p["model_type"] == "fc") == 0
    assert exp_dst.warm_start_from(exp_src.db_path, "src_exp", filter_fn=lambda p: p["model_type"] == "cnn") == 2


def test_experiment_solo_oom_no_retries(tmp_path):
    tpl = TemplateBase("oom_zero")
    tpl.add_range("modular_space_size", [4, 8], "int")
    calls = 0

    def eval_fn(_cfg):
        nonlocal calls
        calls += 1
        raise RuntimeError("CUDA out of memory")

    exp = Experiment(
        name="oom_zero_exp",
        template=tpl,
        objective="val_acc",
        eval_fn=eval_fn,
        config_cls=ComposedTrialConfig,
        output_dir=str(tmp_path),
        initialization_budget=1,
        parallel_init=1,
        oom_retries=0,
        oom_final_status="abandoned",
        random_seed=0,
    )
    exp.run(n_trials=1)
    assert calls == 1
    assert exp.client._experiment.trials[0].status == TrialStatus.ABANDONED


# ---------------------------------------------------------------------------
# Confirmed bug regression tests (xfail strict=True until library is fixed)
# ---------------------------------------------------------------------------


def test_bug_mark_trial_failed_on_error(tmp_path):
    tpl = TemplateBase("err_bug")
    tpl.add_range("learning_rate", [1e-4, 1e-2], "float", scaling="log")

    exp = Experiment(
        name="err_bug_exp",
        template=tpl,
        objective="val_acc",
        eval_fn=lambda _c: (_ for _ in ()).throw(ValueError("boom")),
        config_cls=ComposedTrialConfig,
        output_dir=str(tmp_path),
        initialization_budget=1,
        random_seed=0,
    )
    trials = exp.client.get_next_trials(max_trials=1)
    t_idx, params = next(iter(trials.items()))
    cfg = exp._build_trial_config(t_idx, params)
    res = exp._run_single_trial(cfg)
    assert res.status == "error"
    exp._record_trial_result(t_idx, cfg, res)
    assert exp.client._experiment.trials[t_idx].status == TrialStatus.FAILED


def test_bug_mark_trial_failed_on_oom_failed_status(tmp_path):
    tpl = TemplateBase("oom_fail_bug")
    tpl.add_range("learning_rate", [1e-4, 1e-2], "float", scaling="log")

    exp = Experiment(
        name="oom_fail_exp",
        template=tpl,
        objective="val_acc",
        eval_fn=lambda _c: (_ for _ in ()).throw(RuntimeError("CUDA out of memory")),
        config_cls=ComposedTrialConfig,
        output_dir=str(tmp_path),
        initialization_budget=1,
        oom_retries=0,
        oom_final_status="failed",
        random_seed=0,
    )
    trials = exp.client.get_next_trials(max_trials=1)
    t_idx, params = next(iter(trials.items()))
    cfg = exp._build_trial_config(t_idx, params)
    res = exp._run_single_trial(cfg)
    assert res.is_oom
    exp._record_trial_result(t_idx, cfg, res)
    assert exp.client._experiment.trials[t_idx].status == TrialStatus.FAILED


def test_bug_reconcile_running_fail_on_resume(tmp_path):
    tpl = TemplateBase("reconcile_fail_bug")
    tpl.add_range("learning_rate", [1e-4, 1e-2], "float", scaling="log")

    exp1 = Experiment(
        name="crash_fail_exp",
        template=tpl,
        objective="val_acc",
        eval_fn=lambda _c: {"val_acc": 0.9},
        config_cls=ComposedTrialConfig,
        output_dir=str(tmp_path),
        initialization_budget=2,
        random_seed=0,
    )
    exp1.client.get_next_trials(max_trials=1)

    exp2 = Experiment(
        name="crash_fail_exp",
        template=tpl,
        objective="val_acc",
        eval_fn=lambda _c: {"val_acc": 0.9},
        config_cls=ComposedTrialConfig,
        output_dir=str(tmp_path),
        reconcile_running="fail",
        initialization_budget=2,
        random_seed=0,
    )
    assert exp2.client._experiment.trials[0].status == TrialStatus.FAILED


def test_bug_solo_oom_retry_off_by_one(tmp_path):
    tpl = TemplateBase("oom_solo_bug")
    tpl.add_range("modular_space_size", [4, 8], "int")
    calls = 0

    def eval_fn(_cfg):
        nonlocal calls
        calls += 1
        raise RuntimeError("CUDA out of memory")

    exp = Experiment(
        name="oom_solo_bug_exp",
        template=tpl,
        objective="val_acc",
        eval_fn=eval_fn,
        config_cls=ComposedTrialConfig,
        output_dir=str(tmp_path),
        initialization_budget=1,
        parallel_init=1,
        oom_retries=1,
        oom_backoff_sec=0.0,
        oom_final_status="abandoned",
        random_seed=0,
    )
    exp.run(n_trials=1)
    assert calls == 2  # Expected 1 initial + 1 retry


def test_bug_trial_config_gpu_id_with_none_device():
    cfg = TrialConfig.from_trial(params={}, fixed={"device": None}, gpu_id=1)
    assert cfg.device == "cuda:1"


def test_bug_extract_metrics_bool_and_none_sem():
    pipe = _SimplePipeline(TrialConfig(), PipelineOptions(display_logs=False, write_logs=False))
    try:
        parsed = pipe.extract_metrics({"metrics": {"flag": True, "acc": (0.9, None)}})
        assert "flag" not in parsed
        assert parsed["acc"] == 0.9
    finally:
        pipe.shutdown()


def test_bug_export_best_leaves_stale_yaml(tmp_path):
    tpl = TemplateBase("stale_best")
    tpl.add_range("learning_rate", [1e-4, 1e-2], "float", scaling="log")
    scores = iter([0.5, 0.95])

    exp = Experiment(
        name="stale_best_exp",
        template=tpl,
        objective="val_acc",
        eval_fn=lambda _c: {"val_acc": next(scores)},
        config_cls=ComposedTrialConfig,
        output_dir=str(tmp_path),
        initialization_budget=2,
        random_seed=0,
    )
    exp.run(n_trials=1)
    assert (tmp_path / "stale_best_exp" / "best" / "trial_000.yaml").exists()

    exp.run(n_trials=2)
    yaml_files = sorted(p.name for p in (tmp_path / "stale_best_exp" / "best").glob("trial_*.yaml"))
    assert yaml_files == ["trial_001.yaml"]


def test_bug_warm_start_error_leaks_global_db_session(tmp_path):
    from ax.storage.sqa_store.db import get_engine

    tpl = TemplateBase("ws_leak")
    tpl.add_range("learning_rate", [1e-4, 1e-2], "float", scaling="log")

    exp = Experiment(
        name="main_exp",
        template=tpl,
        objective="val_acc",
        eval_fn=lambda _c: {"val_acc": 0.9},
        output_dir=str(tmp_path),
        initialization_budget=1,
        random_seed=0,
    )
    bogus_db = str(tmp_path / "other.db")
    with pytest.raises(Exception):
        exp.warm_start_from(bogus_db, "nonexistent_experiment")

    # Global engine should still be bound to exp.db_path, not bogus_db
    assert exp.db_path in str(get_engine().url)
