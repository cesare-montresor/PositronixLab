from __future__ import annotations

import logging
import sys
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest


def _ensure_parallelazy_stub() -> bool:
    """Install minimal `parallelazy` stub only when the real package is unavailable."""
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


_ensure_parallelazy_stub()

import positronixlab.pipeline.pipeline as pipeline_mod
from positronixlab.pipeline.pipeline import Pipeline
from positronixlab.pipeline.pipeline_config import PipelineConfig, PipelineOptions

# ---------------------------------------------------------------------------
# Lightweight fake PluginManager & worker status objects
# ---------------------------------------------------------------------------

@dataclass
class _FakeWorkerStatus:
    state: str = "running"
    errors: list[str] = field(default_factory=list)


@dataclass
class _FakePluginStatus:
    process_alive: bool = True
    worker_status: _FakeWorkerStatus | None = field(default_factory=_FakeWorkerStatus)


class _FakePluginManager:
    def __init__(self, **kwargs: Any):
        self.init_kwargs = kwargs
        self.worker_pydebug: bool = False
        self.work_dir: str = kwargs.get("work_dir") or "/tmp/work"
        self.log_dir: str = kwargs.get("log_dir") or "/tmp/work/logs"
        self.run_log_dir: str = "/tmp/work/logs/run_001"

        self.env: dict[str, Any] = {}
        self.statuses: dict[str, _FakePluginStatus] = {}
        self.rpc_errors: dict[str, list[str]] = {}

        self.monitoring_enabled: bool = False
        self.report_hotkeys: list[str] = []
        self.started_parallel: bool | None = None
        self.stop_calls: int = 0
        self.close_calls: int = 0
        self.raise_on_stop: bool = False

    def env_set(self, key: str, value: Any) -> None:
        self.env[key] = value

    def env_get(self, key: str | None = None) -> Any:
        if key is None:
            return dict(self.env)
        return self.env.get(key)

    def enable_monitoring(self) -> None:
        self.monitoring_enabled = True

    def enable_report_hotkey(self, hotkey: str) -> None:
        self.report_hotkeys.append(hotkey)

    def start_all_plugins(self, parallel: bool = True) -> None:
        self.started_parallel = parallel

    def plugin_status_all(self) -> dict[str, _FakePluginStatus]:
        return self.statuses

    def plugin_errors(self, plugin_id: str) -> list[str]:
        return list(self.rpc_errors.get(plugin_id, []))

    def stop_all_plugins(self) -> None:
        self.stop_calls += 1
        if self.raise_on_stop:
            raise RuntimeError("stop failed")

    def close(self) -> None:
        self.close_calls += 1


class _DummyPipeline(Pipeline):
    def __init__(
        self,
        config: PipelineConfig,
        options: PipelineOptions | None = None,
        complete_fn: Callable[[_DummyPipeline], bool] | None = None,
    ):
        super().__init__(config=config, options=options)
        self._complete_fn = complete_fn or (lambda p: bool(p.get_env("done")))

    def is_complete(self) -> bool:
        return self._complete_fn(self)


@pytest.fixture
def fake_manager(monkeypatch):
    created: list[_FakePluginManager] = []

    def _factory(**kwargs: Any) -> _FakePluginManager:
        mgr = _FakePluginManager(**kwargs)
        created.append(mgr)
        return mgr

    monkeypatch.setattr(pipeline_mod, "PluginManager", _factory)
    monkeypatch.setattr(pipeline_mod, "is_debugpy_attached", lambda: True)
    return created


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_pipeline_abc_cannot_be_instantiated_directly(fake_manager):
    with pytest.raises(TypeError):
        Pipeline(PipelineConfig())  # type: ignore[abstract]


def test_pipeline_init_publishes_config_and_exposes_dirs(fake_manager):
    cfg = PipelineConfig(name="test_run", seed=7, device="cpu")
    opts = PipelineOptions(work_dir="/tmp/custom", display_logs=False)

    pipe = _DummyPipeline(cfg, opts)
    mgr = fake_manager[-1]

    assert pipe.plugin_manager is mgr
    assert mgr.worker_pydebug is True
    assert mgr.init_kwargs["work_dir"] == "/tmp/custom"
    assert mgr.init_kwargs["display_logs"] is False
    assert pipe.get_env("pipeline_config") is cfg
    assert pipe.get_env()["pipeline_config"] is cfg
    assert pipe.work_dir == "/tmp/custom"
    assert pipe.log_dir == mgr.log_dir
    assert pipe.run_log_dir == mgr.run_log_dir


def test_pipeline_monitoring_and_run_parallel_flag(fake_manager):
    pipe = _DummyPipeline(PipelineConfig(), PipelineOptions(parallel_plugins=False))
    mgr = fake_manager[-1]

    pipe.enable_monitoring(hotkey="r")
    assert mgr.monitoring_enabled is True
    assert mgr.report_hotkeys == ["r"]

    pipe.enable_monitoring(hotkey=None)
    assert mgr.report_hotkeys == ["r"]

    pipe.run()
    assert mgr.started_parallel is False

    pipe.run(parallel=True)
    assert mgr.started_parallel is True


def test_pipeline_all_plugins_terminated(fake_manager):
    pipe = _DummyPipeline(PipelineConfig())
    mgr = fake_manager[-1]

    # No plugins loaded -> False
    assert pipe.all_plugins_terminated() is False

    # One alive, one dead -> False
    mgr.statuses = {
        "p1": _FakePluginStatus(process_alive=True),
        "p2": _FakePluginStatus(process_alive=False),
    }
    assert pipe.all_plugins_terminated() is False

    # All dead -> True
    mgr.statuses["p1"].process_alive = False
    assert pipe.all_plugins_terminated() is True


def test_pipeline_is_error_and_get_errors(fake_manager):
    pipe = _DummyPipeline(PipelineConfig())
    mgr = fake_manager[-1]

    mgr.statuses = {
        "ok": _FakePluginStatus(process_alive=True, worker_status=_FakeWorkerStatus(state="running")),
        "no_ws": _FakePluginStatus(process_alive=False, worker_status=None),
    }
    assert pipe.is_error() is False
    assert pipe.get_errors() == {}

    # Worker status with error state & errors list (even if dead)
    mgr.statuses["crashed"] = _FakePluginStatus(
        process_alive=False,
        worker_status=_FakeWorkerStatus(state="error", errors=["boom"]),
    )
    # Alive worker without worker_status.errors falls back to plugin_errors()
    mgr.rpc_errors["ok"] = ["rpc_warning"]

    assert pipe.is_error() is True
    assert pipe.get_errors() == {"crashed": ["boom"], "ok": ["rpc_warning"]}


def test_pipeline_wait_completes_cleanly(fake_manager):
    calls = {"n": 0}

    def _complete(p: _DummyPipeline) -> bool:
        calls["n"] += 1
        return calls["n"] >= 3

    pipe = _DummyPipeline(
        PipelineConfig(),
        PipelineOptions(poll_interval=0.001, timeout_sec=2.0),
        complete_fn=_complete,
    )
    pipe.wait()
    assert calls["n"] == 3
    assert fake_manager[-1].close_calls == 0


def test_pipeline_wait_raises_on_error_and_shuts_down(fake_manager):
    pipe = _DummyPipeline(
        PipelineConfig(),
        PipelineOptions(poll_interval=0.001, terminate_on_error=True),
    )
    mgr = fake_manager[-1]
    mgr.statuses = {
        "trainer": _FakePluginStatus(
            process_alive=False,
            worker_status=_FakeWorkerStatus(state="error", errors=["CUDA OOM"]),
        )
    }

    with pytest.raises(RuntimeError, match="CUDA OOM"):
        pipe.wait()

    assert mgr.stop_calls == 1
    assert mgr.close_calls == 1


def test_pipeline_wait_continues_when_terminate_on_error_false(fake_manager, caplog):
    calls = {"n": 0}

    def _complete(p: _DummyPipeline) -> bool:
        calls["n"] += 1
        return calls["n"] >= 2

    pipe = _DummyPipeline(
        PipelineConfig(),
        PipelineOptions(poll_interval=0.001, terminate_on_error=False),
        complete_fn=_complete,
    )
    mgr = fake_manager[-1]
    mgr.statuses = {
        "worker": _FakePluginStatus(
            process_alive=True,
            worker_status=_FakeWorkerStatus(state="error", errors=["non-fatal"]),
        )
    }

    with caplog.at_level(logging.ERROR, logger=pipeline_mod.__name__):
        pipe.wait()

    assert calls["n"] == 2
    assert mgr.close_calls == 0
    assert any("non-fatal" in rec.message for rec in caplog.records)


def test_pipeline_wait_timeout_shuts_down_and_raises(fake_manager):
    pipe = _DummyPipeline(
        PipelineConfig(),
        PipelineOptions(poll_interval=0.005, timeout_sec=0.02),
        complete_fn=lambda p: False,
    )
    mgr = fake_manager[-1]

    with pytest.raises(TimeoutError, match="exceeded timeout"):
        pipe.wait()

    assert mgr.stop_calls == 1
    assert mgr.close_calls == 1


def test_pipeline_status_snapshot_logging(fake_manager, caplog):
    pipe = _DummyPipeline(PipelineConfig())
    mgr = fake_manager[-1]
    mgr.statuses = {
        "p_ok": _FakePluginStatus(process_alive=True, worker_status=_FakeWorkerStatus(state="running")),
        "p_err": _FakePluginStatus(process_alive=True, worker_status=_FakeWorkerStatus(state="error", errors=["e1"])),
        "p_unk": _FakePluginStatus(process_alive=True, worker_status=None),
        "p_dead": _FakePluginStatus(process_alive=False, worker_status=None),
    }

    with caplog.at_level(logging.INFO, logger=pipeline_mod.__name__):
        pipe._log_status_snapshot(elapsed=1.5, verbose=False)
    messages = [r.message for r in caplog.records]
    assert any("[p_err] ERRORS: ['e1']" in m for m in messages)
    assert any("[p_unk] ALIVE (status unknown)" in m for m in messages)
    assert not any("[p_ok]" in m for m in messages)
    assert not any("[p_dead]" in m for m in messages)

    caplog.clear()
    with caplog.at_level(logging.INFO, logger=pipeline_mod.__name__):
        pipe._log_status_snapshot(elapsed=2.0, verbose=True)
    messages = [r.message for r in caplog.records]
    assert any("[p_ok] STATE: running" in m for m in messages)
    assert any("[p_dead] Terminated" in m for m in messages)


def test_pipeline_shutdown_close_idempotent_and_context_manager(fake_manager):
    pipe = _DummyPipeline(PipelineConfig())
    mgr = fake_manager[-1]
    mgr.raise_on_stop = True

    with pipe as entered:
        assert entered is pipe

    # __exit__ called shutdown() once even though stop_all_plugins raised
    assert mgr.stop_calls == 1
    assert mgr.close_calls == 1

    # Subsequent calls are idempotent no-ops
    pipe.shutdown()
    pipe.close()
    assert mgr.stop_calls == 1
    assert mgr.close_calls == 1
