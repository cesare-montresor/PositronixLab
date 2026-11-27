"""Unit tests for positronixlab.stats.

Covers common, filter, reducer, processor, and stream.
Confirmed library defects are pinned with @pytest.mark.xfail(strict=True).
"""

import subprocess
import sys
import threading
import time
import types
from typing import Any

import numpy as np
import pytest
import torch

from positronixlab.stats.common import _as_tensor, _to_np
from positronixlab.stats.filter import DisplacementFilter
from positronixlab.stats.reducer import (
    REDUCERS,
    DisplacementReducer,
    EmaReducer,
    LastReducer,
    MeanReducer,
    WindowReducer,
)

# ---------------------------------------------------------------------------
# Lightweight CPU stub for UMAP so processor.py & stream.py can be imported
# and tested even on CPU-only hosts where cuML/libcuml.so is unavailable.
# ---------------------------------------------------------------------------


class _FakeUMAP:
    """Deterministic lightweight UMAP stand-in recording constructor/call args."""

    instances: list["_FakeUMAP"] = []

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        random_state: Any = None,
        init: Any = "spectral",
    ) -> None:
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state
        self.init = init
        self.fit_shapes: list[tuple] = []
        self.transform_shapes: list[tuple] = []
        self.fit_transform_shapes: list[tuple] = []
        _FakeUMAP.instances.append(self)

    def _project(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[:, None]
        if x.shape[1] >= self.n_components:
            return x[:, : self.n_components].copy()
        pad = np.zeros((x.shape[0], self.n_components - x.shape[1]), dtype=np.float32)
        return np.concatenate([x, pad], axis=1)

    def fit(self, x: np.ndarray) -> "_FakeUMAP":
        self.fit_shapes.append(tuple(x.shape))
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        self.transform_shapes.append(tuple(x.shape))
        return self._project(x)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        if not isinstance(self.init, str):
            init_arr = np.asarray(self.init)
            if init_arr.shape[0] != x.shape[0]:
                raise ValueError(
                    f"init rows ({init_arr.shape[0]}) != input rows ({x.shape[0]})"
                )
        self.fit_transform_shapes.append(tuple(x.shape))
        return self._project(x)


def _import_processor_and_stream():
    """Import processor & stream, stubbing cuml only if unavailable."""
    if "positronixlab.stats.processor" not in sys.modules:
        try:
            import positronixlab.stats.processor  # noqa: F401
        except (ImportError, OSError):
            cuml_mod = types.ModuleType("cuml")
            manifold_mod = types.ModuleType("cuml.manifold")
            umap_mod = types.ModuleType("cuml.manifold.umap")
            manifold_mod.UMAP = _FakeUMAP
            umap_mod.UMAP = _FakeUMAP
            cuml_mod.manifold = manifold_mod
            sys.modules["cuml"] = cuml_mod
            sys.modules["cuml.manifold"] = manifold_mod
            sys.modules["cuml.manifold.umap"] = umap_mod

    import positronixlab.stats.processor as proc_mod
    import positronixlab.stats.stream as stream_mod

    return proc_mod, stream_mod


@pytest.fixture
def fake_umap_env(monkeypatch):
    """Provide processor & stream modules with UMAP patched to _FakeUMAP."""
    proc_mod, stream_mod = _import_processor_and_stream()
    _FakeUMAP.instances.clear()
    monkeypatch.setattr(proc_mod, "UMAP", _FakeUMAP)
    return proc_mod, stream_mod


# ===========================================================================
# 1. common.py
# ===========================================================================


class TestCommon:
    def test_as_tensor_detaches_and_casts_float32(self):
        src = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
        out = _as_tensor(src)
        assert out.dtype == torch.float32
        assert not out.requires_grad
        assert torch.allclose(out, torch.tensor([1.0, 2.0]))

    def test_as_tensor_float32_view_semantics(self):
        # Documents the aliasing invariant: _as_tensor does NOT guarantee a copy
        src = torch.tensor([1.0, 2.0], dtype=torch.float32)
        out = _as_tensor(src)
        assert out.data_ptr() == src.data_ptr()

        arr = np.array([3.0, 4.0], dtype=np.float32)
        out_np = _as_tensor(arr)
        arr[0] = 99.0
        assert out_np[0].item() == 99.0

    def test_as_tensor_list_of_scalar_tensors(self):
        out = _as_tensor([torch.tensor(1), torch.tensor(2.5), torch.tensor([3.0])])
        assert out.shape == (3,)
        assert out.dtype == torch.float32
        assert torch.allclose(out, torch.tensor([1.0, 2.5, 3.0]))

    def test_as_tensor_python_values(self):
        assert _as_tensor(5).dtype == torch.float32
        assert torch.allclose(_as_tensor([1, 2, 3]), torch.tensor([1.0, 2.0, 3.0]))

    def test_as_tensor_list_of_1d_tensors_crashes(self):
        out = _as_tensor([torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])])
        assert out.shape == (2, 2)

    def test_to_np_returns_independent_copy(self):
        t = torch.tensor([1.0, 2.0], requires_grad=True)
        arr = _to_np(t)
        assert isinstance(arr, np.ndarray)
        t.detach().zero_()
        assert arr.tolist() == [1.0, 2.0]

        src_np = np.array([5, 6])
        copied = _to_np(src_np)
        src_np[0] = 99
        assert copied[0] == 5


# ===========================================================================
# 2. filter.py
# ===========================================================================


class TestDisplacementFilter:
    def test_init_and_shape_change_always_emit(self):
        f = DisplacementFilter(threshold=10.0, method="max")
        assert f.check_and_update("fc1", torch.zeros(4, 2)) is True
        assert f.check_and_update("fc1", torch.zeros(4, 2)) is False
        # Shape change forces emit and updates keyframe
        assert f.check_and_update("fc1", torch.zeros(5, 2)) is True
        assert f.check_and_update("fc1", torch.zeros(5, 2)) is False

    def test_max_method_threshold_gating(self):
        f = DisplacementFilter(threshold=0.5, method="max")
        base = torch.zeros(4, 2)
        assert f.check_and_update("k", base) is True

        # Single neuron moves by 0.4 <= 0.5 -> suppressed
        small = base.clone()
        small[0, 0] = 0.4
        assert f.check_and_update("k", small) is False

        # Single neuron moves by 0.6 > 0.5 -> emitted and keyframe updated
        big = base.clone()
        big[0, 0] = 0.6
        assert f.check_and_update("k", big) is True

        # Same tensor again -> 0 displacement from new keyframe -> suppressed
        assert f.check_and_update("k", big) is False

    def test_rms_method_threshold_gating(self):
        f = DisplacementFilter(threshold=0.5, method="rms")
        base = torch.zeros(4, 2)
        assert f.check_and_update("k", base) is True

        # One neuron moves by 0.8 -> RMS = sqrt(0.64 / 4) = 0.4 <= 0.5 -> suppressed
        moved = base.clone()
        moved[0, 0] = 0.8
        assert f.check_and_update("k", moved) is False

        # All neurons move by 0.6 -> RMS = 0.6 > 0.5 -> emitted
        assert f.check_and_update("k", torch.full((4, 2), 0.6)) is True

    def test_keyframe_clones_input(self):
        f = DisplacementFilter(threshold=0.5, method="max")
        buf = torch.zeros(2, 2)
        f.check_and_update("k", buf)
        buf.fill_(10.0)
        # Mutating caller buffer must not mutate stored keyframe
        assert f.check_and_update("k", torch.full((2, 2), 10.0)) is True

    def test_invalid_method_should_raise(self):
        with pytest.raises(ValueError):
            DisplacementFilter(threshold=0.5, method="invalid_method")

    def test_empty_tensor_max_crashes(self):
        f = DisplacementFilter(threshold=0.5, method="max")
        assert f.check_and_update("k", torch.empty(0, 2)) is True
        assert f.check_and_update("k", torch.empty(0, 2)) is False


# ===========================================================================
# 3. reducer.py
# ===========================================================================


class TestReducers:
    def test_registry_contains_all_reducers(self):
        assert set(REDUCERS.keys()) == {"last", "ema", "mean", "window", "displacement"}

    def test_last_reducer(self):
        r = LastReducer()
        buf = torch.tensor([1.0, 2.0])
        r.update("a", buf)
        buf.fill_(99.0)  # must not affect stored state
        r.update("a", torch.tensor([3.0, 4.0]))
        out = r.drain()
        assert torch.allclose(out["a"], torch.tensor([3.0, 4.0]))
        # Subsequent drain with no updates returns empty dict
        assert r.drain() == {}
        # Mutating returned snapshot must not mutate internal state
        out["a"].zero_()
        r.update("b", torch.tensor([5.0]))
        assert "a" not in r.drain()

    def test_ema_reducer_math_and_shape_reset(self):
        r = EmaReducer(alpha=0.25)
        r.update("x", torch.tensor([10.0, 20.0]))
        r.update("x", torch.tensor([14.0, 24.0]))
        # 0.75 * [10, 20] + 0.25 * [14, 24] = [11, 21]
        out = r.drain()
        assert torch.allclose(out["x"], torch.tensor([11.0, 21.0]))

        # Shape change resets EMA state cleanly
        r.update("x", torch.tensor([1.0, 2.0, 3.0]))
        assert torch.allclose(r.drain()["x"], torch.tensor([1.0, 2.0, 3.0]))

    def test_mean_reducer_block_semantics_and_thread_safety(self):
        r = MeanReducer()
        r.update("loss", torch.tensor(1.0))
        r.update("loss", torch.tensor(3.0))
        assert torch.allclose(r.drain()["loss"], torch.tensor(2.0))
        assert r.drain() == {}

        # Next block starts fresh
        r.update("loss", torch.tensor(10.0))
        assert torch.allclose(r.drain()["loss"], torch.tensor(10.0))

        # Shape change resets block sum/count
        r.update("v", torch.tensor([1.0, 2.0]))
        r.update("v", torch.tensor([3.0, 4.0, 5.0]))
        assert torch.allclose(r.drain()["v"], torch.tensor([3.0, 4.0, 5.0]))

        # Concurrent update + drain should not raise
        errors: list[Exception] = []

        def writer():
            try:
                for i in range(200):
                    r.update("m", torch.tensor(float(i)))
            except Exception as exc:
                errors.append(exc)

        def reader():
            try:
                for _ in range(100):
                    r.drain()
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert errors == []

    def test_window_reducer_rolling(self):
        r = WindowReducer(maxlen=3)
        for v in (1.0, 2.0, 3.0, 4.0):
            r.update("w", torch.tensor(v))
        out = r.drain()
        assert out["w"].tolist() == [2.0, 3.0, 4.0]
        assert r.drain() == {}

    def test_window_reducer_shape_change_crashes(self):
        r = WindowReducer(maxlen=5)
        r.update("w", torch.zeros(2))
        r.update("w", torch.zeros(3))
        out = r.drain()
        assert out["w"].shape[-1] == 3

    def test_window_reducer_maxlen_zero_crashes(self):
        r = WindowReducer(maxlen=0)
        r.update("w", torch.tensor(1.0))
        assert r.drain() == {}

    def test_displacement_reducer_max_and_rms(self):
        with pytest.raises(ValueError):
            DisplacementReducer(threshold=0.5, method="bad")

        r = DisplacementReducer(threshold=0.5, method="max")
        base = torch.zeros(4, 2)
        r.update("fc1", base)
        out = r.drain()
        assert "fc1" in out and torch.allclose(out["fc1"], base)
        assert "fc1" not in r._pending  # freed in _after_drain

        # Below threshold -> not dirty
        r.update("fc1", base + 0.1)
        assert r.drain() == {}

        # Above threshold -> dirty
        moved = base.clone()
        moved[0, 0] = 0.8
        r.update("fc1", moved)
        out2 = r.drain()
        assert torch.allclose(out2["fc1"], moved)

        # RMS mode
        r_rms = DisplacementReducer(threshold=0.5, method="rms")
        r_rms.update("fc1", base)
        r_rms.drain()
        r_rms.update("fc1", moved)  # rms = 0.4 <= 0.5
        assert r_rms.drain() == {}
        r_rms.update("fc1", torch.full((4, 2), 0.6))
        assert "fc1" in r_rms.drain()

    def test_displacement_reducer_empty_tensor_crashes(self):
        r = DisplacementReducer(threshold=0.5, method="max")
        r.update("k", torch.empty(0, 2))
        r.drain()
        r.update("k", torch.empty(0, 2))
        assert r.drain() == {}


# ===========================================================================
# 4. processor.py
# ===========================================================================


class TestProcessors:
    def test_import_processor_without_cuml_should_succeed(self):
        # Simulate an environment where cuml is not installed
        code = (
            "import sys; sys.modules['cuml'] = None; "
            "from positronixlab.stats.processor import OverallProcessor"
        )
        res = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )
        assert res.returncode == 0, res.stderr

    def test_registry_keys(self, fake_umap_env):
        proc_mod, _ = fake_umap_env
        assert set(proc_mod.PROCESSORS.keys()) == {"umap", "overall"}

    def test_umap_processor_collect_fit_warmstart_and_centroids(self, fake_umap_env):
        proc_mod, _ = fake_umap_env
        proc = proc_mod.UmapProcessor(
            n_components=2, n_samples=3, min_neighbors=4, max_neighbors=20, seed=7
        )

        # Collection phase (frames 1 and 2)
        assert proc.process({"fc1": torch.randn(16, 6)}) == []
        assert not proc.ready("fc1")
        assert proc.progress() == {"required": 3, "layers": {"fc1": 1}}
        assert len(proc.collecting()["fc1"]) == 1

        # Feature dim change during collection resets buffer
        assert proc.process({"fc1": torch.randn(16, 8)}) == []
        assert proc.progress()["layers"]["fc1"] == 1

        assert proc.process({"fc1": torch.randn(16, 8)}) == []
        assert proc.progress()["layers"]["fc1"] == 2

        # 3rd snapshot triggers fit and emits 3 frames (indices 0, 1, 2)
        msgs = proc.process({"fc1": torch.randn(16, 8)})
        assert proc.ready("fc1")
        assert len(msgs) == 3
        assert [m["index"] for m in msgs] == [0, 1, 2]
        assert all(m["value"].shape == (16, 2) for m in msgs)
        assert proc.last_index("fc1") == 2
        assert proc.progress()["layers"] == {}

        # Dynamic n_neighbors = clamp(int(sqrt(16))=4, 4, 20) == 4
        assert _FakeUMAP.instances[0].n_neighbors == 4
        assert _FakeUMAP.instances[0].init == "spectral"

        # Warm-start frame (index 3) passes previous 2D array as init
        msgs2 = proc.process({"fc1": torch.randn(16, 8)})
        assert len(msgs2) == 1
        assert msgs2[0]["index"] == 3
        assert proc.last_index("fc1") == 3
        warm_mapper = _FakeUMAP.instances[-1]
        assert isinstance(warm_mapper.init, np.ndarray)
        assert warm_mapper.init.shape == (16, 2)

        # centroids_2d
        labels = np.array([0] * 8 + [1] * 8)
        c2d = proc.centroids_2d("fc1", labels)
        assert c2d is not None
        assert c2d.shape == (2, 2)
        assert c2d.dtype == np.float32
        assert proc.centroids_2d("fc1", np.zeros(5)) is None  # length mismatch
        assert proc.centroids_2d("unknown", labels) is None

    def test_umap_last_index_before_emission_ambiguous(self, fake_umap_env):
        proc_mod, _ = fake_umap_env
        proc = proc_mod.UmapProcessor(n_samples=2)
        assert proc.last_index("fc1") == -1

    def test_umap_n_neighbors_exceeds_small_layer_size(self, fake_umap_env):
        proc_mod, _ = fake_umap_env
        proc = proc_mod.UmapProcessor(n_samples=1, min_neighbors=10, max_neighbors=50)
        mapper = proc._create_mapper(num_neurons=4)
        assert mapper.n_neighbors < 4

    def test_umap_warmstart_neuron_count_change_crashes(self, fake_umap_env):
        proc_mod, _ = fake_umap_env
        proc = proc_mod.UmapProcessor(n_samples=1, min_neighbors=2)
        proc.process({"fc1": torch.randn(8, 4)})
        # Neuron count changes from 8 to 10 after fit
        msgs = proc.process({"fc1": torch.randn(10, 4)})
        assert msgs[0]["value"].shape == (10, 2)

    def test_umap_centroids_2d_empty_crashes(self, fake_umap_env):
        proc_mod, _ = fake_umap_env
        proc = proc_mod.UmapProcessor(n_samples=1, min_neighbors=2)
        proc.process({"fc1": torch.empty(0, 4)})
        c2d = proc.centroids_2d("fc1", np.array([]))
        assert c2d is None or c2d.shape == (0, 2)

    def test_overall_processor(self, fake_umap_env):
        proc_mod, _ = fake_umap_env
        proc = proc_mod.OverallProcessor(topic="stats_overall", key="avg")

        msgs = proc.process({"fc1": torch.tensor([1.0, 3.0]), "fc2": torch.tensor([3.0, 5.0])})
        assert len(msgs) == 3
        overall = msgs[-1]
        assert overall["topic"] == "stats_overall"
        assert overall["key"] == "avg"
        assert torch.allclose(overall["value"], torch.tensor([2.0, 4.0]))

        # Updating only one key still averages across all known keys
        msgs2 = proc.process({"fc1": torch.tensor([5.0, 7.0])})
        assert torch.allclose(msgs2[-1]["value"], torch.tensor([4.0, 6.0]))

    def test_overall_processor_empty_items_crashes(self, fake_umap_env):
        proc_mod, _ = fake_umap_env
        proc = proc_mod.OverallProcessor(topic="stats_overall")
        assert proc.process({}) == []


# ===========================================================================
# 5. stream.py
# ===========================================================================


class TestStream:
    def test_stream_every_n_and_drain(self, fake_umap_env):
        _, stream_mod = fake_umap_env
        extractors = {
            "train": lambda m: list(m["metric"].items()) if m.get("name") == "train" else []
        }
        s = stream_mod.Stream(
            "train",
            {"topic": "metrics", "reducer": "mean", "every_n": 2},
            extractors,
        )
        assert not s.due(time.monotonic())

        # Unrelated message ignored
        s.ingest({"name": "other", "metric": {"loss": 1.0}})
        assert not s.due(time.monotonic())

        s.ingest({"name": "train", "metric": {"loss": 0.6}})
        assert not s.due(time.monotonic())

        s.ingest({"name": "train", "metric": {"loss": 0.4}})
        assert s.due(time.monotonic())

        blocks, n = s.drain()
        assert n == 2
        assert torch.allclose(blocks["loss"], torch.tensor(0.5))
        assert not s.due(time.monotonic())

    def test_stream_interval_s_and_callable_extractor(self, fake_umap_env):
        _, stream_mod = fake_umap_env
        s = stream_mod.Stream(
            "custom",
            {
                "topic": "raw",
                "extract": lambda m: [("val", m["v"])],
                "reducer": "last",
                "interval_s": 0.5,
            },
            extractors={},
        )
        t0 = s._last_emit
        s.ingest({"v": 42.0})
        assert not s.due(t0 + 0.1)
        assert s.due(t0 + 0.6)
        blocks, n = s.drain()
        assert n == 1
        assert blocks["val"].item() == 42.0

    def test_stream_with_overall_processor(self, fake_umap_env):
        _, stream_mod = fake_umap_env
        s = stream_mod.Stream(
            "loss",
            {
                "topic": "metrics",
                "extract": lambda m: [(m["layer"], m["v"])],
                "reducer": "mean",
                "processor": "overall",
                "processor_args": {"topic": "loss_overall"},
                "every_n": 2,
            },
            extractors={},
        )
        s.ingest({"layer": "l1", "v": [1.0, 2.0]})
        s.ingest({"layer": "l2", "v": [3.0, 4.0]})
        blocks, n = s.drain()
        assert n == 2
        msgs = s.processor.process(blocks)
        assert msgs[-1]["topic"] == "loss_overall"
        assert torch.allclose(msgs[-1]["value"], torch.tensor([2.0, 3.0]))

    def test_merge_streams_rules(self, fake_umap_env):
        _, stream_mod = fake_umap_env
        base: dict[str, dict[str, Any]] = {
            "pos": {
                "topic": "positions",
                "reducer": "ema",
                "reducer_args": {"alpha": 0.3},
                "processor": "umap",
                "processor_args": {"n_samples": 50, "min_dist": 0.1},
            },
            "loss": {"topic": "metrics", "reducer": "mean"},
            "train": {"topic": "metrics", "reducer": "mean"},
        }
        user = {
            # Same processor kind -> merge processor_args
            "pos": {"processor_args": {"n_samples": 20}},
            # Disable via None
            "loss": None,
            # Disable via enabled=False
            "train": {"enabled": False},
            # Change reducer kind without reducer_args -> old args dropped
            "new_s": {"topic": "t", "reducer": "last", "enabled": True},
        }
        merged = stream_mod.Stream.merge_streams(user, base)
        assert "loss" not in merged and "train" not in merged
        assert merged["pos"]["processor_args"] == {"n_samples": 20, "min_dist": 0.1}
        assert "enabled" not in merged["new_s"]
        # Original base must not be mutated
        assert base["pos"]["processor_args"]["n_samples"] == 50

        # Switching reducer kind drops stale reducer_args
        switched = stream_mod.Stream.merge_streams({"pos": {"reducer": "mean"}}, base)
        assert "reducer_args" not in switched["pos"]

    def test_stream_empty_generator_extractor_increments_count(self, fake_umap_env):
        _, stream_mod = fake_umap_env

        def empty_gen(_msg):
            if False:
                yield ("k", 1.0)

        s = stream_mod.Stream(
            "g",
            {"topic": "t", "extract": empty_gen, "reducer": "last", "every_n": 1},
            extractors={},
        )
        s.ingest({})
        assert not s.due(time.monotonic())

    def test_stream_displacement_gated_should_not_be_due(self, fake_umap_env):
        _, stream_mod = fake_umap_env
        s = stream_mod.Stream(
            "pos",
            {
                "topic": "positions",
                "extract": lambda m: [("fc1", m["p"])],
                "reducer": "displacement",
                "reducer_args": {"threshold": 1.0, "method": "max"},
                "every_n": 1,
            },
            extractors={},
        )
        base = torch.zeros(4, 2)
        s.ingest({"p": base})
        s.drain()
        # Small move below threshold -> reducer stays clean, stream should not emit empty blocks
        s.ingest({"p": base + 0.01})
        assert not s.due(time.monotonic())

    def test_merge_streams_none_args_crashes(self, fake_umap_env):
        _, stream_mod = fake_umap_env
        base = {"s": {"topic": "t", "reducer": "ema", "reducer_args": {"alpha": 0.3}}}
        merged = stream_mod.Stream.merge_streams({"s": {"reducer_args": None}}, base)
        assert merged["s"].get("reducer_args") is None
