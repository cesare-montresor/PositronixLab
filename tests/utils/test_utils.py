"""Unit tests for positronixlab.utils.utils."""

import os
import random
import subprocess
import sys
import threading

import numpy as np
import pytest
import torch
import torch.nn as nn

from positronixlab.utils import utils
from positronixlab.utils.utils import (
    assert_tensor,
    camel_to_kebab,
    camel_to_snake,
    check_tensor,
    class_name,
    f1s_from_truth_table,
    get_thread_name,
    info_tensor,
    is_valid_tensor,
    load_data,
    model_hash,
    module_name,
    process_start_spawn,
    safe_tensor,
    save_data,
    seed_everything,
    session_path,
    set_best,
    signature_settings,
    suppress_output,
    truth_table_from_cm,
)


# ---------------------------------------------------------------------------
# process_start_spawn
# ---------------------------------------------------------------------------


def test_process_start_spawn_sets_spawn_when_unset():
    code = (
        "import multiprocessing as mp; "
        "from positronixlab.utils.utils import process_start_spawn; "
        "assert mp.get_start_method(allow_none=True) is None; "
        "process_start_spawn(); "
        "print(mp.get_start_method())"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert proc.stdout.strip() == "spawn"


def test_process_start_spawn_warns_when_fork():
    code = "\n".join([
        "import multiprocessing as mp, warnings",
        "from positronixlab.utils.utils import process_start_spawn",
        "mp.set_start_method('fork')",
        "with warnings.catch_warnings(record=True) as w:",
        "    warnings.simplefilter('always')",
        "    process_start_spawn(force=False)",
        "    print(len(w), w[0].message if w else '')",
    ])
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert "Expected spawn start method, got 'fork'" in proc.stdout


@pytest.mark.xfail(
    strict=True,
    reason="Bug B2: process_start_spawn(force=True) only warns when start method is 'fork' and never calls mp.set_start_method('spawn', force=True)",
)
def test_process_start_spawn_force_overrides_fork():
    code = (
        "import multiprocessing as mp; "
        "from positronixlab.utils.utils import process_start_spawn; "
        "mp.set_start_method('fork'); "
        "process_start_spawn(force=True); "
        "print(mp.get_start_method())"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert proc.stdout.strip() == "spawn"


# ---------------------------------------------------------------------------
# suppress_output
# ---------------------------------------------------------------------------


def test_suppress_output_silences_c_level_stderr_and_restores():
    code = "\n".join([
        "import os",
        "from positronixlab.utils.utils import suppress_output",
        "os.write(2, b'BEFORE\\n')",
        "with suppress_output(stderr=True):",
        "    os.write(2, b'SUPPRESSED\\n')",
        "os.write(2, b'AFTER\\n')",
    ])
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert "BEFORE" in proc.stderr
    assert "SUPPRESSED" not in proc.stderr
    assert "AFTER" in proc.stderr


def test_suppress_output_restores_fd_on_exception():
    code = "\n".join([
        "import os",
        "from positronixlab.utils.utils import suppress_output",
        "try:",
        "    with suppress_output(stderr=True):",
        "        os.write(2, b'SUPPRESSED\\n')",
        "        raise RuntimeError('boom')",
        "except RuntimeError:",
        "    pass",
        "os.write(2, b'RESTORED\\n')",
    ])
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert "SUPPRESSED" not in proc.stderr
    assert "RESTORED" in proc.stderr


@pytest.mark.xfail(
    strict=True,
    reason="Bug B6: suppress_output(stderr=False) does not flush sys.stdout before restoring fd 1, so buffered print() output leaks when piped",
)
def test_suppress_output_flushes_buffered_python_stdout():
    code = "\n".join([
        "from positronixlab.utils.utils import suppress_output",
        "with suppress_output(stderr=False):",
        "    print('SHOULD_BE_HIDDEN')",
    ])
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert proc.stdout == ""


# ---------------------------------------------------------------------------
# session_path
# ---------------------------------------------------------------------------


def test_session_path_create_cache_and_force(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(utils, "last_session_path", None)

    p1 = session_path("exp", uid="001", createDirectory=True)
    assert p1 == "output/exp_001/"
    assert (tmp_path / "output" / "exp_001").is_dir()

    # Without force=True, returns cached path even with different args
    p2 = session_path("other", uid="002", createDirectory=True, force=False)
    assert p2 == p1

    # With force=True, generates a new path and respects createDirectory=False
    p3 = session_path("other", uid="002", createDirectory=False, force=True)
    assert p3 == "output/other_002/"
    assert not (tmp_path / "output" / "other_002").exists()


# ---------------------------------------------------------------------------
# String & thread & class naming helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "src,expected_kebab,expected_snake",
    [
        ("CamelCase", "camel-case", "camel_case"),
        ("HTTPResponse", "http-response", "http_response"),
        ("getHTTP2Response", "get-http2-response", "get_http2_response"),
        ("simple", "simple", "simple"),
        ("", "", ""),
    ],
)
def test_camel_conversions(src, expected_kebab, expected_snake):
    assert camel_to_kebab(src) == expected_kebab
    assert camel_to_snake(src) == expected_snake


def test_get_thread_name_extracts_target_and_normalizes():
    results = []

    def worker():
        results.append(get_thread_name())

    t = threading.Thread(target=worker, name="Thread-1 (My-Worker Name)")
    t.start()
    t.join()
    assert results == ["my_worker_name"]

    t2 = threading.Thread(target=worker, name="Plain-Thread Name")
    t2.start()
    t2.join()
    assert results[-1] == "plain_thread_name"


def test_module_and_class_name():
    layer = nn.Linear(2, 3)
    assert class_name(layer) == "Linear"
    assert module_name(layer) == "torch.nn.modules.linear"


# ---------------------------------------------------------------------------
# is_valid_tensor & assert_tensor
# ---------------------------------------------------------------------------


def test_is_valid_tensor_cases():
    assert is_valid_tensor(torch.tensor([1.0, 2.0])) is True
    assert is_valid_tensor(None) == "none"
    assert is_valid_tensor([1.0, 2.0]) == "not_tensor"
    assert is_valid_tensor(torch.empty(0)) == "empty"
    assert is_valid_tensor(torch.tensor([1.0, float("nan")])) == "nan"
    assert is_valid_tensor(torch.tensor([1.0, float("inf")])) == "inf"
    assert is_valid_tensor(torch.tensor([1.0, float("-inf")])) == "inf"


def test_assert_tensor_passes_and_raises():
    assert_tensor(torch.ones(2, 2), name="ok")
    with pytest.raises(ValueError, match="Invalid tensor: bad, reason: nan"):
        assert_tensor(torch.tensor([float("nan")]), name="bad")


# ---------------------------------------------------------------------------
# seed_everything
# ---------------------------------------------------------------------------


def test_seed_everything_reproducibility():
    seed_everything(123)
    r1 = (random.random(), float(np.random.rand()), float(torch.rand(1).item()))
    seed_everything(123)
    r2 = (random.random(), float(np.random.rand()), float(torch.rand(1).item()))
    assert r1 == r2
    assert os.environ["PYTHONHASHSEED"] == "123"
    assert torch.backends.cudnn.deterministic is True


@pytest.mark.xfail(
    strict=True,
    reason="Bug B3: seed_everything() sets torch.backends.cudnn.benchmark = True, contradicting cudnn.deterministic = True",
)
def test_seed_everything_disables_cudnn_benchmark():
    seed_everything(42)
    assert torch.backends.cudnn.benchmark is False


# ---------------------------------------------------------------------------
# check_tensor & info_tensor
# ---------------------------------------------------------------------------


def test_check_tensor_noop_when_debug_false():
    # Should not raise when debug=False
    check_tensor(torch.tensor([float("nan")]), name="t", debug=False)


def test_check_tensor_raises_on_nan_and_inf_when_debug_true():
    with pytest.raises(Exception, match="NaN detected"):
        check_tensor(torch.tensor([float("nan")]), name="nan_t", debug=True)
    with pytest.raises(Exception, match="NaN detected"):
        check_tensor(torch.tensor([float("inf")]), name="inf_t", debug=True)


def test_info_tensor_prints_stats_when_debug_true(capsys):
    info_tensor(torch.tensor([0.0, 1.0, 2.0]), name="sample", debug=True)
    out = capsys.readouterr().out
    assert "Tensor: sample" in out
    assert "Shape: torch.Size([3])" in out
    assert "Zeros: 1" in out


@pytest.mark.xfail(
    strict=True,
    reason="Bug B4: check_tensor(t, debug=True) calls info_tensor(tensor, name) without forwarding debug=debug",
)
def test_check_tensor_forwards_debug_to_info_tensor(capsys):
    with pytest.raises(Exception):
        check_tensor(torch.tensor([float("nan")]), name="broken", debug=True)
    out = capsys.readouterr().out
    assert "Tensor: broken" in out


@pytest.mark.xfail(
    strict=True,
    reason="Bug B4: info_tensor default argument debug=debug is early-bound to False at import time, ignoring runtime utils.debug = True",
)
def test_info_tensor_respects_module_debug_flag(monkeypatch, capsys):
    monkeypatch.setattr(utils, "debug", True)
    info_tensor(torch.tensor([1.0, 2.0]), name="runtime_dbg")
    out = capsys.readouterr().out
    assert "Tensor: runtime_dbg" in out


@pytest.mark.xfail(
    strict=True,
    reason="Bug B5: info_tensor() calls tensor.mean() directly, raising RuntimeError on integer tensors",
)
def test_info_tensor_handles_integer_tensor():
    info_tensor(torch.tensor([1, 2, 3], dtype=torch.int64), name="ints", debug=True)


@pytest.mark.xfail(
    strict=True,
    reason="Bug B5: info_tensor() calls tensor.min() without checking numel() == 0, raising RuntimeError on empty tensors",
)
def test_info_tensor_handles_empty_tensor():
    info_tensor(torch.tensor([]), name="empty", debug=True)


# ---------------------------------------------------------------------------
# truth_table_from_cm & f1s_from_truth_table
# ---------------------------------------------------------------------------


def test_truth_table_and_f1s_from_cm():
    # Binary CM: row=true, col=pred
    # Class 0: TP=5, FP=1, FN=2, TN=10 -> F1 = 10 / 13
    # Class 1: TP=10, FP=2, FN=1, TN=5 -> F1 = 20 / 23
    cm = np.array([[5, 2], [1, 10]])
    tt = truth_table_from_cm(cm)
    np.testing.assert_array_equal(tt, np.array([[5, 1, 2, 10], [10, 2, 1, 5]]))

    f1s = f1s_from_truth_table(tt)
    np.testing.assert_allclose(f1s, np.array([10 / 13, 20 / 23]))


def test_f1s_from_truth_table_handles_zero_division():
    tt = np.array([[0, 0, 0, 5], [2, 0, 0, 3]])
    f1s = f1s_from_truth_table(tt)
    np.testing.assert_allclose(f1s, np.array([0.0, 1.0]))


# ---------------------------------------------------------------------------
# set_best
# ---------------------------------------------------------------------------


def test_set_best_min_and_max_and_zero_guard():
    bests = {}
    assert set_best(bests, "loss", 2.0, min=True) == (1, None)
    assert bests["loss"] == 2.0

    # Improvement (min)
    diff, old = set_best(bests, "loss", 1.0, min=True)
    assert diff == pytest.approx(0.5)
    assert old == 2.0
    assert bests["loss"] == 1.0

    # No improvement (min)
    assert set_best(bests, "loss", 1.5, min=True) == (-1, 1.0)
    assert set_best(bests, "loss", 1.0, min=True) == (-1, 1.0)

    # Zero-division guard when old best == 0
    bests["zero"] = 0.0
    diff, old = set_best(bests, "zero", 0.5, min=False)
    assert diff == 1.0
    assert old == 0.0

    # Improvement & non-improvement (max)
    assert set_best(bests, "acc", 0.5, min=False) == (1, None)
    diff, old = set_best(bests, "acc", 0.75, min=False)
    assert diff == pytest.approx(0.5)
    assert old == 0.5
    assert set_best(bests, "acc", 0.7, min=False) == (-1, 0.75)


# ---------------------------------------------------------------------------
# signature_settings
# ---------------------------------------------------------------------------


def test_signature_settings_deterministic_and_sensitive():
    s1 = signature_settings({"b": 2, "a": [1, 3]})
    s2 = signature_settings({"a": [1, 3], "b": 2})
    s3 = signature_settings({"a": [1, 4], "b": 2})
    assert s1 == s2
    assert len(s1) == 32  # default md5 hex
    assert s1 != s3

    sha = signature_settings({"a": 1}, algo="sha256")
    assert len(sha) == 64


# ---------------------------------------------------------------------------
# save_data & load_data
# ---------------------------------------------------------------------------


def test_save_and_load_json_pickle_pt(tmp_path):
    payload = {"name": "demo", "values": [1, 2, 3]}

    for ext in (".json", ".pkl", ".pickle"):
        p = tmp_path / "nested" / f"data{ext}"
        assert save_data(str(p), payload) is True
        assert load_data(str(p)) == payload

    pt_path = tmp_path / "tensor.pt"
    t = torch.tensor([1.0, 2.0, 3.0])
    assert save_data(str(pt_path), {"t": t}) is True
    loaded = load_data(str(pt_path))
    assert torch.equal(loaded["t"], t)


def test_save_and_load_unsupported_and_missing(tmp_path):
    bad = tmp_path / "file.txt"
    assert save_data(str(bad), {"a": 1}) is False
    assert load_data(str(tmp_path / "does_not_exist.json")) is None

    bad.write_text("hello")
    assert load_data(str(bad)) is None


def test_save_and_load_yaml_roundtrip(tmp_path):
    p = tmp_path / "config.yaml"
    payload = {"lr": 0.01, "layers": [16, 8]}
    assert save_data(str(p), payload) is True
    assert load_data(str(p)) == payload


def test_save_data_supports_yml_extension(tmp_path):
    p = tmp_path / "config.yml"
    assert save_data(str(p), {"a": 1}) is True
    assert load_data(str(p)) == {"a": 1}


def test_save_and_load_data_kwargs_forwarding(tmp_path):
    from pathlib import Path

    # JSON kwargs (sort_keys, indent, default=str)
    jp = tmp_path / "META.JSON"
    assert save_data(str(jp), {"b": 1, "a": Path("x")}, sort_keys=True, indent=None, default=str) is True
    assert load_data(str(jp)) == {"a": "x", "b": 1}

    # Torch kwargs (map_location, weights_only=False for arbitrary dict payload)
    pt = tmp_path / "bundle.pt"
    payload = {"meta": {"step": 5}, "t": torch.tensor([1.0, 2.0])}
    assert save_data(str(pt), payload) is True
    loaded = load_data(str(pt), map_location="cpu", weights_only=False)
    assert loaded["meta"]["step"] == 5
    assert torch.equal(loaded["t"], payload["t"])


# ---------------------------------------------------------------------------
# safe_tensor
# ---------------------------------------------------------------------------


def test_safe_tensor_conversions():
    # Tensor input is detached, CPU, and >= 1D
    grad_t = torch.tensor(5.0, requires_grad=True)
    out = safe_tensor(grad_t)
    assert out.requires_grad is False
    assert out.device.type == "cpu"
    assert out.shape == (1,)
    assert out.item() == 5.0

    # Scalars
    assert torch.equal(safe_tensor(3), torch.tensor([3.0]))
    assert torch.equal(safe_tensor(2.5), torch.tensor([2.5]))
    assert torch.equal(safe_tensor(np.float32(4.0)), torch.tensor([4.0]))

    # Sequences & arrays
    assert torch.equal(safe_tensor([1, 2, 3]), torch.tensor([1, 2, 3]))
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    assert torch.equal(safe_tensor(arr), torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


# ---------------------------------------------------------------------------
# model_hash
# ---------------------------------------------------------------------------


def test_model_hash_deterministic_and_change_sensitive():
    torch.manual_seed(0)
    m1 = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))
    torch.manual_seed(0)
    m2 = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))

    h1 = model_hash(m1)
    h2 = model_hash(m2)
    assert h1 == h2
    assert len(h1) == 64

    # Mutate a single weight
    with torch.no_grad():
        m2[0].weight[0, 0] += 1e-4
    assert model_hash(m2) != h1

    # Different algorithm
    assert len(model_hash(m1, algo="md5")) == 32


def test_model_hash_special_dtypes_and_scalars():
    class BufferModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("bf16", torch.tensor([1.0, 2.0], dtype=torch.bfloat16))
            self.register_buffer("flag", torch.tensor([True, False]))
            self.register_buffer("scalar", torch.tensor(3.14))

    m = BufferModel()
    h = model_hash(m)
    assert isinstance(h, str) and len(h) == 64
