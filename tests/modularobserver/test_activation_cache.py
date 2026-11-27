"""Unit tests for ActivationCache, reduce_activation_2d, and infer_layer_info."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from positronixlab.baselines import ModelLoader
from positronixlab.dataprovider import AddMulDataProvider
from positronixlab.modularobserver import ActivationCache, infer_layer_info, reduce_activation_2d


class _FakeConv1D(nn.Module):
    """Mimics HuggingFace GPT-2 Conv1D."""

    def __init__(self, nf: int) -> None:
        super().__init__()
        self.nf = nf


_FakeConv1D.__name__ = "Conv1D"


def test_activation_cache_build_and_hit(tmp_path: Path) -> None:
    dp = AddMulDataProvider(n_digits=1, n_train=32, n_val=16, n_test=16, batch_size=16, seed=0)
    kwargs = ModelLoader.infer_kwargs("csordas:mlp", dp, {"layer_sizes": [16, 8]})
    model = ModelLoader.create("csordas:mlp", device="cpu", **kwargs)

    conv = nn.Conv2d(3, 4, kernel_size=3)
    assert reduce_activation_2d(torch.randn(2, 4, 5, 5), conv).shape == (2, 4)
    assert infer_layer_info("conv1", conv)["out_units"] == 4
    assert infer_layer_info("c_fc", _FakeConv1D(12))["out_units"] == 12

    cache_path = ActivationCache.compute_cache_path(
        tmp_path,
        signature_fields={"model": "mlp", "split": "val"},
        cache_mode="activations",
        cache_dtype="float32",
        prefix="test_mlp",
    )
    cache = ActivationCache(cache_path, cache_mode="activations", cache_dtype="float32")
    calls = 0

    def _provider():
        nonlocal calls
        calls += 1
        return model, dp.val_loader()

    meta1, tensors1, hit1 = cache.load_or_build(_provider, device="cpu")
    assert not hit1 and calls == 1
    assert meta1["datapoint_count"] == 16
    assert len(tensors1) > 0

    meta2, tensors2, hit2 = cache.load_or_build(_provider, device="cpu")
    assert hit2 and calls == 1
    assert meta2["datapoint_count"] == 16
    assert set(tensors2.keys()) == set(tensors1.keys())


def test_activation_cache_scores_mode_applies_conv_reducer(tmp_path: Path) -> None:
    """Verify scores mode passes raw 4D conv outputs to reducers instead of pre-averaging."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(2, 6, kernel_size=3, padding=1))
    # Spatial input with zero mean per channel but non-zero L2/max across spatial dims
    x = torch.randn(4, 2, 5, 5)
    y = torch.zeros(4, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    cache_l2 = ActivationCache(
        tmp_path / "l2", cache_mode="scores", reducer="conv_l2", scoring="quantile"
    )
    cache_max = ActivationCache(
        tmp_path / "max", cache_mode="scores", reducer="conv_max", scoring="quantile"
    )
    _, t_l2 = cache_l2.build(model, loader, device="cpu")
    _, t_max = cache_max.build(model, loader, device="cpu")
    assert t_l2["0"].shape == (4, 6)
    assert t_max["0"].shape == (4, 6)
    # If pre-averaged to 2D before activations_to_scores, conv_l2 and conv_max would be identical.
    assert not torch.allclose(t_l2["0"], t_max["0"])
