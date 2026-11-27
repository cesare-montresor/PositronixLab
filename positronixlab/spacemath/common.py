
from __future__ import annotations

import functools
import getpass
import math
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar, cast

import torch
from torch import nn

F = TypeVar("F", bound=Callable[..., Any])


default_compile_mode = "default" #"reduce-overhead" # 'default', 'reduce-overhead', 'max-autotune'

torch.set_float32_matmul_precision('high')

def zclamp_(x: torch.Tensor, eps: float = 1e-8, force_zero: bool = False) -> torch.Tensor:
    """In-place: push near-zero non-zero values to ±eps, or zero if force_zero is True."""
    mask = (x.abs() < eps) & (x != 0)
    if force_zero:
        x[mask] = 0
    else:
        x[mask] = x[mask].sign() * eps
    return x

def zclamp(x: torch.Tensor, eps: float = 1e-8, force_zero: bool = False) -> torch.Tensor:
    """Out-of-place version of zclamp_."""
    return zclamp_(x.clone(), eps, force_zero)

def spow_(x: torch.Tensor, power: float, eps: float = 1e-8, force_zero: bool = True) -> torch.Tensor:
    """In-place safe pow: clamps near-zero values before raising to power. Pushes near-zero values to ±eps or zero if force_zero is True."""
    zclamp_(x, eps, force_zero)       # explicit free-function call, no ordering magic
    x.pow_(power)
    return x

def spow(x: torch.Tensor, power: float, eps: float = 1e-8, force_zero: bool = True) -> torch.Tensor:
    """Out-of-place safe pow."""
    return spow_(x.clone(), power, eps, force_zero)

# Monkey-patch — done once, clearly separated from logic
_TensorCls: Any = torch.Tensor
_TensorCls.zclamp_ = zclamp_
_TensorCls.zclamp = zclamp
_TensorCls.spow_ = spow_
_TensorCls.spow = spow




def infer_shape(self, x_in:torch.Tensor, x_out:torch.Tensor):
    """Infer batch size, input shape, and output shape from the input and output tensors."""
    batch_size = x_in[0].shape[0]
    input_shape = x_in[0].shape[1:]
    output_shape = x_out.shape[1:]
    return batch_size, input_shape, output_shape



def is_activation(module: nn.Module):
    return module.__class__.__module__ == 'torch.nn.modules.activation'

def is_layer(module: nn.Module):
    if isinstance(module, nn.Identity):
        return False
    name = module.__class__.__module__.lower()
    layers = [
        'torch.nn.modules.linear',
        'torch.nn.modules.conv',
        'torch.nn.modules.rnn',
        'torch.nn.modules.transformer'
    ]
    return isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)) or name in layers

def is_conv(module: nn.Module):
    return isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)) or module.__class__.__module__ == 'torch.nn.modules.conv'

def is_linear(module: nn.Module):
    return isinstance(module, nn.Linear)


def _compile_enabled() -> bool:
    return os.environ.get("POSITRONIXLAB_COMPILE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def torch_compile(func: F) -> F:
    """
    Simple torch.compile decorator. Stack under @staticmethod / @classmethod:

        @staticmethod
        @torch_compile
        def f(x, y): ...

        @classmethod
        @torch_compile
        def f(cls, x, y): ...

    Set ``POSITRONIXLAB_COMPILE=0`` to disable compilation.
    """
    if not _compile_enabled():
        return func
    compiled = torch.compile(func, mode=default_compile_mode)
    return cast(F, functools.wraps(func)(compiled))


def classmethod_compile(func: Callable[..., Any]) -> classmethod:
    """
    @classmethod_compile
    def f(cls, x, y): ...
    """
    return classmethod(torch_compile(func))


def staticmethod_compile(func: Callable[..., Any]) -> staticmethod:
    """
    @staticmethod_compile
    def f(x, y): ...
    """
    return staticmethod(torch_compile(func))


def _is_safe_cache_dir(path: str) -> bool:
    """Return True only if *path* is a recognizable, non-root compile cache directory."""
    if not path:
        return False
    resolved = Path(path).expanduser().resolve()
    forbidden = {Path("/").resolve(), Path("/tmp").resolve(), Path.home().resolve()}
    if resolved in forbidden:
        return False
    name_lower = resolved.name.lower()
    parts_lower = [part.lower() for part in resolved.parts]
    return (
        "torchinductor" in name_lower
        or "triton" in name_lower
        or (any("triton" in part for part in parts_lower) and name_lower == "cache")
    )


def clear_torch_compile_cache(triton: bool = True, verbose: bool = False) -> None:
    """Reset Dynamo/Inductor state and wipe on-disk compile caches."""
    if verbose:
        print("[clear_torch_compile_cache] --------------------------- ")

    import torch._dynamo as dynamo
    dynamo.reset()

    try:
        import torch._inductor.codecache as codecache
        clear_fn = getattr(codecache, "clear_cache", None)
        if clear_fn is not None:
            clear_fn()
        else:
            from torch._inductor.utils import clear_inductor_caches
            clear_inductor_caches()
    except Exception as e:
        if verbose:
            print(f"[skip] inductor codecache.clear_cache(): {e}")

    try:
        user = getpass.getuser()
    except Exception:
        user = os.environ.get("USER", "")

    paths = []
    if user:
        paths.append(f"/tmp/torchinductor_{user}")
    if custom := os.environ.get("TORCHINDUCTOR_CACHE_DIR"):
        paths.append(custom)
    if triton:
        paths.append(os.path.expanduser("~/.triton/cache"))
        if custom := os.environ.get("TRITON_CACHE_DIR"):
            paths.append(custom)

    for p in paths:
        if p and os.path.isdir(p) and _is_safe_cache_dir(p):
            shutil.rmtree(p, ignore_errors=True)
            if verbose:
                print(f"removed {p}")

    if verbose:
        print("torch.compile cache cleared.")
        print("[clear_torch_compile_cache] --------------------------- ")


def effective_rank(positions: torch.Tensor) -> float:
    """Compute spectral effective rank ``exp(H(sigma_norm))`` of centered 2D embeddings.

    Used to quantify dimensional utilization and detect embedding collapse in
    modular spaces (Roy & Vetterli, 2007).
    """
    if positions.ndim != 2 or positions.shape[0] < 2 or positions.shape[1] < 2:
        return 1.0
    centered = positions.detach().cpu().float()
    centered = centered - centered.mean(dim=0, keepdim=True)
    sv = torch.linalg.svdvals(centered)
    total = float(sv.sum().item())
    if total <= 1e-12:
        return 0.0
    p = (sv / total).clamp_min(1e-12)
    entropy = float(-(p * p.log()).sum().item())
    return math.exp(entropy)

    
