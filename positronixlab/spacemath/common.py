
from __future__ import annotations
import functools
import torch
from torch import nn
import os
import shutil

from typing import Any, Callable, TypeVar
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
torch.Tensor.zclamp_ = zclamp_
torch.Tensor.zclamp  = zclamp
torch.Tensor.spow_   = spow_
torch.Tensor.spow    = spow




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


def compile(func: F) -> F:
    """
    Simple torch.compile decorator. Stack under @staticmethod / @classmethod:

        @staticmethod
        @compile
        def f(x, y): ...

        @classmethod
        @compile
        def f(cls, x, y): ...
    """
    compiled = torch.compile(func, mode=default_compile_mode)
    return functools.wraps(func)(compiled)


def classmethod_compile(func: F) -> classmethod:
    """
    @classmethod_compile
    def f(cls, x, y): ...
    """
    return classmethod(compile(func))


def staticmethod_compile(func: F) -> staticmethod:
    """
    @staticmethod_compile
    def f(x, y): ...
    """
    return staticmethod(compile(func))




def clear_torch_compile_cache(triton: bool = True, verbose: bool = False) -> None:
    
    if verbose:
        print(f"[clear_torch_compile_cache] --------------------------- ")

    """Reset Dynamo/Inductor state and wipe on-disk compile caches."""
    import torch._dynamo as dynamo
    dynamo.reset()

    try:
        import torch._inductor.codecache as codecache
        if hasattr(codecache, "clear_cache"):
            codecache.clear_cache()
        else:
            from torch._inductor.utils import clear_inductor_caches
            clear_inductor_caches()
    except Exception as e:
        if verbose:
            print(f"[skip] inductor codecache.clear_cache(): {e}")
    

    paths = [f"/tmp/torchinductor_{os.environ.get('USER', '')}"]
    if custom := os.environ.get("TORCHINDUCTOR_CACHE_DIR"):
        paths.append(custom)
    if triton:
        paths.append(os.path.expanduser("~/.triton/cache"))
        if custom := os.environ.get("TRITON_CACHE_DIR"):
            paths.append(custom)

    for p in paths:
        if p and os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
            if verbose:
                print(f"removed {p}")

    if verbose:
        print("torch.compile cache cleared.")
        print(f"[clear_torch_compile_cache] --------------------------- ")
    
