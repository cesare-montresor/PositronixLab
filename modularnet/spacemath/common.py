
from __future__ import annotations
from typing import Any, Callable

import torch
from torch import nn

default_compile_mode = "reduce-overhead" # 'default', 'reduce-overhead', 'max-autotune'

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
    name = module.__class__.__module__.lower()
    layers = [
        'torch.nn.modules.linear',
        'torch.nn.modules.conv',
        'torch.nn.modules.rnn',
        'torch.nn.modules.transformer'
    ]
    return name in layers

def is_conv(module: nn.Module):
    return module.__class__.__module__ == 'torch.nn.modules.conv'

def is_linear(module: nn.Module):
    return module.__class__.__module__ == 'torch.nn.modules.linear'


def set_torch_compile_defaults(mode: str = "reduce-overhead"):
    """
    Set the default mode for torch.compile globally.
    
    Args:
        mode (str): The compilation mode. Options include 'default', 'reduce-overhead', 'max-autotune', etc.
    """
    global default_compile_mode
    default_compile_mode = mode

def compile(*compile_args, **compile_kwargs) -> Callable[..., Callable[..., Any]]:
    """
    Custom decorator for standard instance methods (with 'self').
    
    While @torch.compile works directly on instance methods natively, 
    this provides a consistent API alongside `classmethod_compile`.
    """
    def decorator(func) -> Callable[..., Any]:
        compile_kwargs.setdefault("mode", default_compile_mode)
        return torch.compile(func, *compile_args, **compile_kwargs)
    return decorator

def classmethod_compile(*compile_args, **compile_kwargs)-> Callable[..., classmethod[Any, Callable[..., Any], Any]]:
    """
    Custom decorator that safely applies torch.compile and 
    converts the result into a classmethod. ( order would matter )
    
    This ensures the correct bottom-to-top execution order:
    raw function -> torch.compile -> classmethod descriptor.
    """
    def decorator(func) -> classmethod[Any, Callable[..., Any], Any]:
        # 1. Pass the raw function through our instance method compiler
        compiled_func = compile(*compile_args, **compile_kwargs)(func)
        # 2. Wrap the resulting compiled function in a classmethod descriptor
        return classmethod(compiled_func)
    return decorator

def staticmethod_compile(*compile_args, **compile_kwargs)-> Callable[..., staticmethod[Any, Callable[..., Any], Any]]:
    """
    Custom decorator that safely applies torch.compile and 
    converts the result into a staticmethod. ( order would matter )
    
    This ensures the correct bottom-to-top execution order:
    raw function -> torch.compile -> staticmethod descriptor.
    """
    def decorator(func)  -> staticmethod[Callable[..., Any], Any]:
        # 1. Pass the raw function through our instance method compiler
        compiled_func = compile(*compile_args, **compile_kwargs)(func)
        # 2. Wrap the resulting compiled function in a staticmethod descriptor
        return staticmethod(compiled_func)
    return decorator

