import copy
import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.multiprocessing.reductions import reduce_tensor

logger = logging.getLogger(__name__)


def mark_graph_output(t: torch.Tensor) -> torch.Tensor:
    """Call this immediately after a reduce-overhead compiled fn returns.
    PT2 compiled graphs sometimes pool memory in ways that shouldn't cross IPC."""
    if isinstance(t, torch.Tensor):
        t._from_graph_pool = True
    return t


def _is_ipc_received(t: torch.Tensor) -> bool:
    """True if `t` sits on a CUDA storage that was itself received via IPC
    from another process (and therefore can't be re-shared as-is), OR if
    it was pooled by a compiled graph."""
    if getattr(t, "_from_graph_pool", False):
        return True

    if not t.is_cuda:
        return False
        
    try:
        reduce_tensor(t)
        return False
    except RuntimeError as e:
        if "received from another process" in str(e):
            return True
        raise


def _module_needs_clone(module: nn.Module) -> bool:
    return any(
        _is_ipc_received(t)
        for t in (*module.parameters(recurse=True), *module.buffers(recurse=True))
    )


def _deepcopy_without_hooks(module: nn.Module) -> nn.Module:
    """deepcopy(module) will choke if any hook closure holds state that
    isn't deepcopy-able (a Queue, a Channel, a CUDA event, a lock...).
    Hooks are meaningless once the module crosses a process boundary
    anyway, so strip *all* hook registries for the duration of the copy
    and restore them on the original afterward. We iterate by attribute
    name rather than hardcoding _forward_hooks/_forward_pre_hooks/
    _backward_hooks, since torch has added more hook dicts over time
    (_forward_hooks_with_kwargs, _backward_pre_hooks, etc.) and any of
    them can hold non-copyable closures."""
    saved = []
    for m in module.modules():
        hook_attrs = {
            name: val for name, val in vars(m).items()
            if "hook" in name and isinstance(val, OrderedDict)
        }
        saved.append((m, hook_attrs))
        for name in hook_attrs:
            setattr(m, name, OrderedDict())

    try:
        return copy.deepcopy(module)
    except Exception:
        logger.error(
            "deepcopy failed while sanitizing module %r for IPC even after "
            "stripping hooks -- likely a non-hook attribute holding a "
            "non-copyable object (lock, open handle, CUDA event, etc.)",
            type(module).__name__,
            exc_info=True,
        )
        raise
    finally:
        for m, hook_attrs in saved:
            for name, val in hook_attrs.items():
                setattr(m, name, val)


def sanitize_for_ipc(obj):
    """
    Recursively return a version of `obj` safe to put on a
    torch.multiprocessing Pipe/Queue, cloning only when necessary.

      - torch.Tensor -> obj.clone() if IPC-received OR compiled-graph output
      - nn.Module    -> deepcopy(obj) if any param/buffer needs cloning
      - dict/list/tuple -> rebuilt recursively (payloads are usually dicts
        of tensors)
      - anything else -> returned unchanged
    """
    if obj is None:
        return None

    if isinstance(obj, torch.Tensor):
        return obj.clone() if _is_ipc_received(obj) else obj

    if isinstance(obj, nn.Module):
        return _deepcopy_without_hooks(obj) if _module_needs_clone(obj) else obj

    if isinstance(obj, dict):
        return {k: sanitize_for_ipc(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        cloned = [sanitize_for_ipc(v) for v in obj]
        return type(obj)(cloned) if isinstance(obj, tuple) else cloned

    return obj