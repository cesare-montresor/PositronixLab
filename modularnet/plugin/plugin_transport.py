"""
ipc_safe.py

One function — `to_sendable(obj)` — that makes a CUDA tensor or nn.Module
safe to put on a torch.multiprocessing Pipe/Queue more than once.

Why this is needed:
Once a CUDA tensor is reconstructed in a process via IPC (which is what
torch.multiprocessing does transparently the first time a CUDA tensor
crosses a process boundary), its underlying storage is permanently flagged
`received_cuda=True` at the C++ level. Trying to share that storage again
-- forwarding it to a third process, or even echoing it back to the
original sender -- raises:

    RuntimeError: Attempted to send CUDA tensor received from another
    process; this is not currently supported. Consider cloning before
    sending.

There's no public flag to read this directly, so `_is_ipc_received` checks
it the same way pickling would: by calling the actual reduction function
torch registers for CUDA tensors and seeing if it raises. This is cheap in
both branches -- torch dedupes/caches shares per-storage, and the guard
raises immediately (before any real IPC work) when it fires.
"""

import copy
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.multiprocessing.reductions import reduce_tensor


def sanitize_for_ipc(obj):
    """
    Return a version of `obj` safe to put on a torch.multiprocessing
    Pipe/Queue, cloning only when actually necessary:

      - torch.Tensor  -> obj.clone()    if its storage was IPC-received
      - nn.Module     -> deepcopy(obj)  if any param/buffer was IPC-received
      - anything else -> returned unchanged
    """
    if obj is None: return None

    if isinstance(obj, nn.Module):
        return _deepcopy_without_hooks(obj) if _module_needs_clone(obj) else obj

    if isinstance(obj, torch.Tensor):
        return obj.clone() if _is_ipc_received(obj) else obj

    return obj



def _is_ipc_received(t: torch.Tensor) -> bool:
    """True if `t` sits on a CUDA storage that was itself received via IPC
    from another process (and therefore can't be re-shared as-is)."""
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
    anyway, so strip them for the duration of the copy and restore them
    on the original afterward."""
    saved = [
        (m, m._forward_hooks, m._forward_pre_hooks, m._backward_hooks)
        for m in module.modules()
    ]
    for m, *_ in saved:
        m._forward_hooks = OrderedDict()
        m._forward_pre_hooks = OrderedDict()
        m._backward_hooks = OrderedDict()
    try:
        return copy.deepcopy(module)
    finally:
        for m, fh, fph, bh in saved:
            m._forward_hooks, m._forward_pre_hooks, m._backward_hooks = fh, fph, bh


def to_sendable(obj):
    """
    Return a version of `obj` safe to put on a torch.multiprocessing
    Pipe/Queue, cloning only when actually necessary:

      - torch.Tensor  -> obj.clone()    if its storage was IPC-received
      - nn.Module     -> deepcopy(obj)  if any param/buffer was IPC-received
      - anything else -> returned unchanged

    Call this on the specific object you're about to put in a message
    (e.g. `result = to_sendable(result)` in respond()), not blindly on an
    entire nested payload -- it doesn't recurse into dicts/dataclasses.
    """
    if isinstance(obj, nn.Module):
        return _deepcopy_without_hooks(obj) if _module_needs_clone(obj) else obj

    if isinstance(obj, torch.Tensor):
        return obj.clone() if _is_ipc_received(obj) else obj

    return obj