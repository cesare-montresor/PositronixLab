
import contextlib
import dataclasses
import hashlib
import json
import logging
import multiprocessing as mp
import os
import pickle
import re
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

debug = False

last_session_path = None
_session_paths: dict[tuple[str, str, str | None], str] = {}

_SUPPRESS_LOCK = threading.RLock()
_SUPPRESS_STATE: dict[int, tuple[int, int, int]] = {}  # fd -> (depth, saved_fd, devnull_fd)
_UNSET = object()


logger = logging.getLogger(__name__)

def process_start_spawn(force=False):
    current = mp.get_start_method(allow_none=True)
    if current not in ('spawn', None):
        warnings.warn(
            f"Expected spawn start method, got '{current}'. CUDA may be unreliable.",
            stacklevel=2,
        )
    elif current is None:
        mp.set_start_method('spawn', force=force)

@contextlib.contextmanager
def suppress_output(stderr: bool = True):
    """Thread-safe, re-entrant C-level file descriptor suppressor.

    Args:
        stderr: If True, suppresses fd 2 (stderr); if False, suppresses fd 1 (stdout).
    """
    fd = 2 if stderr else 1
    stream = sys.stderr if fd == 2 else sys.stdout
    with _SUPPRESS_LOCK:
        depth, saved_fd, devnull_fd = _SUPPRESS_STATE.get(fd, (0, -1, -1))
        if depth == 0:
            stream.flush()
            saved_fd = os.dup(fd)
            try:
                devnull_fd = os.open(os.devnull, os.O_WRONLY)
            except BaseException:
                os.close(saved_fd)
                raise
            try:
                os.dup2(devnull_fd, fd)
            except BaseException:
                os.close(devnull_fd)
                os.close(saved_fd)
                raise
        _SUPPRESS_STATE[fd] = (depth + 1, saved_fd, devnull_fd)
    try:
        yield
    finally:
        with _SUPPRESS_LOCK:
            state = _SUPPRESS_STATE.get(fd)
            if state is not None:
                depth, saved_fd, devnull_fd = state
                if depth == 1:
                    stream.flush()
                    os.dup2(saved_fd, fd)
                    os.close(devnull_fd)
                    os.close(saved_fd)
                    del _SUPPRESS_STATE[fd]
                else:
                    _SUPPRESS_STATE[fd] = (depth - 1, saved_fd, devnull_fd)

def session_path(
    name: str,
    uid: Any | None = None,
    createDirectory: bool = True,
    force: bool = False,
    basedir: str | os.PathLike[str] | None = None,
) -> str:
    """Return (and optionally create) a session output directory cached per ``(basedir, name, uid)``.

    Args:
        name: Logical session/experiment name (sanitized to remove path separators).
        uid: Optional explicit unique ID; defaults to a timestamp when omitted.
        createDirectory: Create the directory on disk if it does not exist.
        force: Bypass the cache and generate a new session path.
        basedir: Base output directory (defaults to ``$POSITRONIXLAB_OUTPUT_DIR`` or ``output``).
    """
    global last_session_path
    base = str(basedir) if basedir is not None else os.environ.get("POSITRONIXLAB_OUTPUT_DIR", "output")
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip()).strip("._") or "session"
    uid_key = str(uid) if uid is not None else None
    cache_key = (base, safe_name, uid_key)

    if not force and cache_key in _session_paths:
        path = _session_paths[cache_key]
        if createDirectory:
            os.makedirs(path, exist_ok=True)
        last_session_path = path
        return path

    resolved_uid = uid_key if uid_key is not None else str(time.time()).replace(".", "_")
    dir_path = Path(base) / f"{safe_name}_{resolved_uid}"
    path = f"{dir_path.as_posix()}/"
    _session_paths[cache_key] = path
    last_session_path = path
    if createDirectory:
        os.makedirs(path, exist_ok=True)
    return path

def camel_to_kebab(text):
    # Find transitions from lowercase/number to uppercase
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1-\2', text)
    # Find transitions from lowercase to uppercase/number
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1-\2', s1)
    return s2.lower()

def camel_to_snake(text):
    # Find transitions from lowercase/number to uppercase
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
    # Find transitions from lowercase to uppercase/number
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
    return s2.lower()


def get_thread_name():
    thread = threading.current_thread()

    thread_name = thread.name.lower()
    target_name = re.compile(r'\((.+)\)').search(thread_name)
    if target_name is not None:
        thread_name = target_name.group(1)
    
    thread_name = thread_name.replace('(', '').replace(')', '').replace(' ', '_').replace('-', '_')

    return thread_name

def assert_tensor(tensor:torch.Tensor, name="tensor"):
    valid = is_valid_tensor(tensor)
    if valid is not True:
        import traceback
        raise ValueError(f"Invalid tensor: {name}, reason: {valid}", tensor, traceback.format_stack())

def is_valid_tensor(tensor:torch.Tensor):
    if tensor is None:
        return "none"
    if not isinstance(tensor, torch.Tensor):
        return "not_tensor"
    if tensor.numel() == 0:
        return "empty"
    if torch.isnan(tensor).any().item():
        return "nan"
    if torch.isinf(tensor).any().item():
        return "inf"
    return True


def seed_everything(seed: int):
    import os
    import random

    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    

def check_tensor(tensor, name="tensor", debug = debug):
    if not debug:
        return
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"NaN detected in {name}")
        info_tensor(tensor, name)
        raise Exception("NaN detected")
    
def info_tensor(tensor, name="tensor", threshold=1e-10, debug = debug):
    if not debug:
        return
    print("-----")
    print(f"Tensor: {name}")
    print(f"Shape: {tensor.shape}")
    print(f"Min: {tensor.min()}")
    print(f"Max: {tensor.max()}")
    print(f"Mean: {tensor.mean()}")
    if torch.isnan(tensor).any().item():
        print("NaNs!")
    if torch.isinf(tensor).any().item():
        print("Infs!")
    near_zero_count = (torch.abs(tensor) < threshold).sum()
    if near_zero_count > 0:
        print(f"Zeros: {near_zero_count}")
    print("=====")

def module_name(module):
    return module.__class__.__module__

def class_name(module):
    return module.__class__.__name__


def truth_table_from_cm(confusion_matrix:np.ndarray):
    truth_table = []
    for i in range(len(confusion_matrix)):
        tp = confusion_matrix[i, i] # True Positives: diagonal element
        fp = np.sum(confusion_matrix[:, i]) - tp # False Positives: sum of column i, excluding diagonal
        fn = np.sum(confusion_matrix[i, :]) - tp # False Negatives: sum of row i, excluding diagonal  
        tn = np.sum(confusion_matrix) - tp - fp - fn # True Negatives: total - tp - fp - fn
        truth_table.append(np.array((tp, fp, fn, tn)))
    return  np.array(truth_table)

def f1s_from_truth_table(tt:np.ndarray):
    f1s = []
    for tp, fp, fn, _tn in tt:
        f1 = (2 * tp) / ((2 * tp) + fp + fn) if ((2 * tp) + fp + fn) > 0 else 0
        f1s.append(f1)
    return np.array(f1s)


def set_best(bests, key, val, min=True):
    # 1. Handle new keys
    if key not in bests:
        bests[key] = val
        return 1, None
    
    # 2. Check for improvement
    if (min and val < bests[key]) or (not min and val > bests[key]):
        
        # Prevent ZeroDivisionError if the old best was exactly 0
        if bests[key] == 0:
            diff = 1.0
        else:
            diff = abs((bests[key] - val) / bests[key])
            
        old_val = bests[key]
        bests[key] = val
        return diff, old_val

    # 3. No improvement
    return -1, bests[key]

def _json_default(obj: Any) -> Any:
    """Deterministic JSON encoder for common ML/Python types (Path, numpy, torch, dataclasses)."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (torch.dtype, torch.device)):
        return str(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=str)
    raise TypeError(f"Object of type {type(obj).__name__} is not deterministically JSON-serializable")


def signature_settings(data: dict, algo: str = "md5") -> str:
    content = json.dumps(data, sort_keys=True, default=_json_default).encode("utf-8")
    sign = hashlib.new(algo, content, usedforsecurity=False).hexdigest()
    return sign



def save_data(path: str, data: Any, **kwargs: Any) -> bool:
    """Save *data* to *path*, dispatching by file extension.

    Supported extensions (case-insensitive): ``.json``, ``.yaml``/``.yml``,
    ``.pkl``/``.pickle``, ``.pt``/``.pth``. Extra ``**kwargs`` are forwarded to
    the underlying serializer (e.g., ``indent``, ``sort_keys``, ``default`` for
    JSON; ``protocol`` for pickle; ``pickle_protocol`` for ``torch.save``).
    """
    path = os.path.abspath(str(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ext = os.path.splitext(path)[1].strip().lower()
    if ext == '.json':
        kwargs.setdefault('indent', 2)
        with open(path, 'w') as f:
            json.dump(data, f, **kwargs)
    elif ext in ('.yaml', '.yml'):
        with open(path, 'w') as f:
            yaml.safe_dump(data, f, **kwargs)
    elif ext in ('.pkl', '.pickle'):
        with open(path, 'wb') as f:
            pickle.dump(data, f, **kwargs)
    elif ext in ('.pt', '.pth'):
        torch.save(data, path, **kwargs)
    else:
        logger.error(f"Unsupported extension: {ext}")
        return False
    return True


def load_data(path: str, *, default: Any = _UNSET, **kwargs: Any) -> Any:
    """Load data from *path*, dispatching by file extension.

    Supported extensions (case-insensitive): ``.json``, ``.yaml``/``.yml``,
    ``.pkl``/``.pickle``, ``.pt``/``.pth``. Extra ``**kwargs`` are forwarded to
    the underlying loader (e.g., ``map_location`` and ``weights_only`` for
    ``torch.load``, which defaults to ``weights_only=True``).

    Args:
        path: File path to load.
        default: Optional fallback value returned if *path* does not exist.
            When omitted, a missing file raises :class:`FileNotFoundError`.
        **kwargs: Loader-specific keyword arguments.

    Raises:
        FileNotFoundError: If *path* does not exist and *default* was not provided.
        ValueError: If the file extension is unsupported.
        TypeError: If unexpected keyword arguments are passed for YAML ``safe_load``.

    Warning:
        Loading ``.pkl``/``.pickle`` files or ``.pt``/``.pth`` files with
        ``weights_only=False`` executes arbitrary pickled code. Only load files
        from trusted sources.
    """
    path = os.path.abspath(str(path))
    if not os.path.exists(path):
        if default is not _UNSET:
            return default
        raise FileNotFoundError(f"Data file not found: {path}")
    ext = os.path.splitext(path)[1].strip().lower()

    if ext == '.json':
        with open(path) as f:
            return json.load(f, **kwargs)
    elif ext in ('.yaml', '.yml'):
        with open(path) as f:
            if 'Loader' in kwargs:
                return yaml.load(f, **kwargs)
            if kwargs:
                raise TypeError(
                    f"Unexpected keyword arguments for yaml.safe_load: {sorted(kwargs)}. "
                    "Pass an explicit Loader=... to use yaml.load()."
                )
            return yaml.safe_load(f)
    elif ext in ('.pkl', '.pickle'):
        with open(path, 'rb') as f:
            return pickle.load(f, **kwargs)
    elif ext in ('.pt', '.pth'):
        kwargs.setdefault('weights_only', True)
        return torch.load(path, **kwargs)
    else:
        raise ValueError(f"Unsupported extension: {ext!r}")

def safe_tensor(val: Any) -> torch.Tensor:
    if isinstance(val, torch.Tensor):
        return torch.atleast_1d(val.detach().cpu())
    
    # 1. Catch ANYTHING that can be represented as a single number 
    # (Python ints/floats, 0-D numpy arrays, numpy scalars, custom metric objects)
    try:
        return torch.tensor([float(val)])
    except (TypeError, ValueError):
        pass
        
    # 2. If it cannot be cast to a single float, it's an array/iterable.
    # Route it safely through numpy to guarantee PyTorch can parse its dimensions.
    try:
        arr = np.asarray(val)
        return torch.atleast_1d(torch.from_numpy(arr))
    except Exception:
        # Absolute fallback
        return torch.atleast_1d(torch.as_tensor(val))



def model_hash(model: torch.nn.Module, algo: str = "sha256") -> str:
    try:
        hasher = hashlib.new(algo, usedforsecurity=False)
    except TypeError:
        hasher = hashlib.new(algo)
    state_dict = model.state_dict()

    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]
        
        # Hash metadata so reshaped tensors or renamed layers don't collide
        hasher.update(key.encode("utf-8"))
        hasher.update(str(tensor.shape).encode("utf-8"))
        hasher.update(str(tensor.dtype).encode("utf-8"))

        # Flatten, move to CPU, and view as uint8 to handle 0-D scalars 
        # and dtypes NumPy doesn't natively support (like bfloat16/float8)
        raw_bytes = (
            tensor.detach()
            .cpu()
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
        hasher.update(raw_bytes)

    return hasher.hexdigest()