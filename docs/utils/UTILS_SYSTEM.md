# Utils System Architecture

## 1. Purpose

`positronixlab/utils` is the shared low-level utility module for `positronixlab`. It provides standalone helper functions for process/FD management, session paths, string and thread naming, tensor validation/inspection, RNG seeding, classification metrics, metric tracking, deterministic hashing, and multi-format data serialization.

## 2. Module Layout

```
positronixlab/utils/
├── __init__.py          # Empty package marker (import from positronixlab.utils.utils)
├── utils.py             # All 17 utility functions and module-level state
├── analysis.md          # Code review, bug catalog, and recommendations
├── docs/
│   ├── UTILS_SYSTEM.md  # Architecture and technical reference (this file)
│   └── UTILS_USAGE.md   # Practical usage guide and examples
├── tests/
│   └── test_utils.py    # Pytest unit test suite
└── examples/
    └── example_utils.py # Self-contained CPU example script
```

## 3. Function Inventory

All functions live in [`positronixlab/utils/utils.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/utils/utils.py):

| Category | Function | Signature | Notes |
|---|---|---|---|
| **Process & I/O Control** | `process_start_spawn` | `(force=False)` | Sets `multiprocessing` start method to `'spawn'` if unset; warns if non-`spawn`. |
| | `suppress_output` | `(stderr: bool = True)` | Context manager redirecting C-level fd `2` (stderr) or `1` (stdout) to `os.devnull`. |
| **Paths & Naming** | `session_path` | `(name, uid=None, createDirectory=True, force=False)` | Generates and caches `output/<name>_<uid>/`. |
| | `camel_to_kebab` | `(text)` | Converts `CamelCase` to `kebab-case`. |
| | `camel_to_snake` | `(text)` | Converts `CamelCase` to `snake_case`. |
| | `get_thread_name` | `()` | Returns normalized lowercase current thread name (extracts `(...)` target if present). |
| | `module_name` | `(module)` | Returns `module.__class__.__module__` for an object instance. |
| | `class_name` | `(module)` | Returns `module.__class__.__name__` for an object instance. |
| **Tensor Validation & Debug** | `is_valid_tensor` | `(tensor: torch.Tensor)` | Returns `True` if valid, else reason string (`"none"`, `"not_tensor"`, `"empty"`, `"nan"`, `"inf"`). |
| | `assert_tensor` | `(tensor: torch.Tensor, name="tensor")` | Raises `ValueError` if `is_valid_tensor(tensor) is not True`. |
| | `check_tensor` | `(tensor, name="tensor", debug=debug)` | Raises `Exception` on NaN/Inf when `debug=True`. |
| | `info_tensor` | `(tensor, name="tensor", threshold=1e-10, debug=debug)` | Prints shape, min, max, mean, NaN/Inf flags, and near-zero count when `debug=True`. |
| | `safe_tensor` | `(val: Any) -> torch.Tensor` | Converts tensors, scalars, NumPy arrays, or iterables into a CPU 1D+ `torch.Tensor`. |
| **Reproducibility & Hashing** | `seed_everything` | `(seed: int)` | Seeds Python `random`, `PYTHONHASHSEED`, NumPy, and PyTorch CPU/CUDA RNGs. |
| | `signature_settings` | `(data: dict, algo: str = "md5") -> str` | Deterministic hash of a JSON-serializable dict (`sort_keys=True`). |
| | `model_hash` | `(model: torch.nn.Module, algo: str = "sha256") -> str` | Deterministic hash of all sorted `state_dict()` keys, shapes, dtypes, and raw bytes. |
| **Metrics & Tracking** | `truth_table_from_cm` | `(confusion_matrix: np.ndarray)` | Computes per-class `(tp, fp, fn, tn)` rows from an `N x N` confusion matrix. |
| | `f1s_from_truth_table` | `(tt: np.ndarray)` | Computes per-class F1 scores from a `(N, 4)` truth table. |
| | `set_best` | `(bests, key, val, min=True)` | Tracks best (min or max) value per key in a dict; returns `(rel_improvement, old_val)` or `(-1, current_best)`. |
| **Data Serialization** | `save_data` | `(path: str, data)` | Saves `.json`, `.yaml`, `.pkl`/`.pickle`, or `.pt`/`.pth`; creates parent dirs. |
| | `load_data` | `(path: str)` | Loads `.json`, `.yaml` (currently broken on PyYAML 6+), `.pkl`/`.pickle`, or `.pt`/`.pth` (`weights_only=True`). |

## 4. Module-Level State & Side Effects

`utils.py` defines three module-level globals:
- `debug = False` (`line 18`): Intended as a global debug flag for `check_tensor` and `info_tensor`, but bound early as default argument values at import time (mutating `utils.debug` at runtime does not affect default arguments).
- `last_session_path = None` (`line 20`): Process-global cache mutated by `session_path()`. Not thread-safe.
- `logger = logging.getLogger(__name__)` (`line 23`): Currently unused (`save_data`/`load_data` call root `logging.error`).

### Dependency Profile
Importing `positronixlab.utils.utils` eagerly imports `torch`, `numpy`, `yaml` (PyYAML), `pickle`, `multiprocessing`, `hashlib`, `json`, `re`, and `threading`.

## 5. Operational Caveats

- **Thread safety**: `suppress_output()` mutates process-wide file descriptors `1` or `2` and is not thread-safe. `session_path()` mutates module global `last_session_path` without locking.
- **Python stdio buffering**: `suppress_output()` operates on OS-level file descriptors without flushing `sys.stdout`/`sys.stderr`. Always flush Python streams inside the block (or use it specifically for C/C++ library output such as FAISS).
- **Truthiness of `is_valid_tensor()`**: Always check `is_valid_tensor(t) is True`, never `if is_valid_tensor(t):`, because failure codes are non-empty (truthy) strings.
- **YAML loading**: `load_data()` currently calls `yaml.load(f)` without `Loader`, raising `TypeError` under PyYAML 6+. Use `.json`, `.pkl`, or `.pt` until fixed.
- **PyTorch `weights_only=True`**: `load_data()` loads `.pt`/`.pth` files with `weights_only=True`; only tensors and safe primitives can be deserialized.
