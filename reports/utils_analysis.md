# `positronixlab/utils` — Module Analysis

## 1. Overview & Refactor Status

`positronixlab/utils` is a lightweight, single-file utility package collecting general-purpose helpers used across `positronixlab` (multiprocessing startup, C-level FD suppression, session paths, string/thread formatting, tensor validation/debugging, seeding, confusion-matrix metrics, best-metric tracking, deterministic hashing, and multi-format I/O).

| File | Role | Status |
|---|---|---|
| [`__init__.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/utils/__init__.py) | Empty package marker (0 bytes; exports nothing). | Minimal / Unrefactored |
| [`utils.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/utils/utils.py) | 17 standalone utility functions plus module globals `debug`, `last_session_path`, and unused `logger`. | Active (Accumulated helpers; no recent refactor) |
| `tests/` | Previously empty; now contains [`test_utils.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/utils/tests/test_utils.py). | Added |
| `docs/` | [`UTILS_SYSTEM.md`](file:///home/cesare/Projects/PositronixLab/positronixlab/utils/docs/UTILS_SYSTEM.md) and [`UTILS_USAGE.md`](file:///home/cesare/Projects/PositronixLab/positronixlab/utils/docs/UTILS_USAGE.md). | Added |
| `examples/` | [`example_utils.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/utils/examples/example_utils.py). | Added |

### Active Callers in the Repository
Only a small subset of functions is currently imported by active code:
- `suppress_output`: [`modularcluster/modularcluster_kmeans.py:17`](file:///home/cesare/Projects/PositronixLab/positronixlab/modularcluster/modularcluster_kmeans.py#L17), [`modularcluster/modularcluster_kmeans_stream.py:13`](file:///home/cesare/Projects/PositronixLab/positronixlab/modularcluster/modularcluster_kmeans_stream.py#L13)
- `load_data`: [`modularui/examples/example_ui.py:17`](file:///home/cesare/Projects/PositronixLab/positronixlab/modularui/examples/example_ui.py#L17), [`tasks/compositionality/examples/example_atlas.py:40`](file:///home/cesare/Projects/PositronixLab/positronixlab/tasks/compositionality/examples/example_atlas.py#L40)
- Legacy callers:
  - [`deeplearning/deeplearning_legacy.py:17`](file:///home/cesare/Projects/PositronixLab/positronixlab/deeplearning/deeplearning_legacy.py#L17) (`set_best`)
  - [`spacemath/legacy/spaceloss.py:8`](file:///home/cesare/Projects/PositronixLab/positronixlab/spacemath/legacy/spaceloss.py#L8) (`assert_tensor`, `info_tensor`, and removed `is_conv`)

---

## 2. Bugs & Defects

### B1. `load_data()` crashes unconditionally on `.yaml` files (High)
- **Location:** [`utils.py:217-219`](file:///home/cesare/Projects/PositronixLab/positronixlab/utils/utils.py#L217-L219)
- **Root Cause:** Calls `yaml.load(f)` without specifying a `Loader` argument. PyYAML >= 6.0 (6.0.3 in `.venv`) removed the default loader for security reasons, so `yaml.load(f)` always raises `TypeError: load() missing 1 required positional argument: 'Loader'`.
- **Impact:** Any `.yaml` file written by `save_data()` cannot be read back with `load_data()`. Should use `yaml.safe_load(f)`.

### B2. `process_start_spawn(force=True)` ignores `force=True` when a non-`spawn` method is already set (High)
- **Location:** [`utils.py:25-30`](file:///home/cesare/Projects/PositronixLab/positronixlab/utils/utils.py#L25-L30)
- **Root Cause:**
  ```python
  if current not in ('spawn', None):
      warnings.warn(...)
  elif current is None:
      mp.set_start_method('spawn', force=force)
  ```
  When `current` is `'fork'` or `'forkserver'`, execution enters the first branch, emits a warning, and **never** calls `mp.set_start_method('spawn', force=True)`. Conversely, when `current is None`, `force` has no effect because no start method is set yet. Additionally, `warnings.warn()` defaults to `UserWarning`, which `positronixlab.logger` globally suppresses.
- **Impact:** Callers passing `force=True` to enforce `'spawn'` for CUDA safety silently remain on `'fork'`.

### B3. `seed_everything()` sets contradictory cuDNN flags and omits multi-GPU seeding (Medium)
- **Location:** [`utils.py:97-108`](file:///home/cesare/Projects/PositronixLab/positronixlab/utils/utils.py#L97-L108)
- **Root Cause:**
  1. Sets `torch.backends.cudnn.deterministic = True` (`line 107`) immediately followed by `torch.backends.cudnn.benchmark = True` (`line 108`). PyTorch's reproducibility guidelines explicitly require `cudnn.benchmark = False` for deterministic cuDNN kernel selection.
  2. Calls `torch.cuda.manual_seed(seed)` (`line 106`) instead of `torch.cuda.manual_seed_all(seed)`, leaving non-default CUDA devices unseeded.
  3. Sets `os.environ['PYTHONHASHSEED'] = str(seed)` (`line 103`), which has no effect on `hash()` randomization in the already-running Python interpreter (only affects child processes).
- **Impact:** Non-deterministic GPU training runs despite calling `seed_everything()`.

### B4. `check_tensor()` and `info_tensor()` suffer from early-bound `debug` default and missing forwarding (Medium)
- **Location:** [`utils.py:111-118`](file:///home/cesare/Projects/PositronixLab/positronixlab/utils/utils.py#L111-L118)
- **Root Cause:**
  1. Both signatures bind `debug = debug` at import time (locking the default to `False`). Mutating `utils.debug = True` at runtime has **no effect** on calls that omit `debug`.
  2. Inside `check_tensor()`, line 115 calls `info_tensor(tensor, name)` without passing `debug=debug`. Even when a caller explicitly runs `check_tensor(t, debug=True)`, `info_tensor` receives its bound default `debug=False` and prints nothing.
  3. When `tensor` contains `Inf` (not `NaN`), `check_tensor()` still prints `"NaN detected in ..."` and raises bare `Exception("NaN detected")` (`lines 114, 116`).
- **Impact:** Runtime debugging toggle `utils.debug = True` is broken, and `check_tensor(..., debug=True)` fails to print the diagnostic summary before raising.

### B5. `info_tensor()` crashes on integer, boolean, and empty tensors (Medium)
- **Location:** [`utils.py:123-125`](file:///home/cesare/Projects/PositronixLab/positronixlab/utils/utils.py#L123-L125)
- **Root Cause:**
  - `tensor.mean()` (`line 125`) raises `RuntimeError: mean(): could not infer output dtype. Input dtype must be either a floating point or complex dtype` on integer/boolean tensors (`int64`, `int32`, `bool`).
  - `tensor.min()` (`line 123`) raises `RuntimeError` when `tensor.numel() == 0`.
- **Impact:** Using `info_tensor()` on label tensors, masks, or empty slices crashes instead of logging diagnostics.

### B6. `suppress_output()` fails to flush Python stdio buffers before redirecting/restoring FDs (Medium)
- **Location:** [`utils.py:32-43`](file:///home/cesare/Projects/PositronixLab/positronixlab/utils/utils.py#L32-L43)
- **Root Cause:** `suppress_output()` redirects C-level file descriptor `1` or `2` via `os.dup2()` without flushing `sys.stdout` / `sys.stderr` before entering or before restoring in `finally:`. When `stdout` is piped or redirected (block-buffered in Python), `print()` calls inside `with suppress_output(stderr=False):` stay in Python's buffer and are flushed *after* fd 1 is restored, leaking the supposedly suppressed text. Additionally, if `os.open()` fails on line 36, `old_stderr` (line 35) is leaked because it was duplicated before the `try:` block.
- **Impact:** `suppress_output(stderr=False)` leaks buffered Python output in non-TTY environments (pipelines, CI, pytest, redirected logs).

### B7. `is_valid_tensor()` returns truthy strings on failure (Medium — API footgun)
- **Location:** [`utils.py:88-94`](file:///home/cesare/Projects/PositronixLab/positronixlab/utils/utils.py#L88-L94)
- **Root Cause:** Returns `True` when valid and non-empty strings (`"none"`, `"not_tensor"`, `"empty"`, `"nan"`, `"inf"`) when invalid. Because non-empty strings are truthy in Python, any caller writing `if is_valid_tensor(t):` will always evaluate the condition as `True`.
- **Impact:** High risk of silently bypassing validation unless callers remember to write `is_valid_tensor(t) is True`.

### B8. `session_path()` global cache silently ignores new `name`/`uid` and skips directory creation (Low)
- **Location:** [`utils.py:45-54`](file:///home/cesare/Projects/PositronixLab/positronixlab/utils/utils.py#L45-L54)
- **Root Cause:** Once `last_session_path` is populated, subsequent calls with a different `name`, `uid`, or `createDirectory=True` (after an initial `createDirectory=False`) return the stale cached path unless `force=True`. The base directory `'output/'` is hardcoded relative to CWD using string concatenation.
- **Impact:** Subtle path collisions in multi-session scripts and test suites.

### B9. `camel_to_kebab()` and `camel_to_snake()` double separators on mixed-delimiter inputs (Low)
- **Location:** [`utils.py:56-68`](file:///home/cesare/Projects/PositronixLab/positronixlab/utils/utils.py#L56-L68)
- **Root Cause:** The first regex `re.sub('(.)([A-Z][a-z]+)', ...)` matches any character `(.)`, including existing `_`, `-`, or spaces. Thus `camel_to_snake("foo_Bar")` produces `"foo__bar"` and `camel_to_kebab("foo-Bar")` produces `"foo--bar"`.
- **Impact:** Malformed identifiers when inputs already contain separators.

### B10. `safe_tensor()` emits NumPy deprecation warning, produces inconsistent dtypes, and aliases NumPy memory (Low)
- **Location:** [`utils.py:229-247`](file:///home/cesare/Projects/PositronixLab/positronixlab/utils/utils.py#L229-L247)
- **Root Cause:**
  - `float(val)` (`line 236`) on a 1-element `ndim > 0` NumPy array (e.g. `np.array([5.0])`) triggers NumPy 1.25+ `DeprecationWarning` (will become a `TypeError` in a future NumPy release).
  - Scalar ints/bools become `torch.float32` via `torch.tensor([float(val)])`, lists of ints become `torch.int64`, and lists of floats become `torch.float64` via `np.asarray`.
  - `torch.from_numpy(arr)` (`line 244`) shares underlying memory with writable NumPy arrays (caller mutations alias) and warns on read-only views.
- **Impact:** Unexpected dtype mismatches and future breakage on 1-element NumPy arrays.

### B11. `model_hash()` crashes on `state_dict()` entries that are not tensors (Low)
- **Location:** [`utils.py:251-276`](file:///home/cesare/Projects/PositronixLab/positronixlab/utils/utils.py#L251-L276)
- **Root Cause:** Iterates over `model.state_dict()` assuming every value is a `torch.Tensor`. Modules implementing `get_extra_state()` (or certain quantized layers) can store arbitrary Python objects or `None` in `state_dict()`, raising `AttributeError` on `tensor.shape`.
- **Impact:** Hashing models with extra state fails instead of handling or serializing non-tensor entries.

### B12. `save_data()` / `load_data()` extension handling, unused `logger`, and ambiguous `None` (Low)
- **Location:** [`utils.py:23, 188-227`](file:///home/cesare/Projects/PositronixLab/positronixlab/utils/utils.py#L188-L227)
- **Root Cause:**
  - Rejects `.yml` (only `.yaml` is handled) and uppercase extensions (`.JSON`, `.YAML`, `.PT`) due to missing `.lower()`.
  - Calls root `logging.error()` (`lines 204, 226`) instead of the module-level `logger` (`line 23`), leaving `logger` completely unused.
  - `load_data()` returns `None` identically for: non-existent file, unsupported extension, and a valid file storing `null`/`None`.
- **Impact:** Confusing I/O failures and inability to distinguish missing files from `None` payloads.

---

## 3. Inconsistencies & Design Issues

1. **Empty `__init__.py`**: [`__init__.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/utils/__init__.py) exports nothing, forcing callers to write `from positronixlab.utils import utils` or `from positronixlab.utils.utils import ...` instead of `from positronixlab.utils import ...`.
2. **Heavy eager imports**: Importing `utils.py` unconditionally imports `torch`, `numpy`, `yaml`, and `multiprocessing`, adding substantial startup latency even for pure-stdlib helpers like `camel_to_snake` or `suppress_output`.
3. **Parameter naming & shadowing**: `session_path` uses camelCase `createDirectory` while all other parameters use `snake_case`; `set_best` shadows the Python builtin `min`.
4. **Duplication with other modules**: `signature_settings()` duplicates `BaseConfig.signature()` in [`positronixlab/pipeline/config.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/pipeline/config.py) (which additionally handles non-JSON types via `default=str`).
5. **`assert_tensor` exception payload**: Passes `(message, tensor, traceback.format_stack())` as 3 separate args to `ValueError` (`line 86`), dumping the entire raw tensor and stack list into the exception representation.
6. **Misleading parameter names in `module_name(module)` / `class_name(module)`**: Passing a class object (`nn.Linear`) returns `'builtins'` / `'type'`; they expect an *instance* (`nn.Linear(2, 2)`).
7. **Broken legacy reference**: [`spacemath/legacy/spaceloss.py:8`](file:///home/cesare/Projects/PositronixLab/positronixlab/spacemath/legacy/spaceloss.py#L8) still imports `is_conv` from `positronixlab.utils.utils`, which no longer exists there.
8. **Missing type annotations & docstrings**: Most functions in `utils.py` lack docstrings and complete type hints.

---

## 4. Opportunities & Recommendations

1. **Fix `load_data()` YAML loading immediately**: Replace `yaml.load(f)` with `yaml.safe_load(f)`, normalize extensions with `.lower()`, support `.yml`, and use `logger.error(...)` instead of root `logging.error(...)`.
2. **Fix `process_start_spawn(force=True)`**: When `force=True` and `current != 'spawn'`, call `mp.set_start_method('spawn', force=True)` (in addition to warning).
3. **Fix `seed_everything()`**: Set `torch.backends.cudnn.benchmark = False` (or expose a `deterministic: bool = True` parameter), call `torch.cuda.manual_seed_all(seed)`, and remove redundant local imports.
4. **Fix `check_tensor()` / `info_tensor()`**: Change default `debug: Optional[bool] = None` (resolving to module-level `debug` at call time), forward `debug=debug` from `check_tensor` to `info_tensor`, guard empty tensors, cast non-float tensors via `.float().mean()`, and raise `ValueError` with the actual NaN/Inf reason.
5. **Fix `suppress_output()` buffering**: Flush `sys.stderr`/`sys.stdout` both before `os.dup2` and inside `finally:` before restoring the original FD; move `os.dup`/`os.open` under proper cleanup guards; optionally support suppressing both streams.
6. **Improve `is_valid_tensor()` API**: Either return a strict `bool` (with a companion `validate_tensor()` returning `Optional[str]`) or clearly deprecate truthiness usage.
7. **Re-export public API in `__init__.py`** (optionally using PEP 562 lazy imports so lightweight helpers do not pay `torch` import cost).
8. **Add `default=str` to `signature_settings()`** to match `BaseConfig.signature()` and support `Path`/`Enum`/NumPy values.
9. **Harden `safe_tensor()` and `model_hash()`**: Check `isinstance(val, np.ndarray)` before `float(val)` in `safe_tensor()`, and guard non-tensor entries in `model_hash()`.

---

## 5. Docs / Tests / Examples Summary

- **Documentation (created):**
  - [`docs/UTILS_SYSTEM.md`](file:///home/cesare/Projects/PositronixLab/positronixlab/utils/docs/UTILS_SYSTEM.md): Architecture, function inventory, dependency profile, global state, and known caveats.
  - [`docs/UTILS_USAGE.md`](file:///home/cesare/Projects/PositronixLab/positronixlab/utils/docs/UTILS_USAGE.md): Practical usage guide and recipes for every utility category.
- **Unit Tests (created):**
  - [`tests/test_utils.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/utils/tests/test_utils.py): Comprehensive CPU-only test suite covering all 17 functions in `utils.py`, including 9 `@pytest.mark.xfail(strict=True)` regression tests pinning the confirmed bugs (B1, B2, B3, B4, B5, B6, B12).
- **Runnable Example (created):**
  - [`examples/example_utils.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/utils/examples/example_utils.py): Self-contained CPU walkthrough demonstrating tensor validation, seeding, metrics, best-value tracking, deterministic hashing, I/O, and FD suppression.
