# `positronixlab/utils` — Open Issues Analysis

> **Status (2026-11-27):** **35 passed, 5 xfailed, 0 failed** (`tests/utils/`).
> **Solved issues archive:** Resolved B1, B3 (P0), B6, B8, B12, v0.1.3 A7–A9, and `__init__.py` moved to [solved/utils_solved.md](file:///home/cesare/Projects/PositronixLab/reports/solved/utils_solved.md).

---

## 1. Open Bugs & Defects

### High / Medium Severity

1. **`process_start_spawn(force=True)` ignores `force=True` when non-`spawn` is already set (B2)**
   - **Location:** `positronixlab/utils/utils.py:33-38` (xfail `test_utils.py:71`)
   - **Details:** When current method is `'fork'` or `'forkserver'`, it warns and never calls `mp.set_start_method('spawn', force=True)`.

2. **`check_tensor()` / `info_tensor()` early-bound `debug`, missing forwarding, and wrong Inf message (B4)**
   - **Location:** `positronixlab/utils/utils.py:164-172` (xfails `test_utils.py:300, 311`)
   - **Details:** `debug=debug` bound at import time; `check_tensor` does not forward `debug=debug` to `info_tensor`; reports `"NaN detected"` on `Inf`.

4. **`info_tensor()` crashes on integer, boolean, and empty tensors (B5)**
   - **Location:** `positronixlab/utils/utils.py:176-178` (xfails `test_utils.py:322, 330`)
   - **Details:** `tensor.mean()` fails on int/bool dtypes; `tensor.min()` fails on `numel() == 0`.

5. **`is_valid_tensor()` returns truthy strings on failure (B7)**
   - **Location:** `positronixlab/utils/utils.py:141-147`
   - **Details:** Returns `"none"`, `"nan"`, etc., so `if is_valid_tensor(t):` is always `True`.

### Low Severity

6. **`model_hash()` crashes on non-tensor `state_dict()` entries (B11):** `utils.py:371-395` — fails on `get_extra_state()` entries.
8. **`safe_tensor()` broad `except Exception`, NumPy deprecation, dtype inconsistency, memory aliasing (B10 / v0.1.3 C10):** `utils.py:338-356` — silently converts `"nan"` strings via `float(val)`.
9. **`camel_to_kebab()` / `camel_to_snake()` double separators on mixed inputs (B9):** `utils.py:109-121`.

---

## 2. Open Design & Hygiene Issues

- **Heavy eager imports:** `utils/__init__.py` eagerly imports `torch`, `numpy`, `yaml`, `multiprocessing` even for pure-stdlib helpers.
- **API quirks:** `session_path` camelCase `createDirectory`; `set_best` shadows builtin `min`; `assert_tensor` dumps raw tensor & stack into `ValueError` args; `signature_settings` duplicates `BaseConfig.signature()`.
- **Hygiene (v0.1.3 F):** One-line `if`s (`utils.py:129, 142-146, 165, 172, 179-182`) and `print()` calls in library code (`utils.py:167, 173-183`).
