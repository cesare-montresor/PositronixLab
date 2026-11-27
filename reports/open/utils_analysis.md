# `positronixlab/utils` — Open Issues Analysis

> **Status (2026-11-27):** **40 passed, 0 xfailed, 0 failed** (`tests/utils/`).
> **Solved issues archive:** Resolved B1–B6, B8, B12, v0.1.3 A7–A9, and `__init__.py` moved to [solved/utils_solved.md](file:///home/cesare/Projects/PositronixLab/reports/solved/utils_solved.md).

---

## 1. Open Bugs & Defects

### Medium Severity

1. **`is_valid_tensor()` returns truthy strings on failure (B7)**
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
