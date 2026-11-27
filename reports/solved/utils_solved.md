# `positronixlab/utils` — Solved Issues Archive

> **Archived:** 2026-11-27 (`v0.1.3` / `5740a0c` + working tree).
> **Active open issues:** See [utils_analysis.md](file:///home/cesare/Projects/PositronixLab/reports/utils_analysis.md).

---

## Resolved Bugs & Improvements

| ID | Summary | Resolution Evidence |
| :--- | :--- | :--- |
| **B1 / B12 / v0.1.3 A9a–b, A9e** | `load_data()` crashed on `.yaml` (PyYAML 6), rejected `.yml`/uppercase extensions, returned ambiguous `None`, dropped YAML kwargs, used root `logging.error` | Fixed in `utils.py:256-336`: defaults to `yaml.safe_load(f)`, lowercases extensions, supports `.yml`, uses `logger.error`, raises `FileNotFoundError`/`ValueError` (with optional `default=`), validates unexpected kwargs, documents `.pkl` trust warnings. Tested at `test_utils.py:456-504`. |
| **B6 / v0.1.3 A7** | `suppress_output()` was racy across threads and failed to flush Python stdio buffers | Fixed in `utils.py:26-27, 40-69`: `_SUPPRESS_LOCK`, ref-counted nesting depth, and `stream.flush()` on entry/exit. Tested at `test_utils.py:124-149`. |
| **B8 / v0.1.3 A8** | `session_path()` global cache ignored new `name`/`uid` and skipped `makedirs` | Fixed in `utils.py:24, 71-107`: caches per `(base, safe_name, uid_key)` in `_session_paths`, sanitizes `name`, supports `basedir`/`POSITRONIXLAB_OUTPUT_DIR`, creates dir on cache hit. Tested at `test_utils.py:157-184`. |
| **v0.1.3 A9c** | `signature_settings()` crashed on `Path`/NumPy/`dtype`/dataclasses and failed MD5 under FIPS | Fixed in `utils.py:232-252`: added `_json_default` encoder and `usedforsecurity=False`. |
| **Exports** | Empty `utils/__init__.py` | Populated (`utils/__init__.py:1-47`) exporting all 21 functions with `__all__`. |
| **Error-path & FIPS (2026-11-27)** | `suppress_output()` `KeyError`/FD leak if `os.open`/`os.dup2` raised; `model_hash()` missing `usedforsecurity=False` | Guarded `os.open`/`os.dup2` cleanup & `_SUPPRESS_STATE.get(fd)` (`utils.py:49-78`); added `usedforsecurity=False` fallback in `model_hash()` (`utils.py:371-375`). |
