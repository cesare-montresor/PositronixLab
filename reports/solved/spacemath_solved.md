# `positronixlab.spacemath` — Solved Issues Archive

> **Archived:** 2026-11-27 (`v0.1.3` / `5740a0c` + working tree).
> **Active open issues:** See [spacemath_analysis.md](file:///home/cesare/Projects/PositronixLab/reports/spacemath_analysis.md).

---

## Resolved Bugs & Improvements

| ID | Summary | Resolution Evidence |
| :--- | :--- | :--- |
| **#1 (v2)** | Score explosion (`-2e8`) & sign inversion on negative values in `score_normalize_by_threshold` | Fixed in `scoring.py:19` (`(values - threshold + eps) * idxs_on`); passing test `test_spacemath_extended.py:255`. *(Legacy `spacemath.py:134` still open.)* |
| **#6 (`is_linear` / `nn.Identity`)** | `is_linear(nn.Identity())` and `is_layer(nn.Identity())` returned `True` | `is_linear` now uses `isinstance(module, nn.Linear)` (`common.py:78-79`); `is_layer` excludes `nn.Identity` (`common.py:64-65`). *(Note: `is_layer` still matches `nn.Bilinear`.)* |
| **#7** | `contrast_distance_*` vs `contrast_softmax_ce` cosine centroid scale inconsistency | `contrast.py:65` now normalizes `positions` before computing centroids in cosine mode; passing test `test_spacemath_extended.py:379`. |
| **#13 (Conv1d)** | `reduce_conv_locality` lacked 3D `Conv1d` support | 3D `Conv1d` fallback added at `reducers.py:31-32`. *(Signed `max_pool2d` & 5D `Conv3d` remain open.)* |
| **#16 / v0.1.3 A10a** | Misplaced docstring in `clear_torch_compile_cache` | Fixed at `common.py:145`; verified by `test_spacemath_extended.py:131`. |
| **v0.1.3 A10b** | Destructive `rmtree` on user-controlled cache env vars | `_is_safe_cache_dir()` added (`common.py:127-141, 179`), refusing `/`, `/tmp`, `$HOME` and requiring `torchinductor`/`triton` in path. |
| **v0.1.3 A10c** | Wrong `$USER` directory in `clear_torch_compile_cache` | Uses `getpass.getuser()` with `$USER` fallback (`common.py:163-166`). |
| **v0.1.3 A10d / F** | `def compile(func)` shadowed Python builtin; redundant f-strings (`F541`); unused `nn`/`F` in `sigreg.py` | Renamed to `torch_compile` (`common.py:91`, `contrast.py:8`); redundant f-strings and unused `sigreg.py` imports removed. |
| **v0.1.3 A10e** | No way to disable `torch.compile` | Added `POSITRONIXLAB_COMPILE=0\|false\|no\|off` env switch (`common.py:82-88, 105-106`, tested at `test_spacemath_extended.py:141-146`). |
| **Feature** | 3D sequence reducers for Transformers/RNNs | Added `reduce_seq_mean`, `reduce_seq_max`, `reduce_seq_cls`, `reduce_seq_energy`, updated `reduce_linear` to `(B, T, U) -> (B, U)`, and added `DEFAULT_REDUCER_BY_LAYER` entries for `conv1d`/`rnn`/`lstm`/`transformer` (`reducers.py:13-14, 70-126`). |
| **Cleanup (2026-11-27)** | Stale 3D `reduce_linear` test & missing `__init__.py` exports | Updated `tests/spacemath/test_spacemath_extended.py:219` to expect `(4, 5)`; re-exported `reduce_seq_*` and `DEFAULT_REDUCER_BY_LAYER` in `spacemath/__init__.py`. |
