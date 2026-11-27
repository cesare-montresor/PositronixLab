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
| **#9 (P0, 2026-11-27)** | `loss_batch_repulsion` ignored `mixed` for 2D inputs | Filtered `centers_on[mixed > 0]` for 2D inputs (`losses.py:46-52`); xfail removed and passing. |
| **#10 (P0, 2026-11-27)** | `mask_cluster_membership` used `max()+1` instead of unique non-negative cluster count | Uses `torch.unique(lab[lab >= 0])` + `torch.searchsorted` (`weighting.py:70-86`); xfail removed and passing. |
| **#11 (P0, 2026-11-27)** | `spow`/`spow_` defaulted to `force_zero=True`, producing `inf` for negative powers near zero | Changed default to `force_zero=False` (`common.py:36, 42`); xfail removed and passing. |
| **#1 (P2, 2026-11-27)** | `infer_shape` had stray `self` parameter | Removed stray `self` (`common.py:56`); xfail removed and passing. |
| **#2 (P2, 2026-11-27)** | `loss_layer_alignment([])` indexed `[0]` on empty sequence | Added `if len(centers_per_layer) == 0: return torch.tensor(0.0)` (`losses.py:117-118`); xfail removed and passing. |
| **#3 (P2, 2026-11-27)** | Euclidean `distance_elementwise(x, x)` returned `sqrt(eps)=1e-4` instead of `0.0` | Replaced with grad-safe double-`torch.where` Euclidean norm (`space_type.py:69-72`); xfail removed and passing. |
| **#13 (Legacy, 2026-11-27)** | `SpaceMath.activations_to_scores` `LayerType.CONV` unpacked `(B, 1, k)` into `q, scores` | Computed per-kernel scores via `rank_kernels_by_activation_per_sample` and quantile threshold `q` (`spacemath.py:107-113`); xfail removed and passing. |
| **#14 (Legacy, 2026-11-27)** | `SpaceMath.get_quantile_k_kernels_per_sample` indexed `sorted_indices` with unsorted mask positions | Filtered `sorted_indices[b]` via `selected_mask[b, order]` (`spacemath.py:258-263`); xfail removed and passing. |
| **#16 (Legacy, 2026-11-27)** | `SpaceMath.distance` Euclidean pairwise `.squeeze()` collapsed `(1, N)` to `(N,)` | Removed `.squeeze()` (`spacemath.py:81`); xfail removed and passing. |
