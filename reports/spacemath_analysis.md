# `positronixlab.spacemath` — Open Issues Analysis

> **Status (2026-11-27 — `v0.1.3` / `5740a0c` + working tree):** **44 passed, 9 xfailed, 0 failed** (`tests/spacemath/`).
> **Solved issues archive:** Resolved v2 bugs (#1-v2, #6-`is_linear`, #7, #16, v0.1.3 A10a–e, stale 3D test & exports) moved to [solved/spacemath_solved.md](file:///home/cesare/Projects/PositronixLab/reports/solved/spacemath_solved.md).

---

## 1. Open Bugs & Defects

### High / Medium Severity (Active v2 Primitives)
1. **`infer_shape` has a stray `self` parameter & mismatched `x_in` typing** (`common.py:51-56`, xfail `test_spacemath_extended.py:149`): Calling `infer_shape(x_in, x_out)` raises `TypeError`.
2. **`loss_layer_alignment` crashes on empty sequence & returns `sqrt(eps)` on single-layer 3D Euclidean tensor** (`losses.py:116-126`, xfail `test_spacemath_extended.py:346`).
3. **`loss_batch_repulsion` silently ignores `mixed` for 2D inputs** (`losses.py:45-55`, xfail `test_spacemath_extended.py:352`): `if mixed is not None:` is nested inside `if centers_on.ndim == 3:`.
4. **`mask_cluster_membership` fails for single non-zero cluster ID & non-contiguous IDs** (`weighting.py:74-81`, xfail `test_spacemath_extended.py:300`): Uses `max()+1` instead of unique non-negative cluster count.
5. **`spow` / `spow_` defaults to `force_zero=True` → `inf` on negative powers** (`common.py:32-40`, xfail `test_spacemath_extended.py:161`).
6. **Euclidean `distance_elementwise(x, x)` returns `sqrt(eps)=1e-4` instead of `0.0`** (`space_type.py:70`, xfail `test_spacemath_extended.py:202`): Inconsistent with `distance_pairwise` (`0.0`).
7. **Sign-preserving scorers & silent Softmax override (v0.1.3 B4)** (`scoring.py:19-20, 55-86, 139`):
   - `score_sign_preserving_*` with ReLU collapses negative branch to `[0, 1]` with no warning.
   - `activations_to_scores` overrides `score_method` when `isinstance(activation_fn, nn.Softmax)` even when `apply_activation=False`.
   - `score_normalize_by_threshold` multiplies by mask (`inf * 0 = NaN`) rather than `torch.where`.
8. **`is_layer` still matches `nn.Bilinear`** (`common.py:63-73`): Module-string check includes `nn.Bilinear` and misses custom `nn.Linear` subclasses.

### Low Severity / Edge Cases (Active v2 Primitives)
9. **`reduce_conv_locality` signed `max_pool2d` & missing 5D `Conv3d`** (`reducers.py:37`): `max_pool2d` runs before `.abs()` (weakest negative response wins on pre-ReLU maps); crashes on 5D `Conv3d`.
10. **`stats_update_` with non-zero `threshold`** (`weighting.py:28-29`): Clamps active/inactive scores at `0.0`.
11. **`sigreg_weak` / `sigreg_strong` return `NaN` on empty `(0, D)` input** (`sigreg.py:23, 36`).

### Legacy `SpaceMath` (`spacemath.py` — Still Imported by `modularspace`, `modularobserver`, `tasks/compositionality`)
12. **`SpaceMath.normalize_by_threshold` negative-row explosion** (`spacemath.py:134`): `scores_on = scores * idxs_on` still unshifted.
13. **`SpaceMath.activations_to_scores` `LayerType.CONV` unpacks `(B, 1, k)` into `q, scores`** (`spacemath.py:111-114`, xfail `test_spacemath_extended.py:446`).
14. **Mismatched kernel indices in `SpaceMath.get_quantile_k_kernels_per_sample`** (`spacemath.py:259-263, 273`, xfail `test_spacemath_extended.py:453`).
15. **`SpaceMath.activations_to_scores` computes quantile on `act(xd)` but normalizes raw `xd`** (`spacemath.py:122-127`).
16. **`SpaceMath.distance` Euclidean pairwise `.squeeze()` collapses singleton dims** (`spacemath.py:85`, xfail `test_spacemath_extended.py:468`).

---

## 2. Open Design, Performance & Hygiene Issues

- **Missing exports & helpers:** `spacemath/__init__.py` does not export the 4 new `reduce_seq_*` functions, `DEFAULT_REDUCER_BY_LAYER`, or `common.py` helpers (`torch_compile`, `is_linear`, etc.). `losses.py` lacks `get_loss(name)`. `sigreg(method=...)` is case-sensitive.
- **CPU `torch.compile` latency (v0.1.3 D5):** `POSITRONIXLAB_COMPILE=0` opt-out exists, but compilation still runs by default on CPU (where compile overhead dominates tiny layers).
- **Import-time global side effects (`common.py:15, 41-44`):** Mutates `torch.set_float32_matmul_precision('high')` and monkey-patches `torch.Tensor.{zclamp, zclamp_, spow, spow_}`.
- **Overly permissive `SpaceType.normalize_name` (`space_type.py:22-25`):** Uses `.startswith("cos")` / `.startswith("euc")`.
- **Hygiene:** `print()` calls in `common.py:147, 161, 182, 185-186` and `spacemath.py:155, 159`; Italian comments in `sigreg.py:25, 36`; redundant `int(round(...))` in `scoring.py:49`.
- **Deprecate `spacemath.py`:** Move `LayerType` / `ActivationType` out of legacy `spacemath.py` so `modularobserver` and `layerspace` can drop legacy coupling.
