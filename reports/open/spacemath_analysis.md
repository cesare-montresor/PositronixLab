# `positronixlab.spacemath` — Open Issues Analysis

> **Status (2026-11-27):** **53 passed, 0 xfailed, 0 failed** (`tests/spacemath/`).
> **Solved issues archive:** Resolved all xfail bugs (v2 #1, #2, #3, #6-`is_linear`, #7, #9, #10, #11, #16, legacy #13/#14/#16, v0.1.3 A10a–e, stale 3D test & exports) moved to [solved/spacemath_solved.md](file:///home/cesare/Projects/PositronixLab/reports/solved/spacemath_solved.md).

---

## 1. Open Bugs & Defects

### Medium Severity (Active v2 Primitives)
1. **Sign-preserving scorers & silent Softmax override (v0.1.3 B4)** (`scoring.py:19-20, 55-86, 139`):
   - `score_sign_preserving_*` with ReLU collapses negative branch to `[0, 1]` with no warning.
   - `activations_to_scores` overrides `score_method` when `isinstance(activation_fn, nn.Softmax)` even when `apply_activation=False`.
   - `score_normalize_by_threshold` multiplies by mask (`inf * 0 = NaN`) rather than `torch.where`.
2. **`is_layer` still matches `nn.Bilinear`** (`common.py:63-73`): Module-string check includes `nn.Bilinear` and misses custom `nn.Linear` subclasses.

### Low Severity / Edge Cases (Active v2 Primitives)
3. **`reduce_conv_locality` signed `max_pool2d` & missing 5D `Conv3d`** (`reducers.py:37`): `max_pool2d` runs before `.abs()` (weakest negative response wins on pre-ReLU maps); crashes on 5D `Conv3d`.
4. **`stats_update_` with non-zero `threshold`** (`weighting.py:28-29`): Clamps active/inactive scores at `0.0`.
5. **`sigreg_weak` / `sigreg_strong` return `NaN` on empty `(0, D)` input** (`sigreg.py:23, 36`).

### Legacy `SpaceMath` (`spacemath.py` — Still Imported by `modularspace`, `modularobserver`, `tasks/compositionality`)
6. **`SpaceMath.normalize_by_threshold` negative-row explosion** (`spacemath.py:134`): `scores_on = scores * idxs_on` still unshifted.
7. **`SpaceMath.activations_to_scores` computes quantile on `act(xd)` but normalizes raw `xd`** (`spacemath.py:122-127`).

---

## 2. Open Design, Performance & Hygiene Issues

- **Missing exports & helpers:** `spacemath/__init__.py` does not export the 4 new `reduce_seq_*` functions, `DEFAULT_REDUCER_BY_LAYER`, or `common.py` helpers (`torch_compile`, `is_linear`, etc.). `losses.py` lacks `get_loss(name)`. `sigreg(method=...)` is case-sensitive.
- **CPU `torch.compile` latency (v0.1.3 D5):** `POSITRONIXLAB_COMPILE=0` opt-out exists, but compilation still runs by default on CPU (where compile overhead dominates tiny layers).
- **Import-time global side effects (`common.py:15, 41-44`):** Mutates `torch.set_float32_matmul_precision('high')` and monkey-patches `torch.Tensor.{zclamp, zclamp_, spow, spow_}`.
- **Overly permissive `SpaceType.normalize_name` (`space_type.py:22-25`):** Uses `.startswith("cos")` / `.startswith("euc")`.
- **Hygiene:** `print()` calls in `common.py:147, 161, 182, 185-186` and `spacemath.py:155, 159`; Italian comments in `sigreg.py:25, 36`; redundant `int(round(...))` in `scoring.py:49`.
- **Deprecate `spacemath.py`:** Move `LayerType` / `ActivationType` out of legacy `spacemath.py` so `modularobserver` and `layerspace` can drop legacy coupling.
