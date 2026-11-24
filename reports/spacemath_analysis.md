# `positronixlab.spacemath` — Code Review & Architecture Analysis

## 1. Overview & Refactor Status

`positronixlab/spacemath` provides the mathematical primitives that project neural network Computation Units (CUs) into a decoupled embedding space (see [PARKING LOT PROBLEM.md](docs/PARKING%20LOT%20PROBLEM.md) for the intuitive formulation).

The module is mid-refactor from the monolithic v1 `SpaceMath` class (`spacemath.py`) and removed `spaceloss.py` (archived under `legacy/`) into composable, registry-backed functional primitives:

| File | Role | Status | Primary Consumers |
| :--- | :--- | :--- | :--- |
| `space_type.py` | Geometry (`SpaceType`: `cosine`/`euclidean`), normalization, SIGReg scaling, elementwise & pairwise distances | Active (v2) | `contrast.py`, `losses.py`, `modularspace/{layerspace,space_step}.py` |
| `reducers.py` | Architecture reducers (`linear`, `conv_locality`, `conv_l2`, `conv_mean_abs`, `conv_max`) + `REDUCERS` / `get_reducer` | Active (v2) | `scoring.py` |
| `scoring.py` | Activation-to-score normalization (`quantile`, `top_k`, `sign_preserving_row`, `sign_preserving_batch`, `softmax`) + `activations_to_scores` | Active (v2) | `modularspace/{layerspace,space_step}.py` |
| `weighting.py` | Active/inactive splitting, running `(U, 4)` stats, scope masks (`global`, `cluster`), weighters (`raw`, `inv_freq`, `uniform`) | Active (v2) | `modularspace/{layerspace,layerspace_old,space_step}.py` |
| `contrast.py` | `torch.compile`-accelerated scoped contrastive objectives (`distance_log`, `distance_linear`, `softmax_ce`) returning `ContrastResult` | Active (v2) | `modularspace/{layerspace,space_step}.py` |
| `losses.py` | Auxiliary losses (`loss_centers_repulsion`, `loss_batch_repulsion`, `loss_sigreg`, `loss_sigreg_centers`, `loss_layer_alignment`) | Active (v2) | `modularspace/{layerspace,space_step}.py` |
| `sigreg.py` | Isotropic Gaussian regularization (`sigreg`, `sigreg_weak`, `sigreg_strong`) | Active (v2) | `losses.py` |
| `common.py` | Tensor monkey-patches (`zclamp`, `spow`), module-type predicates, `torch.compile` decorators, cache clearing | Active (shared) | `space_type.py`, `contrast.py`, `spacemath.py`, `modularobserver/`, `modularspace/` |
| `spacemath.py` | Legacy v1 `SpaceMath`, `LayerType`, `ActivationType` | Legacy (still imported) | `modularspace/{modularspace,layerspace,layerspace_old}.py`, `modularobserver/{modelmapper,layerobserver}.py`, `tasks/compositionality/atlas_check.py` |

**Refactor status**: The v2 primitive pipeline (`reducers -> scoring -> weighting -> contrast -> losses`) is cleanly decoupled and already wired into `modularspace/space_step.py` and `modularspace/layerspace.py`. However, `spacemath.py` cannot be removed yet because `ActivationType`, `LayerType`, and `SpaceMath` are still imported across `modularspace`, `modularobserver`, and `tasks/compositionality`.

---

## 2. Bugs & Defects

### Critical / High Severity

1. **Score explosion & sign inversion on negative values in `score_normalize_by_threshold` and `SpaceMath.normalize_by_threshold`**
   - **Files**: `scoring.py:10-24`, `spacemath.py:132-139`
   - **Root cause**: `scores_on = values * idxs_on` does **not** shift by `threshold` (unlike `scores_off = ((values - threshold) - eps) * idxs_off`). When all `values` in a row are negative (e.g., `[-10, -5, -2, -1]` with `threshold = -2`), `scores_on` is `[0, 0, -2, -1]`, so `scores_on.amax(dim=1)` picks the masked-out `0.0` entries, making `scores_max = eps` (`1e-8`). Dividing yields `[-1.0, -0.375, -2e8, -1e8]`. Even when only some `values >= threshold` are negative, they remain negative instead of mapping to `[0, 1]`.
   - **Impact**: Breaks the `[-1, 1]` score contract, explodes magnitudes to `~1e8`, and flips active units to negative so `split_active_group(..., threshold=0.0)` misclassifies them as inactive. Affects pre-activation linear layers, LeakyReLU/GELU/SiLU/Tanh outputs, and any row with negative threshold.

2. **`SpaceMath.activations_to_scores` for `LayerType.CONV` crashes on every batch size != 2 (and corrupts batch size 2)**
   - **File**: `spacemath.py:111-114`
   - **Root cause**: `cls.get_quantile_k_kernels_per_sample(...)[1].unsqueeze(1).detach()` has shape `(B, 1, k)`, and line 112 unpacks it as `q, scores = ...`, which unpacks along the batch dimension `B`.
   - **Impact**: Raises `ValueError: too many values to unpack (expected 2)` for any `B != 2` (or `not enough values` for `B == 1`). For `B == 2`, `q` receives sample 0's scores and `scores` receives sample 1's scores.

3. **Mismatched kernel indices in `SpaceMath.get_quantile_k_kernels_per_sample`**
   - **File**: `spacemath.py:259-263`
   - **Root cause**: `selected_mask[b]` is indexed by **unsorted** channel index `0..C-1`, so `sample_indices = torch.where(selected_mask[b])[0]` are raw channel indices. Line 262 then indexes `sorted_indices[b, sample_indices]` (which is sorted by descending score), returning unrelated kernel indices that do not correspond to `scores[b, sample_indices]`.
   - **Impact**: Silently returns wrong kernel IDs. Additionally, the return type annotation (`tuple[Tensor, Tensor, Tensor]`) and docstring promise 3 tensors, while line 273 returns 2.

4. **`SpaceMath.activations_to_scores` computes threshold on `act(xd)` but normalizes raw `xd`**
   - **File**: `spacemath.py:122-127`
   - **Root cause**: `q` is computed from `torch.quantile(act(xd), ...)` when `activation_fn is not None`, but line 127 calls `cls.normalize_by_threshold(xd, q)` on unactivated `xd`.
   - **Impact**: Compares raw pre-activations against post-activation quantiles (e.g., raw logits vs. Sigmoid quantile in `(0, 1)`), producing wrong active/inactive splits.

5. **`infer_shape` has a stray `self` parameter and mismatched `x_in` typing**
   - **File**: `common.py:49-54`
   - **Root cause**: Defined as a module-level function `def infer_shape(self, x_in: torch.Tensor, x_out: torch.Tensor):`, likely copied out of a class. Also indexes `x_in[0]`, assuming a PyTorch forward-hook `tuple[Tensor, ...]`, despite the `torch.Tensor` annotation.
   - **Impact**: Calling `infer_shape(x_in, x_out)` raises `TypeError: infer_shape() missing 1 required positional argument: 'x_out'`.

6. **`is_linear` and `is_layer` misclassify `nn.Identity` (and `nn.Bilinear`) as linear layers**
   - **File**: `common.py:61-75`
   - **Root cause**: Exact string check `module.__class__.__module__ == 'torch.nn.modules.linear'`. In PyTorch, `nn.Identity` and `nn.Bilinear` are defined in `torch.nn.modules.linear`.
   - **Impact**: `is_linear(nn.Identity())` and `is_layer(nn.Identity())` both return `True`, causing `ModularObserver` and `ModelMapper` to treat `nn.Identity` as an observable linear layer. Also fails to recognize any user-defined subclass of `nn.Linear`/`nn.Conv2d`.

### Medium Severity

7. **`contrast_distance_*` vs. `contrast_softmax_ce` geometric inconsistency in cosine space**
   - **File**: `contrast.py:65-69` vs. `contrast.py:159-162`
   - **Root cause**: `_softmax_ce_contrast_core` projects `positions` onto the unit sphere (`P = space_normalize(positions, ...)`) **before** computing `centers_on`, whereas `_distance_contrast_core` computes `centers_on = w_on_norm @ positions` from **unnormalized** `positions`.
   - **Impact**: In cosine space, if `positions` drift to unequal norms during optimization, high-norm units disproportionately dominate `centers_on` in `contrast_distance_log`/`contrast_distance_linear`.

8. **`loss_layer_alignment` crashes on empty sequence and returns non-zero on single-layer 3D Euclidean tensor**
   - **File**: `losses.py:116-126`
   - **Root cause**: When `centers_per_layer == []`, `len < 2` is true and `centers_per_layer[0]` raises `IndexError`. When `centers_per_layer` is a 3D `Tensor` with `L == 1`, it skips the `< 2` guard and computes Euclidean `distance_elementwise(x, x)` which returns `sqrt(eps) = 1e-4` instead of `0.0`.

9. **`loss_batch_repulsion` silently ignores `mixed` for 2D inputs**
   - **File**: `losses.py:45-55`
   - **Root cause**: The `if mixed is not None:` filtering branch is nested inside `if centers_on.ndim == 3:`. Passing 2D `centers_on (bs, D)` with 1D `mixed (bs,)` silently ignores `mixed`.

10. **`mask_cluster_membership` fails to return `None` when a single cluster has ID >= 1 and creates empty rows for non-contiguous IDs**
    - **File**: `weighting.py:74-81`
    - **Root cause**: `K = int(lab.max().item()) + 1` assumes contiguous 0-based IDs. For `labels = [-1, -1, 2, 2]` (only 1 non-noise cluster), `K = 3 >= 2`, returning a `(3, U)` mask with 2 all-zero rows instead of `None`. Also, 2D labels like `(U, 1)` pass `lab.numel() == num_units` and produce a 3D `(K, 1, U)` mask.

11. **`spow` / `spow_` defaults to `force_zero=True`, producing `inf` for negative powers on near-zero inputs**
    - **File**: `common.py:17-38`
    - **Root cause**: `spow_(x, power, eps=1e-8, force_zero=True)` zeroes values with `|x| < eps` before `x.pow_(power)`, so `spow(tensor([1e-10]), -1.0)` returns `inf`. Even with `force_zero=False`, `zclamp_` excludes exact `0.0` via `(x != 0)`, so `0.0 ** -1.0` still returns `inf`.

12. **`SpaceMath.distance` Euclidean pairwise `.squeeze()` drops singleton dimensions**
    - **File**: `spacemath.py:85`
    - **Root cause**: `torch.cdist(x, y, p=2).squeeze()` collapses `(1, N)` to `(N,)` and `(1, 1)` to scalar `()`, inconsistent with cosine pairwise `torch.mm(x_norm, y_norm.T)` which always preserves `(M, N)`.

### Low Severity / Edge Cases

13. **`reduce_conv_locality` uses signed `max_pool2d` before `.abs()` and only supports 4D `Conv2d`** (`reducers.py:30-33`): On negative pre-ReLU feature maps, `max_pool2d` selects the *least negative* (weakest magnitude) value instead of the strongest response. Also fails on `Conv1d` (3D) and `Conv3d` (5D) unlike `conv_l2`/`conv_mean_abs`/`conv_max`.
14. **`stats_update_` with non-zero `threshold`** (`weighting.py:28-29`): Clamps `scores * idxs_on` at `0.0` (dropping negative active scores if `threshold < 0`) and `-(scores * idxs_off)` at `0.0` (dropping positive inactive scores if `threshold > 0`).
15. **`sigreg_weak` / `sigreg_strong` on empty `(0, D)` input** (`sigreg.py:24, 38`): Returns `NaN` when `N == 0` (guarded in `loss_sigreg`, but unguarded when called directly).
16. **Misplaced docstring in `clear_torch_compile_cache`** (`common.py:114-117`): Placed after `if verbose:`, so `clear_torch_compile_cache.__doc__` is `None`.

---

## 3. Inconsistencies & Design Issues

- **Missing `get_loss` helper**: `reducers.py`, `scoring.py`, `weighting.py`, and `contrast.py` all expose `<REGISTRY>` + `get_<kind>(name)` with `.strip().lower()` normalization. `losses.py` exposes `LOSSES` but no `get_loss(name)`, and `sigreg(method=...)` is case-sensitive (`"WEAK"` raises `ValueError`).
- **Incomplete `__init__.py` exports**: `DEFAULT_REDUCER_BY_LAYER` (from `reducers.py`) and `common.py` helpers (`is_linear`, `is_conv`, `is_layer`, `is_activation`, `zclamp`, `spow`, `compile`, `staticmethod_compile`, `classmethod_compile`, `clear_torch_compile_cache`) are not exported in `__init__.py`.
- **Import-time global side effects (`common.py:15, 41-44`)**: Importing `spacemath` mutates global PyTorch matmul precision (`torch.set_float32_matmul_precision('high')`) and monkey-patches `torch.Tensor.{zclamp, zclamp_, spow, spow_}`. Neither is documented in `__init__.py`, and monkey-patching `torch.Tensor` can collide with other libraries or confuse static type checkers.
- **Unconditional import-time `torch.compile` (`contrast.py:52, 147`, `common.py:78-91`)**: `@torch_compile` compiles immediately using module-global `default_compile_mode` with no opt-out flag or env var. In CPU-only, CI, or read-only-home sandboxes, invoking `contrast_*` fails with `InductorError: OSError: Read-only file system: '~/.triton'` unless `TRITON_CACHE_DIR` is overridden.
- **Euclidean self-distance mismatch (`space_type.py:70` vs `100`)**: `distance_elementwise(x, x, "euclidean")` returns `sqrt(eps) = 1e-4` (because of `.clamp(min=eps).sqrt()`), whereas `distance_pairwise(x, x, "euclidean")` uses `torch.cdist` and returns `0.0`.
- **Silent override in `activations_to_scores` (`scoring.py:139`)**: Passing `activation_fn=nn.Softmax()` silently overrides an explicit `score_method` (e.g., `"top_k"`), even when `apply_activation=False`.
- **Overly permissive `SpaceType.normalize_name` (`space_type.py:22-25`)**: Uses `.startswith("cos")` / `.startswith("euc")`, which silently accepts arbitrary prefixes like `"cost"` or `"eucalyptus"`.
- **Code hygiene**: Unused imports `nn` and `F` and Italian comments in `sigreg.py:2-3, 27, 38`; unreachable `else: raise Exception(...)` in `spacemath.py:96-97`; `print("WARNING: ...")` instead of `logger.warning(...)` in `spacemath.py:155, 159`.

---

## 4. Opportunities & Recommendations

1. **Fix `score_normalize_by_threshold` first (highest impact)**: Shift active scores by `threshold` (`scores_on = (values - threshold) * idxs_on`) and guard degenerate rows where `scores_max == 0` (e.g., constant rows), or explicitly clamp/normalize via `(values - threshold)` on both sides so `[-1, 1]` holds for any real-valued distribution.
2. **Unify cosine normalization in `contrast.py`**: Normalize `positions` via `space_normalize` at the start of `_distance_contrast_core` (matching `_softmax_ce_contrast_core`) so cosine geometry is strictly norm-invariant across all contrastive objectives.
3. **Make `torch.compile` opt-in / configurable**: Support an env var (e.g., `POSITRONIX_COMPILE=0`) or lazy compilation wrapper in `common.py` so unit tests, CPU debugging, and read-only containers run instantaneously without Triton/Inductor failures.
4. **Replace module-string predicates with `isinstance` checks**: Use `isinstance(m, (nn.Linear, nn.LazyLinear))`, `isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, ...))`, etc. in `common.py` so `nn.Identity` is excluded and custom subclasses work.
5. **Complete the v2 migration & deprecate `spacemath.py`**:
   - Add `get_loss(name)` to `losses.py` and case-insensitive `method` to `sigreg.py`.
   - Move `LayerType` and `ActivationType` (or replace with clean enums) out of legacy `spacemath.py` so `modularobserver` and `layerspace` don't depend on legacy code.
   - Add optional `generator: Optional[torch.Generator]` to `sigreg_weak` / `sigreg_strong` for reproducible experiments and deterministic tests.
   - Exclude the zero diagonal in `loss_batch_repulsion` (or divide by `N * (N - 1)`) so loss magnitude doesn't depend on batch size `N`.
   - Add `nn.Module` / `Transformer` / `RNN` reducers (e.g., sequence-mean / sequence-max `(B, T, U) -> (B, U)`) to `reducers.py` to support functional modularity beyond MLPs and Conv2d.

---

## 5. Docs / Tests / Examples Summary

- **Documentation (`positronixlab/spacemath/docs/`)**:
  - Preserved [PARKING LOT PROBLEM.md](docs/PARKING%20LOT%20PROBLEM.md).
  - Added [SPACEMATH_SYSTEM.md](docs/SPACEMATH_SYSTEM.md) (architecture, pipeline stages, math formulations, `torch.compile` boundaries, known caveats).
  - Added [SPACEMATH_USAGE.md](docs/SPACEMATH_USAGE.md) (end-to-end recipes, registry lookups, scoped contrast, SIGReg, legacy migration notes).
- **Tests (`positronixlab/spacemath/tests/`)**:
  - Existing [test_spacemath_primitives.py](tests/test_spacemath_primitives.py): 24 tests covering v2 registries, distances, scoring, conv reducers, weighting, masks, contrast forward/backward, and additional losses.
  - Added [test_spacemath_extended.py](tests/test_spacemath_extended.py): covers `common.py` (`zclamp`, `spow`, predicates, `compile` decorators), `sigreg.py` (`sigreg_weak`, `sigreg_strong`, sketching, invalid method), `space_type.py` (`SpaceType` normalization & edge cases), `spacemath.py` (`LayerType`, `ActivationType`, `SpaceMath`), `reducers.py`, `weighting.py`, `losses.py`, `contrast.py` edge cases, plus `@pytest.mark.xfail(strict=True)` tests locking in every confirmed bug above.
- **Examples (`positronixlab/spacemath/examples/`)**:
  - Added [example_spacemath.py](examples/example_spacemath.py): standalone CPU script demonstrating linear + conv activation scoring, running stats & inverse-frequency weighting, global vs. cluster-scoped contrast, auxiliary losses + SIGReg, and a short Adam optimization loop on CU embeddings.
