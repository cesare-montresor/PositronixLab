# `positronixlab/modularspace` — Open Issues Analysis

> **Status (2026-11-27 — `v0.1.3` / `5740a0c` + working tree):** **40 passed, 4 xfailed, 0 failed** (`tests/modularspace/`).
> **Note:** Stale `layerspace_old.py` and `custertracking.example.py` were removed; all 9 bugs below (**B1–B9**) remain open, plus v0.1.3 **B5** (`custertracker.py:147, 156, 161, 165, 223` — precision-only overlap, merge weighting by `old.sizes`, unbounded `self.history`, `assert old is not None`) and minor hygiene (`zip` without `strict=`, one-line `if`).

---

## 1. Overview & Refactor Status

`positronixlab/modularspace` implements the decoupled vector-space "brain atlas" for computation units (CUs / neurons) and their cluster-level identities:

| Component | File | Status | Role |
|---|---|---|---|
| `LayerSpaceConfig` | `config.py` | Active (new) | Standalone `BaseConfig` dataclass holding geometry, stage selectors, loss weights, and method knobs (Ax-friendly, independent of `PipelineConfig`/`nas`). |
| `PRESETS` / `get_preset*` | `presets.py` | Active (new) | Named hyperparameter presets (`v1_centers`, `distance_sigreg_centers`, `distance_sigreg_positions`, `softmax_ce`) reproducing legacy/WIP variants. |
| `SpaceStep` / `SpaceStepSpec` | `space_step.py` | Active (new, partially orphaned) | Stateless composable runner + flat spec wiring `spacemath` scoring, weighting, contrast, and auxiliary losses (`centers`, `batch`, `sigreg`). |
| `LayerSpace` | `layerspace.py` | Active (refactored) | Per-layer CU embedding manager (`positions`, `optimizers`, optional `stats`). Directly resolves config/kwargs and executes the full loss pipeline inline instead of delegating to `SpaceStep`. |
| `LayerSpace` (old) | `layerspace_old.py` | Superseded (still in tree) | Previous version of `LayerSpace` that delegated to `SpaceStep` and accepted `step_spec: SpaceStepSpec \| Mapping \| str`. Not imported anywhere in `__init__.py` or other modules. |
| `ModularSpace` | `modularspace.py` | Legacy-active (lazy export) | V1 per-layer space using `SpaceMath` and `@staticmethod_compile`. Only supports `distance_log` + `centers` repulsion and requires `PipelineConfigV1`-compatible fields. |
| `ModularOverSpace` | `modularoverspace.py` | Active (lazy export) | Per-layer wrapper around `ClusterTracker` maintaining stable outer-space coordinates for cluster identities. |
| `ClusterTracker` (`ClusterIdentityTracker`) | `custertracker.py` | Active (typo in filename) | Vectorized overlap-matrix cluster identity tracker supporting `init`, `match`, `split`, `merge`, `switch`, `add`, `remove`. |
| Prototype script | `custertracking.example.py` | Stale (module root) | Original standalone prototype of `custertracker.py` left at package root instead of `examples/` or `legacy/`. |

**Refactor status summary:** The migration from `ModularSpace` (v1) + ad-hoc `layerspace_*.py` scripts to `LayerSpace` + `LayerSpaceConfig` + `presets` + `spacemath` primitives is mostly complete and tested (13/13 existing tests pass on CPU when `TRITON_CACHE_DIR` is writable). However, the refactor stopped mid-cleanup: `LayerSpace` was rewritten to inline `SpaceStep.compute_loss` without removing `layerspace_old.py` or reconciling `SpaceStep`/`SpaceStepSpec`, leaving duplicated loss logic and API drift.

---

## 2. Bugs & Defects

### B1. `ModularSpace.get_space()` unconditionally fails when `stats_enabled=False` (default)
- **Location:** `positronixlab/modularspace/modularspace.py:59`
- **Root cause:** `_init_space()` only populates `self.stats[layer_id]` when `self.stats_enabled` is `True` (lines 43–46), yet `get_space()` checks `if None in (layer_info, positions, optimizer, stats): raise ValueError(...)`.
- **Impact:** Instantiating `ModularSpace` with its default `stats_enabled=False` and calling `get_space()`, `process_activations()`, or `process_scores()` immediately raises `ValueError: Layer <id> not found in ModularSpace.`. (Fixed in `layerspace.py:166`, still broken in `modularspace.py:59`.)

### B2. `ModularSpace._process_scores()` computes all-pairs `(bs, bs)` center distance instead of per-sample `(bs,)`
- **Location:** `positronixlab/modularspace/modularspace.py:147`
- **Root cause:** Calls `SpaceMath.distance(center_on, center_off, dim=1, distance_type=config.modular_space_distance)` without `pairwise=False`. `SpaceMath.distance` defaults to `pairwise=True` (`spacemath.py:77`), so it returns a `(bs, bs)` cross-sample distance matrix despite the inline comment `# on the same datapoint`.
- **Impact:** `loss_centers` repels every sample's `center_on` from every other sample's `center_off` (mixing batch-separation with on/off center separation), diverging from `SpaceStep`/`LayerSpace`'s `loss_centers_repulsion` (which is strictly per-sample).

### B3. `LayerSpace.process_scores()` crashes on `loss.backward()` when no loss term produces gradients
- **Location:** `positronixlab/modularspace/layerspace.py:210, 219, 338` (and `space_step.py:183, 191`)
- **Root cause:** `total_loss` is initialized to `zero = positions.new_zeros(())`, which has `requires_grad=False`. If all loss weights are `0.0`, OR if only `loss_cluster > 0.0` (with `clusters_labels=None` or `< 2` clusters) or `loss_sigreg > 0.0` with `sigreg_target="centers"` and no active contrast (`primary is None`), `total_loss` remains a detached scalar zero.
- **Impact:** `loss.backward()` at `layerspace.py:338` raises `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`.

### B4. `ClusterTracker._match_and_project()` crashes when previous timestep has `K_old == 0`
- **Location:** `positronixlab/modularspace/custertracker.py:175`
- **Root cause:** `row = frac[j]` has shape `(0,)` when `k_old == 0`, and `torch.argmax(row)` is called unconditionally without guarding `k_old == 0`.
- **Impact:** Raises `IndexError: argmax(): Expected reduction dim 0 to have non-zero size.` if an empty clustering (`centroids` with shape `(0, D)`) is followed by a non-empty clustering.

### B5. `ClusterTracker` `switch` event calls `_split_positions()` with 1 centroid, making split offset a no-op
- **Location:** `positronixlab/modularspace/custertracker.py:239, 285-287`
- **Root cause:** Line 239 passes `centroids[[child_id]]` (shape `(1, D)` -> `m == 1`) to `_split_positions()`. Lines 286–287 short-circuit `if m == 1: return parent_outer.unsqueeze(0)`, ignoring `centroids[[child_id]]` and `seed=old_id * 1_000_003 + child_id`.
- **Impact:** The "split-off" step of `switch` produces zero displacement (`split_pos == old.outer[old_id]`), contradicting `docs/cluster_identity_tracking.md` §4.5 and rendering the deterministic seed calculation dead code. Additionally, lines 243–245 apply `drift = 0.15 * (weight / size_new)` before multiplying by `weight` again in the weighted average, quadratically damping the switch pull by `0.15 * (weight / size_new)^2` (diverging from both the doc and `custertracking.example.py:288`).

### B6. `LayerSpace._init_layers()` and `ModularSpace._init_layers()` mutate caller's `observed_layers` dict in-place
- **Location:** `positronixlab/modularspace/layerspace.py:134`, `modularspace.py:31`
- **Root cause:** Writes `self.observed_layers[layer_id]["activation"] = act_fn` directly into the caller-supplied nested dict.
- **Impact:** Side-effect mutation of caller state (e.g. sharing an `observed_layers` metadata dict across multiple `LayerSpace` instances or serializing it to JSON later fails because it now contains an `nn.Module`).

### B7. `ModularSpace.process_activations()` raises `KeyError` if `layer_type` is omitted
- **Location:** `positronixlab/modularspace/modularspace.py:65`
- **Root cause:** Indexes `layer_info["layer_type"]` directly instead of `layer_info.get("layer_type", "linear")` (as `LayerSpace` does at `layerspace.py:175`).
- **Impact:** Fails on minimal `observed_layers` entries that omit `layer_type`.

### B8. `ClusterTracker._one_hot()` does not validate upper bound of cluster IDs
- **Location:** `positronixlab/modularspace/custertracker.py:68-70`
- **Root cause:** Checks `valid = assignment >= 0` but never validates `assignment < k`.
- **Impact:** Out-of-range labels (non-contiguous IDs or mismatched `centroids.shape[0]`) trigger a raw `IndexError` inside tensor indexing rather than a clear `ValueError`.

### B9. Unnecessary top-level `positronixlab.pipeline` imports couple `ModularSpace` & `ModularOverSpace` to `parallelazy`
- **Location:** `positronixlab/modularspace/modularspace.py:8`, `positronixlab/modularspace/modularoverspace.py:10`
- **Root cause:** Both modules import `PipelineConfig` (and `PipelineConfigV1` unused in `modularspace.py`) at runtime solely for type annotations. Importing `PipelineConfig` triggers `positronixlab.pipeline.pipeline_config` -> `from parallelazy.plugin_data import ManagerOptions`.
- **Impact:** Importing `ModularSpace` or `ModularOverSpace` fails with `ModuleNotFoundError: No module named 'parallelazy'` in any environment where `parallelazy` is absent or inaccessible, even though neither class uses `parallelazy` at runtime. Guarding the import with `if TYPE_CHECKING:` (or annotating with `LayerSpaceConfig` / `BaseConfig`) removes this coupling completely.

---

## 3. Inconsistencies & Design Issues

1. **`LayerSpace` vs `SpaceStep` code duplication & API drift:**
   - `layerspace.py` (lines 198–312) duplicates `SpaceStep.compute_loss()` (`space_step.py:158-290`) almost line-for-line, while `layerspace_old.py` (which delegated cleanly to `SpaceStep`) remains in the package directory.
   - `LayerSpace.__init__` does not accept `SpaceStepSpec` or a preset string (unlike `layerspace_old.py`), forcing callers to convert via `LayerSpaceConfig.from_dict(get_preset(...))`.
   - `space_step.py:98` maps both `"modular_space_distance"` and `"modular_space_type"` to `space_type`, whereas `layerspace.py:86` only reads `"modular_space_distance"`.
   - `layerspace.py:119-120` validates `sigreg_target` eagerly in `__init__`, whereas `space_step.py:283` only validates it lazily inside `compute_loss()` when `loss_sigreg > 0.0`.
2. **Asymmetric handling of `loss_centers` vs `loss_batch` / `loss_sigreg`:**
   - In both `layerspace.py:263` and `space_step.py:236`, `loss_centers` requires `global_res is not None` and is silently skipped when `loss_global == 0.0` even if `cluster_res` is available; `loss_batch` and `loss_sigreg` use `primary = cluster_res if cluster_res is not None else global_res`.
3. **Config mismatches & unused fields:**
   - `LayerSpaceConfig` (`config.py`) omits `modular_space_spatial_radius` (even though `LayerSpace.__init__` and `SpaceStepSpec` expose `spatial_radius: int = 3`).
   - `LayerSpaceConfig` defines `space_epochs`, `modular_modularize_decay`, and `modular_overspace_size`, none of which are consumed by `LayerSpace` or `SpaceStep`.
   - `ModularSpace.__init__` (`modularspace.py:13`) and `ModularOverSpace.__init__` (`modularoverspace.py:27`) annotate `config: PipelineConfig`, but the base `PipelineConfig` (`pipeline_config.py`) only has `name`, `seed`, `device` — it lacks `modular_space_*` and `modular_overspace_size`. `modularspace.py:8` also imports `PipelineConfigV1` unused.
4. **Filename & placement hygiene:**
   - `custertracker.py` has a typo (`custer` instead of `cluster`), mirrored in `custertracking.example.py` and `examples/example_custertracker.py`.
   - `custertracking.example.py` sits at the package root instead of `examples/` or `legacy/`.
   - `layerspace_old.py` sits alongside `layerspace.py` at the package root instead of `legacy/`.
   - `examples/example_custertracker.py` is a 729-line standalone Dash app that does **not** import `ClusterTracker` or anything from `positronixlab.modularspace`.
5. **`ModularOverSpace` quirks (`modularoverspace.py`):**
   - Accepts and stores `observed_layers` (line 34) but never validates or iterates over it.
   - Assigns per-layer seeds as `self.seed + len(self.trackers)` (line 49), making seeds depend on the arrival order of `update()` calls.
   - `ClusterTracker.history` (`custertracker.py:127`) appends every step's event list indefinitely with no cap or opt-out, causing unbounded memory growth in long runs.
6. **Inaccurate type annotations in `modularspace.py`:**
   - `process_activations` (line 63) annotates return type `tuple[tuple, Tensor, Tensor | None]` (actually `tuple[dict[str, np.ndarray], Tensor, np.ndarray | None]`).
   - `_process_scores` (line 102) annotates a nested 2-tuple return type but returns a flat 4-tuple.
7. **`torch.compile` / Triton cache sensitivity:**
   - `spacemath/contrast.py` decorates `_distance_contrast_core` and `_softmax_ce_contrast_core` with `@torch_compile`. In restricted/sandboxed environments where `~/.triton` is read-only, running `LayerSpace` or `SpaceStep` fails with `InductorError: OSError: [Errno 30] Read-only file system: '/home/cesare/.triton'` unless `TRITON_CACHE_DIR` is pointed to a writable directory (e.g. `/tmp/triton_cache`).

---

## 4. Opportunities & Recommendations

1. **Deduplicate `LayerSpace` and `SpaceStep`:** Have `LayerSpace` delegate to `SpaceStep` (or construct a `SpaceStepSpec` from `config` + kwargs overrides) and archive `layerspace_old.py` to `legacy/`. This eliminates ~115 lines of duplicated loss logic and keeps `LayerSpace` and `OverSpace` guaranteed in sync.
2. **Guard zero-gradient loss before `backward()`:** In `LayerSpace.process_scores()`, check `if loss.requires_grad:` before `loss.backward(); optimizer.step()` (or raise a clear `ValueError` if all loss terms are disabled).
3. **Fix `ModularSpace.get_space()` and `pairwise=False`:** Remove `stats` from the `None in (...)` check in `modularspace.py:59`, pass `pairwise=False` at `modularspace.py:147`, and use `layer_info.get("layer_type", "linear")` at `modularspace.py:65`.
4. **Harden `ClusterTracker` and `ModularOverSpace`:**
   - Guard `k_old == 0` and validate `0 <= assignment < k_new` in `custertracker.py`.
   - Resolve the `switch` `_split_positions` `m == 1` no-op (either compute a directional offset from `old.centroids[old_id]` vs `centroids[child_id]` or simplify `switch` explicitly to match the intended formula).
   - Add an optional `max_history` cap to `ClusterTracker` and deterministic per-layer seeding (e.g. hash of `layer_id`) in `ModularOverSpace`.
   - Copy `observed_layers` entries instead of mutating caller dicts in-place.
5. **Clean up package layout:** Move `custertracking.example.py` and `layerspace_old.py` to `legacy/` (or delete once verified unused); rename `custertracker.py` -> `clustertracker.py` with a deprecation shim re-exporting from `custertracker.py` so existing callers don't break.
6. **Align config types:** Annotate `ModularSpace` and `ModularOverSpace` with `LayerSpaceConfig | PipelineConfigV1 | PipelineConfigV2` (or a `Protocol`), and add `modular_space_spatial_radius` to `LayerSpaceConfig` + `_CONFIG_KEY_MAP`.

---

## 5. Docs / Tests / Examples Summary

- **Docs (`positronixlab/modularspace/docs/`):**
  - Preserved existing design notes (`layerloss.md`, `cluster_identity_tracking.md`) and `.mp4` demos.
  - Added `MODULARSPACE_SYSTEM.md` (architecture, decoupled space graph, `LayerSpace`/`SpaceStep`/`ModularSpace`/`ModularOverSpace`/`ClusterTracker` design) and `MODULARSPACE_USAGE.md` (practical recipes, presets, ablations, workarounds).
- **Tests (`positronixlab/modularspace/tests/`):**
  - Existing `test_layerspace.py` (13 tests) passes on CPU (with writable `TRITON_CACHE_DIR`).
  - Added `test_config_presets.py`, `test_space_step.py`, `test_modularspace.py`, and `test_clustertracker.py` covering `LayerSpaceConfig`, `PRESETS`, `SpaceStep`/`SpaceStepSpec`, `ModularSpace`, `ClusterTracker`, and `ModularOverSpace`, with `@pytest.mark.xfail(strict=True)` tests pinning confirmed bugs (B1, B3, B4, B5).
- **Examples (`positronixlab/modularspace/examples/`):**
  - Reviewed `example_clustertracker_minimal.py` (runs cleanly on CPU) and `example_custertracker.py` (standalone Dash demo, does not use `ClusterTracker`).
  - Added `example_layerspace.py` demonstrating `LayerSpace` + `LayerSpaceConfig` + `presets` on synthetic modular activations on CPU.
