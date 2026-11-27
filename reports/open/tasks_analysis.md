# `positronixlab/tasks` — Open Issues Analysis

> **Status (2026-11-27 — `v0.1.3` / `5740a0c` + working tree):** **35 passed, 9 xfailed, 0 failed** (`tests/tasks/`; all 9 xfails are in legacy `tasks/compositionality/`).
> **Solved issues archive:** Unified `SubnetworkMask`/`MaskApplier`/`AtlasDiscoverer`/`ModularityBenchmark` and resolved v0.1.3 A2–A4, B1–B3, C1, C2 (`atlas_discoverer.py`), D1 moved to [solved/tasks_solved.md](file:///home/cesare/Projects/PositronixLab/reports/solved/tasks_solved.md).

---

## 1. Open Issues in New Unified `tasks/` API

### Medium Severity

1. **`SubnetworkMask.to_unit_mask()` wrong axis for HuggingFace `Conv1D`**
   - **Location:** `positronixlab/tasks/masks.py:107-127`
   - **Details:** Always reduces over `w_mask.shape[0]`, which is incorrect for HuggingFace `Conv1D` weights of shape `(in_features, out_features)` (whereas `_broadcast_unit_mask_to_weight` at `masks.py:47-48` explicitly special-cases `Conv1D`).

2. **`MaskApplier(mode="complement_sampled", bias_mode="mask")` edge case**
   - **Location:** `positronixlab/tasks/masks.py:465-474`
   - **Details:** When a weight mask is projected to unit level via `any(dim=1)` and >50% of units are active, `eligible_mask=~u_m` has fewer than `n_active_b` entries and `_sample_random_binary_mask` raises `ValueError` even when weight-level sparsity is >50%.

3. **`ModularityBenchmark.evaluate_mask()` does not expose `bias_mode`**
   - **Location:** `positronixlab/tasks/benchmarks.py:213`
   - **Details:** Does not accept/forward `bias_mode` to `MaskApplier`.

### Low Severity & Performance (v0.1.3 C3, C5, D2)

4. **Single-tensor input assumption (C3):** `atlas_discoverer.py:217-221, 310-311` calls `model(x.to(device))` with no `forward_fn`/`input_adapter` for dict/multi-tensor batches.
5. **Silent clustering cap & fallback (C5):** `atlas_discoverer.py:239-249` silently caps `k_max <= 8` and falls back to 1 cluster for `n_units < 4` without logging.
6. **Double dataloader pass (D2):** `atlas_discoverer.py:542-544, 557-565` runs two separate passes (`fit_atlas` + `_collect_mean_unit_energy`) even when the reference and subtask loaders are identical.

---

## 2. Open Issues in Legacy `tasks/compositionality/` Prototype (Superseded)

The earlier prototype under `positronixlab/tasks/compositionality/` (`atlas_check.py`, `weightmask_check.py`, `compositionality.py`) has been superseded for new benchmarks by the unified `tasks/` API above, but remains exposed and tested (9 strict xfails):

1. **Atlas mask key & 1D shape mismatch with `StaticMaskApplier` (#1):** `atlas_check.py:227-240`, `compositionality.py:27-37` (xfail `test_tasks_compositionality.py:435`).
2. **`WeightmaskCheck.run()` ignores `num_epochs` (#2):** `weightmask_check.py:16, 31` (xfail `:293`).
3. **Empty class slice in `WeightmaskCheck.run()` → `NaN` loss (#3):** `weightmask_check.py:39-51` (xfail `:318`).
4. **`AtlasCheck` leaks forward hooks permanently (#4):** `atlas_check.py:115` (xfail `:226`).
5. **`AtlasCheck._on_prediction()` infinite busy-wait loop (#5):** `atlas_check.py:144-145`.
6. **`MultiClassWeightmaskManager.remove_hooks()` doesn't restore `requires_grad` & not exception-safe (#6):** `weightmask_check.py:26-66, 93-94, 142-156` (xfail `:341`).
7. **`MultiClassWeightmaskManager` `KeyError` on weightless layers (#7):** `weightmask_check.py:81-86, 96-115` (xfail `:352`).
8. **`AtlasCheck.modularity_score()` `ZeroDivisionError` on empty stats (#8):** `atlas_check.py:308-313` (xfail `:211`).
9. **`CompositionalityCheck.run()` `ZeroDivisionError` on empty dataloader (#9):** `compositionality.py:151` (xfail `:424`).
10. **`CompositionalityPipeline` crashes when `dataprovider.test` is `None` (#10):** `compositionality.py:229` (xfail `:453`).
11. **`AtlasCheck._cluster_class_attribution()` device mismatch (#11):** `atlas_check.py:181-187`.
12. **Broken WIP `AtlasCheck._cluster_output_attribution()` (#12 / v0.1.3 B7):** `atlas_check.py:199-225` (shadows args, unvalidated broadcast).
13. **`WeightMasksMetrics.compute_sparsity()` returns density (#13):** `weightmask_check.py:200-203`.
14. **`DiscreteDistribution.update()` mutates `total_count` to Tensor / CPU-GPU mismatch (#14):** `atlas_check.py:37, 49-51`.
15. **Dead code (#15 remaining):** Unused `compute_spatial_loss()` (`weightmask_check.py:131-136`) and dead branch at `:166`.
