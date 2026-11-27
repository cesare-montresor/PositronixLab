# `positronixlab/tasks` — Solved Issues Archive

> **Archived:** 2026-11-27 (`v0.1.3` / `5740a0c` + working tree).
> **Active open issues:** See [tasks_analysis.md](file:///home/cesare/Projects/PositronixLab/reports/tasks_analysis.md).

---

## Resolved Architecture & Unified API Items

- **Unified `tasks/` contract implemented & exported in `tasks/__init__.py:1-35`:**
  - `masks.py`: Canonical `SubnetworkMask` (dotted keys, `unit`↔`weight` projection, set algebra, IoU/IoMin/overlap/true sparsity `1 - l0/max_l0`, save/load) and non-destructive `MaskApplier` (5 ablation modes + `bias_mode`). Supersedes legacy `compositionality/` bugs #1 (key/shape mismatch) and #13 (`compute_sparsity` density bug).
  - `discoverer.py`: `SubnetworkDiscoverer` ABC, `RandomMaskDiscoverer`, `unpack_batch`.
  - `atlas_discoverer.py`: `AtlasDiscoverer` (decoupled `ModularObserver` + `LayerSpace` + `ModularClusterKmeans`, clean `try/finally` hook teardown).
  - `benchmarks.py`: `ModularityBenchmark`, `BenchmarkReport`, `SubnetworkEvalResult`, `default_eval_accuracy`.

## Resolved v0.1.3 Review Findings (`5740a0c` + working tree)

| ID | Summary | Resolution Evidence |
| :--- | :--- | :--- |
| **A2** | `MaskApplier(seed=…)` not reproducible across processes (salted `hash()`) | Uses deterministic `zlib.crc32(layer_name.encode("utf-8")) & 0x7FFFFFFF` (`masks.py:11, 396-399`). Tested at `test_masks_and_discoverers.py:189-215`. |
| **A3** | `MaskApplier.__enter__` not exception-safe; tied-weights restore bug | Wraps `_apply()` in `try/except BaseException: self._restore()`, deduplicates by `weight.data_ptr()`, restores in `reversed()` order (`masks.py:362-389, 479`). Tested at `test_masks_and_discoverers.py:218-265`. |
| **A4** | `_reduce_activation_to_2d` wrong for `Conv1d` / `Conv3d` | Dispatches on `isinstance(module, nn.modules.conv._ConvNd)` → `act.mean(dim=tuple(range(2, act.ndim)))` (`atlas_discoverer.py:28-48`). Tested at `test_atlas_discoverer.py:78-113`. |
| **B1** *(working tree)* | Biases never masked in `MaskApplier` | Added `bias_mode: Literal["keep", "mask"] = "keep"` with backup/restore & tied-bias dedup (`masks.py:22, 325-360, 432-488`). Tested at `test_masks_and_discoverers.py:284-317`. |
| **B2** *(working tree)* | Subtask attribution was non-differential (raw mean `\|act\|`) | Added `attribution_mode` (`"selectivity"` default, `"contrastive"`, `"raw"`), `energy_mode` (`"abs"`, `"relu"`), and `pairwise_iou` metadata logging (`atlas_discoverer.py:106-107, 291-294, 393-481`). Tested at `test_atlas_discoverer.py:139-165`. |
| **B3** | Empty/one-shot loader → empty mask → silent no-op ablation | Added `_ensure_reiterable`, raises `ValueError` on empty energy or empty mask (`atlas_discoverer.py:56-61, 320-324, 369-372`). Tested at `test_atlas_discoverer.py:116-136`. |
| **C1** | `weight.data.mul_(mask)` turned `inf*0`/`NaN*0` into `NaN` | Switched to `masked_fill_(..., 0)` (`masks.py:405, 407, 420, 430`). Tested at `test_masks_and_discoverers.py:268-281`. |
| **C2 (`atlas_discoverer.py` & `benchmarks.py`)** | `next(model.parameters())` `StopIteration` on parameterless models | Added `next(model.parameters(), None)` CPU fallback in `atlas_discoverer.py:51-53` and `benchmarks.py:29` (2026-11-27). |
| **C4 (2026-11-27)** | Stale comment citing fixed `ModularClusterKmeans` float32 label bug | Removed stale comment in `atlas_discoverer.py:262`. |
| **D1** | Per-batch `.abs().cpu()` forced device→host sync | Accumulates energy on-device and calls `.cpu()` once after the loop (`atlas_discoverer.py:290-299, 317-318`). |
| **Legacy #15 (partial)** | Unused imports (`distribution`, `log`) in `compositionality/` | Removed from `atlas_check.py` and `weightmask_check.py`. |
| **Legacy #1 (2026-11-27)** | `StaticMaskApplier` rejected dotted Atlas keys and 1D `(out_features,)` masks | Supported dotted + underscored keys and reshaped 1D `(out_features,)` masks to broadcast against weights (`compositionality.py:28-36`); xfail removed and passing. |
| **Legacy #2 (2026-11-27)** | `WeightmaskCheck.run()` ignored `num_epochs` | Wrapped batch loop in `for _epoch in range(num_epochs):` (`weightmask_check.py:32`); xfail removed and passing. |
| **Legacy #3 (2026-11-27)** | Empty class slice in `WeightMaskCriterion` / `WeightmaskCheck` produced `NaN` loss | Guarded `targets.numel() == 0` in `WeightMaskCriterion.forward` and skipped empty slices in `WeightmaskCheck.run()` (`weightmask_check.py:42-44, 190-193`); xfail removed and passing. |
| **Legacy #4 (2026-11-27)** | `AtlasCheck` leaked forward hooks permanently after `run()` | Attached forward hooks inside `run()` with `try/finally: self.observer.on_forward_layer(None)` (`atlas_check.py:130-134`); xfail removed and passing. |
| **Legacy #6 (2026-11-27)** | `MultiClassWeightmaskManager.remove_hooks()` did not restore `requires_grad` & wasn't exception-safe | Saved/restored `_orig_requires_grad` in `remove_hooks()` and wrapped `WeightmaskCheck.run()` in `try/finally` (`weightmask_check.py:31-75, 100-103, 169-172`); xfail removed and passing. |
| **Legacy #7 (2026-11-27)** | `MultiClassWeightmaskManager` raised `KeyError` on weightless layers | Skipped layers absent from `self.masks` in `_setup_hooks_and_freezing()` (`weightmask_check.py:106-107`); xfail removed and passing. |
| **Legacy #8 (2026-11-27)** | `AtlasCheck.modularity_score()` raised `ZeroDivisionError` on empty stats | Guarded `n_layers > 0` with zero-dict fallback (`atlas_check.py:320-335`); xfail removed and passing. |
| **Legacy #9 (2026-11-27)** | `CompositionalityCheck.run()` raised `ZeroDivisionError` on empty dataloader | Guarded `total_samples > 0` (`compositionality.py:156-157`); xfail removed and passing. |
| **Legacy #10 (2026-11-27)** | `CompositionalityPipeline` crashed when `dataprovider.test` was `None` | Fell back `self.dataprovider.test or self.dataprovider.val or self.dataprovider.train` (`compositionality.py:233`); xfail removed and passing. |
