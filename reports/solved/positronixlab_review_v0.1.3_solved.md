# PositronixLab v0.1.3 Code Review — Solved Issues Archive

> **Archived:** 2026-11-27 (resolved in `5740a0c` + working tree).
> **Remaining open items:** See [positronixlab_review_v0.1.3.md](file:///home/cesare/Projects/PositronixLab/reports/positronixlab_review_v0.1.3.md).

---

## Resolved Items Summary

| ID | Sev. | Resolved In | Location | Resolution Summary |
| :--- | :---: | :--- | :--- | :--- |
| **A1 / E8** | Med | `5740a0c` | `positronixlab/__init__.py:6` | `__version__` now reads `_package_version("positronixlab")`. |
| **A2** | High | `5740a0c` | `tasks/masks.py:11, 396-399` | `MaskApplier(seed=…)` uses deterministic `zlib.crc32(layer_name.encode("utf-8")) & 0x7FFFFFFF`. |
| **A3** | High | `5740a0c` | `tasks/masks.py:362-389, 479` | `__enter__` wraps `_apply()` in `try/except BaseException: self._restore()`, deduplicates tied weights by `data_ptr()`, restores in `reversed()` order. |
| **A4** | High | `5740a0c` | `tasks/atlas_discoverer.py:28-48` | `_reduce_activation_to_2d` dispatches on `nn.modules.conv._ConvNd` for Conv1d/2d/3d; both hook sites pass `module=mod_ref`. |
| **A5** | Med | `5740a0c` | `modularobserver/layerobserver.py:36-48` | `on_grad_weights` fixes late-binding closure (`p=param`), handles `callback=None`, tracks & removes `_weight_grad_handles`. |
| **A6 (`ModularClusterKmeans`)** | Med | `5740a0c` | `modularcluster/modularcluster_kmeans.py:96-98, 115, 129, 225-231` | 1-cluster fallback for `k_min > k_max` (A6a), `< 2` unique-label silhouette guard (A6b), `hasattr(faiss, "StandardGpuResources")` guard (A6c), unused `dists` discarded (A6d). *(A6a/A6b still open in `ModularClusterKmeansStream`.)* |
| **A7** | Med | `5740a0c` | `utils/utils.py:26-27, 40-69` | `suppress_output` now uses `_SUPPRESS_LOCK`, ref-counted nesting depth, and `stream.flush()` on entry/exit. |
| **A8** | Med | `5740a0c` | `utils/utils.py:24, 71-107` | `session_path()` caches per `(base, safe_name, uid_key)`, sanitizes `name`, supports `basedir`/`POSITRONIXLAB_OUTPUT_DIR`. |
| **A9** | Low-Med | `5740a0c` | `utils/utils.py:232-336`; `pipeline/config.py:24` | `load_data` raises `FileNotFoundError`/`ValueError` (with `default=`), validates YAML kwargs, documents `.pkl` warning; `signature_settings` adds `_json_default` & `usedforsecurity=False`; `pipeline/config.py:24` adds `usedforsecurity=False`. |
| **A10** | Med | `5740a0c` | `spacemath/common.py:82-186`, `contrast.py:8` | Fixed docstring (A10a), `_is_safe_cache_dir()` guard (A10b), `getpass.getuser()` (A10c), renamed `compile` → `torch_compile` (A10d), added `POSITRONIXLAB_COMPILE` env switch (A10e). |
| **B1** | High | *working tree* | `tasks/masks.py:22, 325-360, 432-488` | Added `bias_mode: Literal["keep", "mask"] = "keep"` with backup/restore & tied-bias dedup. |
| **B2** | High | *working tree* | `tasks/atlas_discoverer.py:106-107, 291-294, 393-481` | Added differential `attribution_mode` (`"selectivity"` default, `"contrastive"`, `"raw"`), `energy_mode` (`"abs"`, `"relu"`), and `pairwise_iou` metadata logging. |
| **B3** | High | `5740a0c` | `tasks/atlas_discoverer.py:56-61, 320-324, 369-372` | Rejects one-shot iterators and raises `ValueError` on empty energy or empty mask. |
| **C1** | Low-Med | `5740a0c` | `tasks/masks.py:405, 407, 420, 430` | Replaced `weight.data.mul_(mask)` with `masked_fill_(..., 0)` (safe against `inf*0`/`NaN*0`). |
| **C2 (`atlas_discoverer.py`)** | Low-Med | `5740a0c` | `tasks/atlas_discoverer.py:51-53` | `_model_device()` uses `next(model.parameters(), None)` with CPU fallback. *(Still open in `tasks/benchmarks.py:29`.)* |
| **D1** | Perf | `5740a0c` | `tasks/atlas_discoverer.py:290-299, 317-318` | Energy accumulates on-device; single `.cpu()` transfer after loop. |
| **F (partial)** | Hygiene | `5740a0c` + 2026-11-27 | `spacemath/`, `dataprovider/`, `deeplearning/`, `modularobserver/`, `modularui/` | Fixed `F601` (6 duplicate keys in `dataprovider/__init__.py`), `S110` (`C6`), `C7` (`assert`s in `nas`/`modularspace`/`cvr_simclr`), `C8` (`B028` `stacklevel=2`), `C4`, `C2` (`benchmarks.py`), `B009` (`class_removal`), `E712` (`== False`), `layouts.py` lambda, `deeplearning` `B007`/`F541`, `modularobserver` `F841`. |
