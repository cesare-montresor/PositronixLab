# `positronixlab/baselines` — Module Analysis

> **Verification Date:** 2026-11-27 (`v0.1.3` / `5740a0c` + working tree)
> **Pytest (`tests/baselines/`):** **14 passed, 0 xfailed, 0 failed** (+ head-to-head CPU benchmark `examples/tasks/example_atlas_vs_sota.py` exits 0).
> **Solved issues archive:** Minor cleanups moved to [solved/baselines_solved.md](file:///home/cesare/Projects/PositronixLab/reports/solved/baselines_solved.md).

---

## 1. Overview & Architecture

`positronixlab/baselines` packages external SOTA modularity works organized by paper (**Csordás et al.**, **Lepori et al.**, and **NeuroSurgeon**) behind unified, `parallelazy`-free registries (`ModelLoader`, `DataProviderLoader`) and `SubnetworkDiscoverer` adapters so **PositronixLab's `AtlasDiscoverer`** can be benchmarked apples-to-apples on identical models, datasets, and canonical `SubnetworkMask` contracts.

| File / Subpackage | Role | Status |
| :--- | :--- | :--- |
| `__init__.py` | Public exports (`SOTAModel`, `SOTAModelSpec`, `SOTADatasetSpec`, `CompositionalDataProvider`, `SubtaskSplit`, `MASKABLE_LAYER_TYPES`, `get_maskable_layer_names`, `ModelLoader`, `DataProviderLoader`, `CsordasGumbelDiscoverer`, `NeuroSurgeonDiscoverer`, `OddOneOutHead`, `is_neurosurgeon_available`, `wrap_with_neurosurgeon`, and subpackages `csordas`, `lepori`, `neurosurgeon`). | Active |
| `base.py` (195 lines) | `MASKABLE_LAYER_TYPES = (nn.Linear, nn.Conv2d, nn.Conv1d)` (+ dynamic HF `Conv1D`), `get_maskable_layer_names()`, `SOTAModelSpec`, `SOTADatasetSpec`, `SOTAModel` (compatible with `DeeplearningTrain`). | Active |
| `loader.py` (476 lines) | `ModelLoader` (12 registered models, `positronix_sota_bundle_v1` save/load, auto-detection & mask stripping for `neurosurgeon`/`lepori`/`csordas`/`plain` checkpoints) and `DataProviderLoader` (14 registered datasets). | Active |
| `csordas/` | `models.py` (5 architectures: `mlp`, `conv`, `lstm`, `resnet110`, `transformer` + `strip_csordas_mask_keys`), `datasets.py` (6 paper-calibrated builders + `PAPER_CONFIGS`), `discoverer.py` (`CsordasGumbelDiscoverer` via PyTorch 2.x `parametrize`). | Active |
| `lepori/` | `models.py` (`OddOneOutHead`, `CVRDecisionMLP`, `CVRResNetModel`, `CVRViTModel`, `SynEvalBertModel`, `CVRSimCLRModel`, `SimCLRLoss`, `pretrain_simclr`, `strip_lepori_l0_keys`), `datasets.py` (8 paper-calibrated builders + `PAPER_CONFIGS`), `_cvr_gen/` (vendored CVR procedural renderer, 3,355 lines), `_syneval_gen.py` (386 lines, Marvin & Linzen `RawData.zip` generator). | Active |
| `neurosurgeon/` | `adapter.py` (`_import_neurosurgeon` bypassing undeclared `torchmetrics` in `NeuroSurgeon.Probing`, `HFForwardShim`, `NeuroSurgeonWrappedModel`, `wrap_with_neurosurgeon`, `strip_neurosurgeon_mask_keys`) and `discoverer.py` (`NeuroSurgeonDiscoverer`). | Active |

**Architectural strengths:**
- Completely decoupled from `parallelazy` / `pipeline` — imports cleanly in standalone ML environments.
- Non-destructive discovery (`copy.deepcopy` before wrapping/parametrizing) with canonical dotted module keys throughout.
- `neurosurgeon/adapter.py:27-73` gracefully isolates `NeuroSurgeon.Probing`'s undeclared `torchmetrics` dependency and provides a clear `pip install positronixlab[baselines]` error message.
- `_syneval_gen.py` uses local `np.random.default_rng` (no global RNG mutation) and lazy `import requests`.

---

## 2. Bugs, Edge Cases & Hygiene Findings

### Medium Severity
1. **`CVRResNetModel` silently ignores `pretrained=True` when `use_instance_norm=True` (the default)**
   - **Location:** `positronixlab/baselines/lepori/models.py:149`
   - **Details:** `weights = "DEFAULT" if (pretrained and not use_instance_norm) else None` silently drops pretrained weights with no warning when callers pass `pretrained=True` with default `use_instance_norm=True`.
2. **`NeuroSurgeonWrappedModel.forward(*args, **kwargs)` bypasses `CircuitModel` for multi-arg positional calls**
   - **Location:** `positronixlab/baselines/neurosurgeon/adapter.py:136-141`
   - **Details:** When `len(args) > 1`, it calls `self.circuit_model.wrapped_model(*args, **kwargs)` directly, bypassing `CircuitModel.forward`.
3. **Missing optional-dependency hints for `transformers`, `cv2`, and `PIL`**
   - **Location:** `lepori/models.py:224, 288`, `lepori/_cvr_gen/shape.py:4`, `lepori/_cvr_gen/utils.py:4, 6`
   - **Details:** Raw top-level/inline imports raise bare `ModuleNotFoundError` instead of recommending `pip install "positronixlab[baselines]"`.

### Low Severity & Vendored Hygiene
4. **`CsordasGumbelDiscoverer` lacks reproducibility seed, `GradNormTracker`, and `n_mask_samples`** (`csordas/discoverer.py:17-31, 68-82`): `_gumbel_sigmoid_ste` draws from the global RNG with no configurable `seed`/`torch.Generator`.
5. **`MASKABLE_LAYER_TYPES` omits `nn.Conv3d`** (`base.py:20`), whereas `AtlasDiscoverer` explicitly supports `Conv1d/2d/3d`. Also `base.py:180` has a mid-file import of `CompositionalDataProvider`.
6. **Vendored `_cvr_gen/` hygiene:** `_cvr_gen/__init__.py:47-48` mutates global NumPy RNG via `np.random.seed(seed)`; `_cvr_gen/subnetwork_tasks.py:4` uses `from ...utils import *` (star import) and contains 13 `print()` statements; `shape.py` and `utils.py` have 6 unused local variables (`F841`).

---

## 3. Test Coverage Gaps

All 14 tests in `tests/baselines/` pass, covering registry listing, forward passes, OOO head, checkpoint auto-detection, NeuroSurgeon roundtrip, SimCLR pretraining, and paper-calibrated builders. Remaining gaps:
- `NeuroSurgeonDiscoverer` is only tested with `mask_method="continuous_sparsification"` (`tests/tasks/test_masks_and_discoverers.py:151-179`); `"hard_concrete"` and `"magnitude_pruning"` are untested.
- `ModelLoader.load()` with a file path (non-dict checkpoint) and `load_bundle()` error paths are untested.
- `CsordasGumbelDiscoverer` is not yet tested on `Conv2d` models or for sparsity monotonicity across epochs.
