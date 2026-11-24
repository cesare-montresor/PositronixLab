# PositronixLab — Overall Codebase Analysis

> **Scope:** Comprehensive, report-only review of all 14 active modules in `positronixlab/` on the `layerspace` branch (excluding `legacy/` folders, archived files, and the removed `logger/` module).
> **Policy:** Zero library source `.py` files were modified during this review. Confirmed defects are documented in each module's `analysis.md` and pinned with `@pytest.mark.xfail(strict=True)` regression tests.

---

## 1. Executive Summary

`positronixlab` is mid-transition from a monolithic, `PipelineConfig`-bound research codebase into a **modular, composable framework for exploring emergent functional modularity** in neural networks. The recent refactor on the `layerspace` branch made strong architectural strides:
- Decomposing `spacemath` into composable primitives (`scoring`, `contrast`, `weighting`, `losses`, `reducers`, `space_type`, `sigreg`).
- Introducing composable `BaseConfig` mixins (`DataConfig`, `TrainConfig`, `MetaModelConfig`, `LayerSpaceConfig`, `TrialConfig`).
- Rebuilding `nas` around a clean, single-phase Ax 1.3+ `Experiment` / `TrialPipeline` / `TrialConfig` architecture.
- Introducing `LayerSpace`, `SpaceStep`, and `presets` in `modularspace`.

However, because the refactor is in-flight, **six systemic cross-cutting issues** currently undermine reliability, ergonomics, and standalone usability:

1. **Transitive `parallelazy` coupling across pure-PyTorch modules:** `pipeline/pipeline_config.py:6` imports `parallelazy.plugin_data.ManagerOptions` at module top-level. Because `dataprovider`, `deeplearning`, `metamodel`, `modularobserver`, and `modularspace` still import `PipelineConfig` at runtime (often solely for type annotations), importing any of these core ML modules fails with `ModuleNotFoundError: No module named 'parallelazy'` whenever `parallelazy` is absent or inaccessible.
2. **Half-migrated `BaseConfig` vs `PipelineConfigV1`/`V2` split:** Five modules now define clean `BaseConfig` mixins (`config.py`), yet their runtime classes (`DataProviderFactory`, `DeeplearningBase`, `MetaModel`, `MetaCriterion`, `ModularObserver`, `ModularSpace`) still expect legacy `PipelineConfig` attributes (e.g., `config.task`, `config.model_type`), while `TemplateAtlas` still targets V1 knobs and `BaseConfig.from_dict()` fails to coerce runtime `Optional[...]` and PEP 604 `int | None` types.
3. **High-severity silent correctness bugs in core math, training, clustering, and evaluation:**
   - `spacemath`: negative-row score explosion to `-2e8` (`score_normalize_by_threshold`), cosine centroid scale-invariance bug (`_distance_contrast_core`), and `is_linear(nn.Identity()) == True`.
   - `modularcluster`: `float32` labels crashing `bincount_cpu`, `dist.fill_diagonal_(0.0)` corrupting rectangular `(n, k)` distance matrices in `predict()` and every streaming `fit()`, and Hungarian relabeling without reordering centroids.
   - `deeplearning`: `EarlyStopping` maximizing `metric="loss"` by default, `inference()` under-reporting `avg_loss` by `~batch_size` and crashing on 3-tuple `(x, y, idx)` batches.
   - `nas`: `Client.mark_trial_failed(reason=...)` instead of Ax 1.3's `failed_reason=...` crashing on every failure/timeout/reconciliation, plus an unbounded retry loop on persistent failures.
   - `tasks`: `AtlasCheck.extract_weightmasks()` (dotted keys, 1D bool) is incompatible with `StaticMaskApplier` (underscored keys, 2D float), `WeightmaskCheck.run()` ignores `num_epochs`, and `AtlasCheck` leaks forward hooks permanently.
4. **Hook & resource lifecycle leaks:** `modularobserver` has a late-binding closure bug in `on_grad_weights`, leaks handles on re-registration, and omits pre-forward hooks in `clear()`; `tasks/compositionality` leaks hooks and never restores `param.requires_grad`.
5. **Broken import-time & environment assumptions:** Unconditional `from cuml.manifold.umap import UMAP` in `stats/processor.py:14` crashes CPU-only imports of `Stream` and `OverallProcessor`; unconditional `@common.compile` in `spacemath/contrast.py` fails on read-only `~/.triton`; `utils.load_data()` calls `yaml.load(f)` without `Loader` (broken on PyYAML 6); `metamodel.py` eagerly imports `torchsummary`.
6. **Inconsistent package surfaces:** Six modules (`modularobserver`, `modularcluster`, `stats`, `modularui`, `tasks`, `utils`) have completely empty `__init__.py` files, while others (`deeplearning`, `dataprovider`, `metamodel`) perform heavy eager imports at `__init__.py` time.

---

## 2. Module-by-Module Health Matrix

| Module | Refactor Status | Critical / High Bugs | Report | Docs | Tests (Passed / XFail) | Runnable Example |
| :--- | :--- | :---: | :--- | :--- | :--- | :--- |
| **`spacemath`** | Core primitives refactored; legacy `SpaceMath` still used | 7 High | [analysis.md](file:///home/cesare/Projects/PositronixLab/positronixlab/spacemath/analysis.md) | `SPACEMATH_SYSTEM.md`, `SPACEMATH_USAGE.md`, `PARKING LOT PROBLEM.md` | 40 passed, 12 xfail | `examples/example_spacemath.py` |
| **`modularspace`** | `LayerSpace`/`SpaceStep`/`presets` new; `layerspace_old.py` & `ModularSpace` still present | 6 High | [analysis.md](file:///home/cesare/Projects/PositronixLab/positronixlab/modularspace/analysis.md) | `MODULARSPACE_SYSTEM.md`, `MODULARSPACE_USAGE.md`, `layerloss.md`, `cluster_identity_tracking.md` | 36 passed, 2 skipped, 3 xfail | `examples/example_layerspace.py` |
| **`pipeline`** | `BaseConfig` added; `PipelineConfigV1`/`V2` still coupled to `parallelazy` | 4 High/Med | [analysis.md](file:///home/cesare/Projects/PositronixLab/positronixlab/pipeline/analysis.md) | `PIPELINE_SYSTEM.md`, `PIPELINE_USAGE.md` | 30 passed, 3 xfail | `examples/example_pipeline_config.py` |
| **`nas`** | Rewritten for Ax 1.3 (`Experiment`, `TrialConfig`, `TrialPipeline`) | 6 High/Med | [analysis.md](file:///home/cesare/Projects/PositronixLab/positronixlab/nas/analysis.md) | `NAS_SYSTEM.md`, `NAS_USAGE.md` (+ root Ax guides) | 14 passed, 8 xfail | `examples/example_nas_trial.py` |
| **`metamodel`** | `MetaModelConfig` added; runtime still uses `PipelineConfig` | 6 High/Crit | [analysis.md](file:///home/cesare/Projects/PositronixLab/positronixlab/metamodel/analysis.md) | `METAMODEL_SYSTEM.md`, `METAMODEL_USAGE.md` | 27 passed, 12 xfail | `examples/example_metamodel.py` |
| **`deeplearning`** | `TrainConfig` added; runtime still reads `config.task` | 3 High, 6 Med | [analysis.md](file:///home/cesare/Projects/PositronixLab/positronixlab/deeplearning/analysis.md) | `DEEPLEARNING_SYSTEM.md`, `DEEPLEARNING_USAGE.md` | 22 passed, 11 xfail | `examples/example_deeplearning_train.py` |
| **`dataprovider`** | `DataConfig` added; factory & providers not yet migrated | 6 High, 7 Med | [analysis.md](file:///home/cesare/Projects/PositronixLab/positronixlab/dataprovider/analysis.md) | `DATAPROVIDER_SYSTEM.md`, `DATAPROVIDER_USAGE.md` | 24 passed, 7 xfail (`test_higgs.py` stale) | `examples/example_dataprovider.py` |
| **`modularobserver`** | Pre-refactor; linear-only hooks, empty `__init__.py` | 6 High | [analysis.md](file:///home/cesare/Projects/PositronixLab/positronixlab/modularobserver/analysis.md) | `MODULAROBSERVER_SYSTEM.md`, `MODULAROBSERVER_USAGE.md` | 21 passed, 11 xfail | `examples/example_modularobserver.py` |
| **`modularcluster`** | Pre-refactor; duplicated batch/stream KMeans, stale plugin test | 4 High, 2 Med | [analysis.md](file:///home/cesare/Projects/PositronixLab/positronixlab/modularcluster/analysis.md) | `MODULARCLUSTER_API.md`, `MODULARCLUSTER_SYSTEM.md`, `MODULARCLUSTER_USAGE.md` | 9 passed, 6 xfail (`test_kmeans.py`: 11P/3F; `test_kmeans_plugin.py` stale) | `examples/example_modularcluster.py` |
| **`stats`** | Partial refactor (`PcaProcessor` removed, `Displacement` added); broken cuML import | 4 High | [analysis.md](file:///home/cesare/Projects/PositronixLab/positronixlab/stats/analysis.md) | `STATS_SYSTEM.md`, `STATS_USAGE.md` | 22 passed, 15 xfail | `examples/example_stats.py` |
| **`modularui`** | Active Dash UI; broken CSS paths & full-DOM rebuilds | 8 High/Med | [analysis.md](file:///home/cesare/Projects/PositronixLab/positronixlab/modularui/analysis.md) | `MODULARUI_SYSTEM.md`, `MODULARUI_USAGE.md` | 20 passed, 8 xfail | `examples/example_modularui_headless.py` |
| **`template`** | Still V1-oriented; orphaned `TemplateVariant` | 3 High, 3 Med | [analysis.md](file:///home/cesare/Projects/PositronixLab/positronixlab/template/analysis.md) | `CONFIG_SYSTEM.md`, `CONFIG_USAGE.md` | 21 passed, 7 xfail | `examples/example_template.py` |
| **`tasks`** | Early prototype; incompatible mask formats & hook leaks | 7 High | [analysis.md](file:///home/cesare/Projects/PositronixLab/positronixlab/tasks/analysis.md) | `TASKS_SYSTEM.md`, `TASKS_USAGE.md` | 10 passed, 9 xfail | `examples/example_compositionality.py` |
| **`utils`** | Utility grab-bag; PyYAML 6 crash & broken `force=True` | 2 High, 5 Med | [analysis.md](file:///home/cesare/Projects/PositronixLab/positronixlab/utils/analysis.md) | `UTILS_SYSTEM.md`, `UTILS_USAGE.md` | 27 passed, 9 xfail | `examples/example_utils.py` |

**Combined new/verified test suite totals:** **325+ passing tests** and **122 strict `xfail` regression tests** pinning confirmed bugs across all 14 modules (0 unexpected failures when `parallelazy` is accessible; in isolated sandbox runs without `../parallelazy`, 2 `nas` tests that instantiate `PluginManager` fail if earlier modules installed a minimal `parallelazy` stub).

---

## 3. Priority Fix Roadmap (Recommended Execution Order)

### P0 — Unblock Standalone Imports & Silent Correctness Crashes
1. **Decouple `BaseConfig` & core ML modules from `parallelazy`** (`pipeline/pipeline_config.py`, `dataprovider`, `deeplearning`, `metamodel`, `modularobserver`, `modularspace`):
   - Move `PipelineOptions` out of `pipeline_config.py` (or lazy-import `ManagerOptions`).
   - Guard `PipelineConfig` imports under `if TYPE_CHECKING:` in all ML modules.
2. **Fix `nas/experiment.py` Ax 1.3 API crash**:
   - Change `reason=` to `failed_reason=` in all 4 `client.mark_trial_failed()` calls (`experiment.py:215, 369, 374, 377`).
   - Move `_reconcile_interrupted_trials()` outside `_init_or_resume_client()`'s `try/except`, and add a `max_consecutive_failures` circuit breaker to `Experiment.run()`.
3. **Fix `spacemath` math & layer-predicate bugs**:
   - Shift active scores around `threshold` in `score_normalize_by_threshold` (`scoring.py:10-24`) to stop `-2e8` explosions on negative activations.
   - Normalize `positions` before computing centroids in `_distance_contrast_core` (`contrast.py:65-69`) for cosine geometry consistency.
   - Replace `__module__ == 'torch.nn.modules.linear'` in `common.py:61-75` with `isinstance(m, nn.Linear)` (stops `nn.Identity()` being treated as a linear layer).
4. **Fix `modularcluster` distance & label corruption**:
   - Use `torch.as_tensor(labels, dtype=torch.long)` in `modularcluster_kmeans.py:128` (immediately fixes all 3 failing `test_kmeans.py` tests).
   - Only `fill_diagonal_(0.0)` in `_distance(x, y)` when `x is y` (`modularcluster_kmeans.py:90`, `modularcluster_kmeans_stream.py:98`).
   - Reorder `centroids` alongside `labels` in `StreamingClusterTracker._relabel_to_match_previous()` (`streaming_cluster_tracker.py:123-149`).
   - Convert CUDA tensors via `.detach().cpu().numpy()` before passing to FAISS.
5. **Fix `deeplearning` training/eval correctness**:
   - Add `minimize` flag, `metric_last = None`, and `>= max_attempts` to `EarlyStopping` (`deeplearning_train.py:218-246`).
   - Fix sample-weighted loss and 2/3-tuple unpacking in `DeeplearningTrain.inference()` (`deeplearning_train.py:168-204`).
6. **Fix `stats/processor.py:14` & `utils/utils.py:217`**:
   - Remove the unconditional top-level `from cuml.manifold.umap import UMAP` in `stats/processor.py:14` (already handled in `try/except` at line 36).
   - Switch `utils.load_data()` to `yaml.safe_load(f)` (`utils.py:217-219`).

### P1 — Fix Hook Lifecycles, Model Construction, Data Splits, and Evaluation Contracts
7. **Fix `modularobserver` hook safety**:
   - Fix late-binding closure (`m=module_ref, p=param`) in `on_grad_weights` (`modularobserver.py:125`, `layerobserver.py:42`).
   - Remove existing hook handles before overwriting; call `on_pre_forward_layer(None)` inside `ModularObserver.clear()`; make `ModelMapper.map()` device-agnostic.
8. **Fix `metamodel` builder & metric bugs**:
   - Rename `self.custom_metric = {}` dict in `MetaMetric` (`metacriterion.py:45`) so it doesn't shadow `def custom_metric(...)`.
   - Fix `build_model_output_block` (`output_num_layers == 1`, `output_hidden_dim`), omit activation/BN/dropout on final output layers, fix CNN-to-FC `Flatten` + `BatchNorm`, and respect `device` in `MetaModel.__init__`.
9. **Fix `dataprovider` leakage & splits**:
   - Fix `DataProviderFactory.from_config()` to accept `DataConfig` and not pass `"split": None` overriding 3-way splits.
   - Fix `IrisDataProvider` train/val/test leakage, `HIGGS` `.npy.tmp` rename bug + `persistent_workers` epoch staleness, and `DataReader` split-appropriate `shuffle`/`drop_last`.
10. **Fix `tasks/compositionality` mask contract**:
    - Unify mask keys (dotted names) and shapes (1D neuron masks vs 2D weight masks) between `AtlasCheck`, `WeightmaskCheck`, and `StaticMaskApplier`.
    - Add the `num_epochs` loop and empty-class guard to `WeightmaskCheck.run()`, and add `try/finally` hook & `requires_grad` cleanup.
11. **Fix `modularspace` & `template` bugs**:
    - Guard `loss.requires_grad` in `LayerSpace.process_scores()` and deduplicate `LayerSpace` onto `SpaceStep`; fix `ModularSpace.get_space()` (`stats_enabled=False`) and `pairwise=False` in center repulsion; fix `ClusterTracker` `k_old == 0` and `switch` split-off.
    - Fix `TemplateBase.add_choice(..., log_scale=...)` Ax 1.3 `TypeError`, `TemplateVariant.matches_trial_parameters()` missing-key bug, and `TemplateFMnist*` `'fmnist'` vs `'fashion_mnist'` mismatch.
12. **Fix `modularui` stability**:
    - Remove broken `themes/*.css` from `external_stylesheets`, fix `_filtered_layer_ids` (`visible_layers is None`), deep-copy `default_state`, widen `observed_layers` lock scope, and clear stale centroids/colors on scrub-back.

### P2 — Framework Ergonomics & Architectural Consolidation
13. **Finish the `BaseConfig` migration**:
    - Upgrade `BaseConfig.from_dict()` to use `typing.get_type_hints()` + `get_origin`/`get_args` (supporting `Optional[T]` and `T | None`), and validate `isinstance(data, dict)` in `BaseConfig.load()`.
    - Migrate `TemplateAtlas` to V2 (`LayerSpaceConfig` knobs) and retire `PipelineConfigV1`/`V2` once all consumers accept composable mixins.
14. **Standardize package exports & lazy imports**:
    - Populate `__init__.py` in `modularobserver`, `modularcluster`, `stats`, `modularui`, `tasks`, `utils`.
    - Add PEP 562 lazy imports in heavy packages (`deeplearning`, `dataprovider`, `metamodel`, `pipeline`) and an opt-out toggle (`POSITRONIX_COMPILE=0`) for `@common.compile` in `spacemath`.
15. **Consolidate duplicates & remove stale artifacts**:
    - Merge `ModularClusterKmeansStream` into `ModularClusterKmeans`, unify `DisplacementFilter` and `DisplacementReducer` in `stats`, archive `layerspace_old.py` and `custertracking.example.py` to `legacy/`, move `nas/ax_*.md` into `nas/docs/`, and remove or update stale `modularcluster/tests/test_kmeans_plugin.py` and `dataprovider/tests/test_higgs.py`.
16. **Expand modularity coverage beyond Linear layers**:
    - Make `ModularObserver._is_observable_layer()` configurable (Conv/RNN/Transformer) and add sequence reducers (`(B, T, U) -> (B, U)`) in `spacemath/reducers.py` so the framework generalizes cleanly beyond MLPs.

---

## 4. `AGENTS.md` Drift Report (Updates Needed After `layerspace` Refactor)

`AGENTS.md` was last synced at `2e947b8ea88711d13055ce7b2f001284cbb0c17d` and has drifted significantly from the current `layerspace` working tree:

| Section in `AGENTS.md` | Current Stale Text | Actual State on `layerspace` Branch |
| :--- | :--- | :--- |
| **Role in the ecosystem** | `../parallelazy` installed in `.venv` as `0.0.6` | Upgraded to editable `parallelazy 0.2.1` pointing to `../parallelazy`. |
| **Directory map (`logger/`)** | Lists `positronixlab/logger/` and `Logger` singleton | `positronixlab/logger/` has been **deleted** in working tree (logging now provided via `parallelazy`). |
| **Directory map (`spacemath/`)** | Lists `spaceloss.py` | `spaceloss.py` moved to `spacemath/legacy/`; replaced by modular primitives: `scoring.py`, `contrast.py`, `weighting.py`, `losses.py`, `reducers.py`, `space_type.py`, plus refactored `sigreg.py`. |
| **Directory map (`modularspace/`)** | Lists `layerspace_process_scores*.py` / `layerspace_remixed.py` | Those files were deleted; replaced by `config.py` (`LayerSpaceConfig`), `presets.py`, `space_step.py`, refactored `layerspace.py`, and `layerspace_old.py`. |
| **Directory map (`nas/`)** | Describes `NasExperiment`: 2-phase Ax NAS (`common` -> `variant`), `output/<exp>/databases/` | Legacy 2-phase runner moved to `nas/legacy/`; replaced by `Experiment`, `TrialConfig`, `TrialPipeline` (`experiment.py`, `trial_config.py`, `trial_pipeline.py`). |
| **Directory map (`pipeline/`, `dataprovider/`, `deeplearning/`, `metamodel/`)** | Only mentions `PipelineConfig`/`V1`/`V2` | Composable `BaseConfig` (`pipeline/config.py`) and per-module mixins (`DataConfig`, `TrainConfig`, `MetaModelConfig`, `LayerSpaceConfig`) now exist alongside `V1`/`V2`. |
| **Directory map (`stats/`)** | Mentions `Processor`: `Pca/Umap/Overall` | `PcaProcessor` was removed; `DisplacementFilter` and `DisplacementReducer` were added. |
| **Key abstractions (`Import side effects`)** | Mentions `logger/logger.py` filtering `UserWarning` | `positronixlab/logger/logger.py` no longer exists. |
| **Commands (`Passing test suite` / `Full pytest`)** | Lists `logger/tests/test_logger.py` (31 passed) and `is_debugpy_attached` ImportError against `parallelazy 0.0.6` | `test_logger.py` is deleted; `parallelazy 0.2.1` now has `is_debugpy_attached` (the sandbox collection error is due to `../parallelazy` being outside workspace); all 14 modules now have unit tests under `positronixlab/*/tests/` (325+ passing, 122 strict xfails). Also note `TRITON_CACHE_DIR=/tmp/triton_cache TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache` is needed in read-only-home environments when `torch.compile` runs in `spacemath/contrast.py`. |
| **Conventions & gotchas** | Lists `LayerSpace.get_space()` `stats_enabled=False` bug | That check now applies to `ModularSpace.get_space()` (`modularspace.py:59`); `LayerSpace` was rewritten and no longer has that check (though it has new issues documented in `modularspace/analysis.md`). |
| **Docs** | Lists only 5 module docs | All 14 active modules now have `analysis.md`, `docs/*_SYSTEM.md`, `docs/*_USAGE.md`, `tests/`, and runnable `examples/`. |
