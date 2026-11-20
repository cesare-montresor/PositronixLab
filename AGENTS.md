## Keeping this file current
Last synced: 2e947b8ea88711d13055ce7b2f001284cbb0c17d

At the start of every session:
1. Find commits since the last sync: `git log --oneline <Last synced hash>..HEAD`
   and `git diff --stat <Last synced hash>..HEAD`.
2. If those changes touched modules described below, check those sections against the
   code and fix anything stale. If nothing relevant changed, skip this.
3. Update `Last synced` only after a real review, not for unrelated edits.

Self-healing (any time during a session):
- If you find anything in this file that is wrong, outdated, or misleading, fix it
  immediately in the same session, even if unrelated to your current task.
- If you wasted effort rediscovering something a future agent would also have to
  rediscover (a gotcha, a hidden invariant, a command that needs special flags),
  add one concise line for it.
- Resolve `TODO(verify)` items when you can confirm them from code; otherwise leave
  them and ask the user.

At the end of every session:
- Review `git diff` of this session and propose edits to this file. Add only what
  meets the bar above. Remove stale or redundant lines. Prefer deleting or shortening
  over adding. Keep the file under ~150 lines.
- Do not log individual changes; git history does that.
- Commit AGENTS.md changes in their own commit: "docs: sync AGENTS.md".

---

## Purpose
`positronixlab` (Python `~=3.12`, PyTorch) is a research library providing the math, datasets, models, and tools to build a vector-space "brain atlas" for mapping, isolating, and understanding emergent functional modularity in neural networks. It projects computation units (CUs/neurons) into a decoupled embedding space via contrastive learning on forward activations, clusters CUs into computational groups (CGs), tracks cluster identities over time, and supports multi-objective NAS (Ax) and live Dash visualization.

## Role in the ecosystem
Split from a single monorepo into 4 sibling repos:
- **`../parallelazy`** (installed in `.venv` as `0.0.6`): standalone multiprocess worker/plugin framework (spawn/pipes, topic pub/sub, shared env). Do not reimplement plugin runtime or IPC here.
- **`../parallelazy-vscode`**: VS Code extension bridging `debugpy` for `spawn`ed worker processes (minimal maintenance).
- **`positronixlab` (this repo)**: reusable library only (spaces, math, observer, clustering, datasets, NAS, stats, UI components).
- **`../AtlasCompose`**: the only non-library application repo; combines `positronixlab` and `parallelazy` (concrete `plugin_*` wrappers and experiment entry points moved there in `4e70649`).

## Directory map
- `positronixlab/modularspace/` — CU embedding spaces (`ModularSpace`, `LayerSpace`, `ModularOverSpace`), cluster identity tracker (`custertracker.py` [sic]), `legacy/`.
- `positronixlab/spacemath/` — `SpaceMath`, `LayerType`, `ActivationType`, `sigreg.py`, `spaceloss.py`, and `common.py` (`torch.compile` wrappers, tensor monkey-patches).
- `positronixlab/modularobserver/` — `ModelMapper` (static + `TorchFunctionMode` dynamic graph tracing, Cytoscape export) and `ModularObserver` (layer/grad hooks).
- `positronixlab/modularcluster/` — Adaptive FAISS K-Means (`ModularClusterKmeans`), streaming K-Means, `streaming_cluster_tracker.py`.
- `positronixlab/deeplearning/` — `DeeplearningBase` (callback-driven batch/epoch loop), `DeeplearningTrain` (train/val/inference + `EarlyStopping`), `deeplearning_eval.py`.
- `positronixlab/metamodel/` — `BaseModel`/`MetaModel` (`nn_input -> nn_latent -> nn_output` built from `PipelineConfig`), `MetaCriterion`, `metatransform.py`.
- `positronixlab/pipeline/` — `Pipeline` ABC (wraps `parallelazy.PluginManager`) and `PipelineConfig` / `PipelineConfigV1` / `PipelineConfigV2`.
- `positronixlab/dataprovider/` — Auto-registering `DataReader`/`DataProvider` (`__init_subclass__`), `DataProviderFactory`, dataset providers (MNIST, FMNIST, CIFAR10/100, Higgs, Covertype, Iris, Math, SCAN, OddOneOut/CVR/SynEval).
- `positronixlab/template/` — `TemplateBase`/`TemplateVariant` separating tunable Ax `params` from `fixed` constants; dataset templates.
- `positronixlab/nas/` — `NasExperiment`: 2-phase Ax 1.3+ multi-objective NAS (parallel random `common` phase -> concurrent sequential BO `variant` phases, SQLite under `output/<exp>/databases/`).
- `positronixlab/stats/` — High-rate to low-rate aggregation (`Reducer`: Last/Ema/Mean/Window; `Processor`: Pca/Umap/Overall; `Stream`).
- `positronixlab/modularui/` — Thread-safe Dash/Plotly/Cytoscape view store (`ModularUI`, `UILocks`) consuming pre-aggregated stats.
- `positronixlab/tasks/` — Downstream evaluations (`tasks/compositionality/`: `atlas_check.py`, `weightmask_check.py`, `compositionality.py`).
- `positronixlab/logger/`, `positronixlab/utils/` — Process/thread-safe singleton `Logger` (stdout/stderr capture); misc helpers (`process_start_spawn`, `session_path`).

## Key abstractions and invariants
- **Decoupled space graph**: `ModularSpace`/`LayerSpace` never backprop into the main model. Activations are detached, mapped to `[-1, 1]` scores via per-sample quantile `active_q` (`SpaceMath.activations_to_scores`), and optimized per-layer with dedicated Adam optimizers on `positions: [num_cus, modular_space_size]`.
- **`torch.compile` split (`LayerSpace`)**: `_contrast_vectorized` and `_process_activations` use `@staticmethod_compile` (`mode="default"`) with `torch._dynamo.mark_static_address` on `positions`/`stats`; dynamic-shape masking (SIGReg) stays eager in `_process_scores` to avoid graph breaks.
- **Space implementations & config versions**: `ModularSpace` (`PipelineConfigV1`, `modular_space_loss_centers`) is currently the only working implementation. `LayerSpace` (`PipelineConfigV2`, `modular_space_loss_cluster`/`modular_space_loss_sigreg`/`modular_space_sigreg_method`) and `layerspace_process_scores*.py`/`layerspace_remixed.py` are WIP alternative implementations under revision. `PipelineConfig.from_dict()` silently drops unknown keys and coerces primitive types.
- **Observer filter**: `ModularObserver._is_observable_layer()` currently accepts `is_linear(module)` only (Conv/RNN/Transformer are mapped by `ModelMapper` but filtered out by the observer).
- **Model/Trainer contract**: `DeeplearningBase`/`DeeplearningTrain` require `model.device`; `inference()` requires `model.loss(x, y, y_hat)` (from `MetaModel`). `process_batch` accepts `(x, y)` or `(x, y, idx)` (from `IndexedDataset`).
- **Import side effects**: `spacemath/common.py` sets `torch.set_float32_matmul_precision('high')` and monkey-patches `torch.Tensor.{zclamp,zclamp_,spow,spow_}`. `logger/logger.py` globally filters out `UserWarning`.
- **CUDA multiprocessing**: `DataReader` selects `fork` only if CUDA is uninitialized, otherwise `spawn`; `utils.process_start_spawn()` enforces `spawn`.

## Commands
- Environment: `.venv/bin/python`, editable install (`pip install -e .`).
- Passing test suite: `.venv/bin/pytest positronixlab/logger/tests/test_logger.py -v` (31 passed).
- Full pytest: `.venv/bin/pytest -v` — currently fails collection on 2 files:
  - `dataprovider/tests/test_higgs.py`: `ImportError: cannot import name 'is_debugpy_attached' from 'parallelazy'` (imported by `pipeline/pipeline.py` against `parallelazy 0.0.6`).
  - `modularcluster/tests/test_kmeans_plugin.py`: `ModuleNotFoundError: positronixlab.modularcluster.modularcluster_plugin` (stale test from pre-split plugin code).
- K-Means tests: `.venv/bin/pytest positronixlab/modularcluster/tests/test_kmeans.py -v` — 11 passed, 3 failed (`test_print_stats`, `test_sync_euclidean`, `test_sync_cosine`) with `NotImplementedError: "bincount_cpu" not implemented for 'Float'` at `modularcluster_kmeans.py:260`.
- Lint / typecheck / CI: none configured (no ruff/black/mypy/CI files).

## Conventions and gotchas
- Keep typo in `modularspace/custertracker.py` (re-exported in `modularspace/__init__.py`) unless updating all callers.
- `stats/common.py:_as_tensor()` may return a view over recycled IPC/cudagraph buffers; callers retaining values must `.clone()`.
- Known WIP bugs / cleanup items:
  - `ModularSpace.get_space()` and `LayerSpace.get_space()` check `if None in (layer_info, positions, optimizer, stats)` — raises `ValueError` when `stats_enabled=False`.
  - `EarlyStopping.should_stop()` maximizes its metric (`gain = metric - self.metric_last`), even though `add_early_stopping()` defaults `metric="loss"`.
  - `.vscode/launch.json` and `modularcluster/tests/test_kmeans_plugin.py` still reference pre-split paths (`atlas_compose`, `plugin/examples`, `downstream_tasks`, `modularcluster_plugin`) and need cleanup.

## Do not touch
- Archived/legacy code: `positronixlab/*/legacy/`, `deeplearning/deeplearning_legacy.py`, `modularobserver/layerobserver_old.py`, `modularspace/modularspace_original.py`.
- Non-code / generated / fixtures: `draft/`, `papers/`, `data/`, `positronixlab.egg-info/`, `.obsidian/`, `modularui/examples/dump/data.pickle`, `tasks/compositionality/examples/{all_vars.pkl,trained_model.pth}`.

## Docs
- `positronixlab/components.md` — architecture & NAS overview (partially predates `parallelazy`/app split).
- `positronixlab/modularspace/docs/{layerloss.md,cluster_identity_tracking.md}` — `LayerSpace`/SIGReg and `ClusterTracker` design notes.
- `positronixlab/spacemath/docs/PARKING LOT PROBLEM.md` — embedding space geometry notes.
- Module docs: `dataprovider/docs/DATAPROVIDER_SYSTEM.md`, `logger/docs/LOGGER_{SYSTEM,USAGE}.md`, `modularcluster/docs/MODULARCLUSTER_{API,SYSTEM,USAGE}.md`, `stats/docs/STATS_{SYSTEM,USAGE}.md`, `template/docs/CONFIG_{SYSTEM,USAGE}.md`.
