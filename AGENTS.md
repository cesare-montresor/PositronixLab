## Keeping this file current
Last synced: b414936

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
`positronixlab` (Python `~=3.12`, PyTorch) is a research framework providing the math, datasets, models, SOTA baselines, and tools to build a vector-space "brain atlas" for mapping, isolating, and understanding emergent functional modularity in neural networks. It projects computation units (CUs/neurons) into a decoupled embedding space via contrastive learning on forward activations, clusters CUs into computational groups (CGs), tracks cluster identities over time, benchmarks against Csordás & Lepori/NeuroSurgeon weight-masking SOTA, and supports Ax 1.3+ multi-objective NAS and live Dash visualization.

## Role in the ecosystem
Split from a single monorepo into 4 sibling repos (plus SOTA reference symlinks under `libs/`):
- **`../parallelazy`** (installed editable in `.venv` as `0.2.1`): standalone multiprocess worker/plugin framework (spawn/pipes, topic pub/sub, shared env, logging). Do not reimplement plugin runtime or IPC here.
- **`../parallelazy-vscode`**: VS Code extension bridging `debugpy` for `spawn`ed worker processes.
- **`positronixlab` (this repo)**: reusable library only (spaces, math, observer, clustering, datasets, models, SOTA baselines, NAS, stats, UI components).
- **`../AtlasCompose`**: application repo combining `positronixlab` and `parallelazy`.
- **`libs/`**: reference symlinks (`are_neural_nets_modular`, `Compositional_Subnetworks`, `NeuroSurgeon`, `parallelazy`).

## Directory map
- `positronixlab/` — Pure library source code (no `tests/`, `docs/`, `examples/`, or `legacy/`):
  - `spacemath/` — Modular primitives (`scoring.py`, `contrast.py`, `weighting.py`, `losses.py`, `reducers.py`, `space_type.py`, `sigreg.py`), `SpaceMath`, and `common.py` (`torch.compile` wrappers, tensor helpers, `effective_rank`).
  - `modularspace/` — `LayerSpace`, `LayerSpaceConfig`, `SpaceStep`, `presets.py`, `ModularSpace`, `ModularOverSpace`, and `custertracker.py` [sic].
  - `modularobserver/` — `ModelMapper` (static + `TorchFunctionMode` dynamic graph tracing, Cytoscape export), `ModularObserver` (configurable `observable_types` / `target_layers` / `exclude_patterns`), `LayerObserver`, and `ActivationCache` (`activation_cache.py`, lazy-exported, no `parallelazy`).
  - `modularcluster/` — Adaptive FAISS K-Means (`ModularClusterKmeans`), streaming K-Means, `StreamingClusterTracker`.
  - `deeplearning/` — `TrainConfig`, `DeeplearningBase`, `DeeplearningTrain` (+ `EarlyStopping`), `DeeplearningEval`.
  - `metamodel/` — `MetaModelConfig`, `BaseModel`/`MetaModel`, `MetaCriterion`, `MetaTransform`.
  - `dataprovider/` — `DataConfig`, `DataReader`/`DataProvider`, `CompositionalDataProvider`, `DataProviderFactory`, `FastTensorLoader` (`fast_loader.py`), 16 concrete providers (incl. `ClassRemovalDataProvider`, `AddMul`, `TupleArithmetic`, `PermutedMnist`, `SCAN`, `Math`, `CVR`/`CVRSimCLR`/`SynEval`).
  - `tasks/` — Shared evaluation contract (`SubnetworkMask`, `MaskApplier`, `SubnetworkDiscoverer`, `RandomMaskDiscoverer`), PositronixLab's `AtlasDiscoverer`, `ModularityBenchmark` + `make_accuracy_fn`/`resolve_subtask_criterion` (`benchmarks.py`), and `compositionality/` (earlier prototype).
  - `baselines/` — External SOTA works organized by paper (`csordas/`, `lepori/` incl. pure-PyTorch `ContinuousSparsificationDiscoverer`, `neurosurgeon/`) + shared `ModelLoader` (`save`/`load_bundle`/`load`/`infer_kwargs`), `DataProviderLoader`, `CheckpointStore`/`BaselineArtifactLoader` (`artifacts.py`), `SOTAModel`, `CompositionalDataProvider`.
  - `pipeline/` — `BaseConfig` (`config.py`, YAML/JSON + tuple & PEP 604 optional coercion), `Pipeline` ABC (wraps `parallelazy.PluginManager`), `PipelineConfig`/`V1`/`V2`.
  - `nas/` — Ax 1.3+ `Experiment`, `TrialConfig`, `TrialPipeline`.
  - `template/` — `TemplateBase`, `TemplateVariant`, `TemplateFactory`, dataset templates.
  - `stats/` — `Reducer` (`Last`/`Ema`/`Mean`/`Window`/`Displacement`), `Processor` (`Umap`/`Overall`), `DisplacementFilter`, `Stream`.
  - `modularui/` — Thread-safe Dash/Plotly/Cytoscape view store (`ModularUI`, `UILocks`, `layouts`, `docks`, `components`, `plots`).
  - `utils/` — Misc helpers (`process_start_spawn`, `seed_everything`, `session_path`, `save_data`/`load_data`, `atomic_write_json`, `to_plain_python`).
- `tests/` — Unit tests mirroring `positronixlab/` (`pytest.ini` sets `testpaths = tests`).
- `docs/` — Per-module `*_SYSTEM.md` / `*_USAGE.md` docs, Ax guides, `components.md`, and media.
- `examples/` — Standalone runnable CPU examples per module (incl. `examples/tasks/example_atlas_vs_sota.py`).
- `reports/` — Per-module `*_analysis.md` audits (all 15 modules incl. `baselines_analysis.md`), `overall_analysis.md`, `xfail_analysis.md`, `SOTA_integration.md`, and `positronixlab_review_v0.1.3.md`.

## Key abstractions and invariants
- **Decoupled space graph**: `ModularSpace`/`LayerSpace` never backprop into the main model. Activations are detached, mapped to `[-1, 1]` scores, and optimized per-layer with dedicated Adam optimizers on `positions: [num_cus, modular_space_size]`.
- **Composable configs & lazy exports**: `pipeline/config.py:BaseConfig` underpins `DataConfig`, `TrainConfig`, `MetaModelConfig`, `LayerSpaceConfig`, and `TrialConfig`. `dataprovider/__init__.py`, `metamodel/__init__.py`, and `pipeline/__init__.py` use PEP 562 lazy `__getattr__` so standalone ML/SOTA imports never require `parallelazy`.
- **Unified baselines contract**: `positronixlab.baselines` (`ModelLoader`, `DataProviderLoader`, `csordas/`, `lepori/`, `neurosurgeon/`) and `positronixlab.tasks` (`SubnetworkMask`, `MaskApplier`, `AtlasDiscoverer`) allow apples-to-apples comparison of Atlas, NeuroSurgeon, and Csordás Gumbel-Sigmoid on identical models and datasets.
- **NeuroSurgeon import quirk**: `libs/NeuroSurgeon` has an undeclared `torchmetrics` import in `Probing`; `baselines/neurosurgeon/adapter.py` automatically bypasses `Probing` so `Models` and `Masking` work without `torchmetrics`.
- **Import side effects**: `spacemath/common.py` sets `torch.set_float32_matmul_precision('high')` and monkey-patches `torch.Tensor.{zclamp,zclamp_,spow,spow_}`. `spacemath/contrast.py` runs `torch.compile` (disable via `POSITRONIXLAB_COMPILE=0`, or set `TRITON_CACHE_DIR=/tmp/triton_cache TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache` in read-only-home environments).
- **CUDA multiprocessing**: `DataReader` selects `fork` only if CUDA is uninitialized, otherwise `spawn`; `utils.process_start_spawn()` enforces `spawn`.

## Commands
- Environment: `.venv/bin/python` (Python 3.12, editable install).
- Full test suite: `CUPY_CACHE_DIR=/tmp/cupy_cache TRITON_CACHE_DIR=/tmp/triton_cache TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache .venv/bin/pytest -q` (runs all suites under `tests/`; `CUPY_CACHE_DIR` is needed in read-only-home sandboxes when `cuml`/`cupy` is installed; requires access to sibling `../parallelazy` and `libs/NeuroSurgeon` for full pass).
- Baselines & Atlas tests only: `TRITON_CACHE_DIR=/tmp/triton_cache TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache .venv/bin/pytest tests/baselines tests/tasks/test_masks_and_discoverers.py tests/tasks/test_atlas_discoverer.py tests/tasks/test_sota_benchmarks.py -v`.
- Head-to-head Atlas vs SOTA CPU benchmark: `.venv/bin/python examples/tasks/example_atlas_vs_sota.py`.
- Archived pre-cleanup snapshot: git tag `archive/pre-cleanup`.

## Conventions and gotchas
- Keep `drop_last=True` as the default across `DataProvider`/`DataReader` splits (`drop_last=False` causes partial-batch crashes downstream).
- Keep typo in `modularspace/custertracker.py` (re-exported in `modularspace/__init__.py`) unless updating all callers.
- `stats/common.py:_as_tensor()` may return a view over recycled IPC/cudagraph buffers; callers retaining values must `.clone()`.
- Known remaining bugs (P1/P2) are cataloged in `reports/overall_analysis.md` and pinned via `@pytest.mark.xfail(strict=True)` in `tests/`.

## Do not touch
- Non-code / generated / fixtures: `draft/`, `papers/`, `data/`, `positronixlab.egg-info/`, `.obsidian/`, `examples/modularui/dump/data.pickle`, `examples/tasks/compositionality/{all_vars.pkl,trained_model.pth}`.
