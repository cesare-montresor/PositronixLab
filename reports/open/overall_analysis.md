# PositronixLab — Overall Codebase Analysis

> **Scope:** Global verification of all **15 active modules** in `positronixlab/` on `main` (`v0.1.3`, commit `5740a0c` + working tree changes in `tasks/masks.py` and `tasks/atlas_discoverer.py`).
> **Report Organization (2026-11-27):** All resolved issues have been archived under [`reports/solved/`](file:///home/cesare/Projects/PositronixLab/reports/solved/), so every `reports/*_analysis.md`, `positronixlab_review_v0.1.3.md`, and `SOTA_integration.md` roadmap now lists **only remaining open issues**.
> **Solved Issues Archive:** All resolved bugs and completed milestones have been moved into [`reports/solved/`](file:///home/cesare/Projects/PositronixLab/reports/solved/), leaving each active `reports/*_analysis.md` file with **open issues only**.
> **Verification Date:** 2026-11-27.
> **Commands Executed:**
> - Full test suite: `CUPY_CACHE_DIR=/tmp/cupy_cache TRITON_CACHE_DIR=/tmp/triton_cache TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache .venv/bin/pytest -v --tb=short`
> - Static analysis: `ruff check positronixlab --select F,B,PLE,PLW,RUF,S,E7,E9,A,PERF --ignore F403,F405,F401,RUF001,RUF002,RUF003,S101,E741,RUF012 --statistics`
> - CPU SOTA benchmark: `TRITON_CACHE_DIR=/tmp/triton_cache TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache .venv/bin/python examples/tasks/example_atlas_vs_sota.py` (exit 0)

---

## 1. Executive Summary

Since the initial `layerspace`-branch audit, `positronixlab` has undergone a **major stabilization and SOTA integration milestone** (releases `v0.1.0` → `v0.1.3` + `5740a0c` + working tree):
- **New `positronixlab/baselines/` module (15th module):** Unified `ModelLoader` (12 architectures), `DataProviderLoader` (14 datasets), and discoverers for **Csordás et al.** (`CsordasGumbelDiscoverer`), **Lepori et al.** (CVR/SimCLR/SynEval), and **NeuroSurgeon** (`NeuroSurgeonDiscoverer`, bypassing `torchmetrics`).
- **Unified `positronixlab/tasks/` evaluation contract:** Canonical `SubnetworkMask`, exception-safe & reproducible `MaskApplier` (including working-tree `bias_mode`), `AtlasDiscoverer` (including working-tree differential `attribution_mode` & `energy_mode`), and `ModularityBenchmark`.
- **`dataprovider/` completely refactored:** 44/44 tests pass (0 xfails); PEP 562 lazy exports, registry-driven `DataProviderFactory`, `CompositionalDataProvider`, 6 new providers, and real implementations for `SCAN`, `Math`, `CVR`, and `SynEval`.
- **High-severity P0 bugs resolved across the stack:** `positronixlab.__version__` (A1), `nas` Ax 1.3 `failed_reason=` crash, `spacemath` v2 negative-row explosion & cosine centroids & `is_linear(nn.Identity)` & `POSITRONIXLAB_COMPILE` switch (A10), `modularcluster` `torch.long` labels & diagonal corruption & KMeans edge cases (A6), `deeplearning` `EarlyStopping` & `inference()`, `stats` unconditional `cuml` import, `modularobserver` `on_grad_weights` closure & `clear()` & `ModelMapper.map()` device fallback (A5), `utils` `load_data` YAML & thread-safe `suppress_output` & `session_path` (A7–A9), and `tasks` A2/A3/A4/B1/B2/B3/C1/D1.
- **Package surfaces populated:** All 6 previously empty `__init__.py` files (`modularobserver`, `modularcluster`, `stats`, `modularui`, `tasks`, `utils`) are now populated, and `dataprovider`, `metamodel`, `modularobserver`, `modularspace`, `modularui`, `nas`, and `pipeline` use PEP 562 lazy exports.

### Current Full-Suite Verification Results

**Pytest (2026-11-27 — all P0 bugs solved!):** **462 passed, 71 xfailed, 2 failed** across 535+ collected tests *(ZERO real test failures — the only 2 failures are sandbox-only `../parallelazy` stub errors in `tests/nas/test_nas.py`; see [xfail_analysis.md](file:///home/cesare/Projects/PositronixLab/reports/open/xfail_analysis.md) for the remaining 71 P1/P2 `xfail` markers)*.

**Ruff (`ruff check positronixlab tests`):** **0 hits** (all checks passed).
**Typing (`mypy` / `pyright`):** **0 mypy errors**, **0 pyright errors** (see [`reports/typing/typing_summary.md`](file:///home/cesare/Projects/PositronixLab/reports/typing/typing_summary.md)).

**Head-to-head CPU benchmark (`examples/tasks/example_atlas_vs_sota.py`):** Passes (exit 0). With working-tree differential selectivity (`attribution_mode="selectivity"`), Atlas achieves `IoU(add, mul) = 0.133` at `43.3%` sparsity (vs. NeuroSurgeon `0.475` at `53.3%`, Csordás `1.000` at `0.0%`).

**Static analysis (`ruff`):** 112 findings across the selected rule set — notably **6 duplicate dict keys (`F601`)** in `dataprovider/__init__.py:144-149`, 40 one-line `if`s (`E701`), 13 implicit `Optional` (`RUF013`), 10 unused locals (`F841`), 10 redefined loop names (`PLW2901`), and `print()` statements in library code.

---

## 2. Module-by-Module Health Matrix (15 Active Modules)

| Module | Current Status | Remaining High/Med Issues | Report | Pytest (Pass / XFail / Fail) |
| :--- | :--- | :---: | :--- | :---: |
| **`baselines`** *(new)* | Complete SOTA hub (Csordás, Lepori, NeuroSurgeon, SimCLR, `ModelLoader`, `DataProviderLoader`) | 4 Med | [baselines_analysis.md](file:///home/cesare/Projects/PositronixLab/reports/baselines_analysis.md) | 14P / 0X / 0F |
| **`dataprovider`** | Refactored; 16 providers, `CompositionalDataProvider`, lazy exports | 3 Med (`F601` dup keys, `drop_last`, split seeds) | [dataprovider_analysis.md](file:///home/cesare/Projects/PositronixLab/reports/dataprovider_analysis.md) | 44P / 0X / 0F |
| **`tasks`** | Unified `SubnetworkMask`/`MaskApplier`/`AtlasDiscoverer`/`ModularityBenchmark` + legacy `compositionality/` | 3 Med (new API) + 9 legacy xfails | [tasks_analysis.md](file:///home/cesare/Projects/PositronixLab/reports/tasks_analysis.md) | 35P / 9X / 0F |
| **`spacemath`** | v2 primitives + 3D sequence reducers + `POSITRONIXLAB_COMPILE`; legacy `SpaceMath` still present | 3 High/Med (v2) + legacy bugs | [spacemath_analysis.md](file:///home/cesare/Projects/PositronixLab/reports/spacemath_analysis.md) | 43P / 9X / **1F** *(stale test)* |
| **`modularspace`** | `LayerSpace`/`SpaceStep`/`presets` active; `layerspace_old.py` removed | 6 High/Med | [modularspace_analysis.md](file:///home/cesare/Projects/PositronixLab/reports/modularspace_analysis.md) | 40P / 4X / 0F |
| **`modularobserver`** | Generalized (`observable_types`/`target_layers`); A5 & `clear()` fixed | 4 Med | [modularobserver_analysis.md](file:///home/cesare/Projects/PositronixLab/reports/modularobserver_analysis.md) | 26P / 6X / 0F |
| **`modularcluster`** | B1–B5/B7 & A6 fixed in `ModularClusterKmeans`; drift in `ModularClusterKmeansStream` | 3 Med (stream drift) | [modularcluster_analysis.md](file:///home/cesare/Projects/PositronixLab/reports/modularcluster_analysis.md) | 28P / 4X / 0F |
| **`deeplearning`** | B1–B3 fixed (`EarlyStopping`, `inference()`); `TrainConfig` not yet wired | 6 Med (B4–B11) | [deeplearning_analysis.md](file:///home/cesare/Projects/PositronixLab/reports/deeplearning_analysis.md) | 27P / 6X / 0F |
| **`metamodel`** | Lazy `__init__.py` + SOTA re-exports; `MetaModel`/`MetaCriterion` still unrefactored | 6 High/Crit (B1, B3–B15) | [metamodel_analysis.md](file:///home/cesare/Projects/PositronixLab/reports/metamodel_analysis.md) | 28P / 11X / 0F |
| **`pipeline`** | `BaseConfig` + lazy `__init__.py`; `get_errors()` empty-list regression | 4 High/Med + **1F** regression | [pipeline_analysis.md](file:///home/cesare/Projects/PositronixLab/reports/pipeline_analysis.md) | 28P / 3X / **1F** *(bug)* |
| **`nas`** | Ax 1.3 `failed_reason=` fixed; lazy `__init__.py` | 5 High/Med (retry loop, `warm_start`, OOM) | [nas_analysis.md](file:///home/cesare/Projects/PositronixLab/reports/nas_analysis.md) | 15P / 5X / **2F** *(sandbox)* |
| **`template`** | Expanded `__init__.py` (eagerly pulls `ax`); V1-oriented `TemplateAtlas` | 3 High, 3 Med | [template_analysis.md](file:///home/cesare/Projects/PositronixLab/reports/template_analysis.md) | 21P / 7X / 0F |
| **`stats`** | Unconditional `cuml` import fixed; `__init__.py` populated | 4 High/Med (UMAP `None` crash, reducers) | [stats_analysis.md](file:///home/cesare/Projects/PositronixLab/reports/stats_analysis.md) | 23P / 14X / 0F |
| **`modularui`** | Lazy `__init__.py` populated; UI bugs unrefactored | 8 High/Med | [modularui_analysis.md](file:///home/cesare/Projects/PositronixLab/reports/modularui_analysis.md) | 20P / 8X / 0F |
| **`utils`** | A7, A8, A9a–c, A9e, B1, B6, B8, B12 fixed; `__init__.py` populated | 3 Med (`process_start_spawn`, `seed_everything`, `suppress_output` error path) | [utils_analysis.md](file:///home/cesare/Projects/PositronixLab/reports/utils_analysis.md) | 34P / 6X / 0F |

**Totals:** **427 passed, 92 xfailed, 4 failed** (of which 2 are sandbox-only `parallelazy` stub failures, 1 is a stale 3D reducer test, and 1 is a real regression in `pipeline.py:98`). See [xfail_analysis.md](file:///home/cesare/Projects/PositronixLab/reports/xfail_analysis.md) for the complete triage of all 91 `xfail` markers.

---

## 3. Priority Fix Roadmap (Updated)

### P0 — Quick Wins & Immediate Test-Suite Greenlight (≤ 1–2 hours)
1. **Fix `Pipeline.get_errors()` regression (`pipeline/pipeline.py:98`):** Change `if isinstance(raw_errs, (list, tuple)):` to `if isinstance(raw_errs, (list, tuple)) and raw_errs:` so live workers with `errors=[]` still fall back to `plugin_errors(plugin_id)`.
2. **Clean up 4 XPASSing strict xfails & 1 stale test:**
   - Remove `@pytest.mark.xfail(strict=True)` on `tests/modularcluster/test_streaming_kmeans.py:239` and `tests/modularobserver/test_modularobserver.py:334, 467, 484` (and update `:447` / `test_streaming_kmeans.py:233` tuple unpacking for B3).
   - Update `tests/spacemath/test_spacemath_extended.py:219` to assert `(4, 5)` for the new 3D `(B, T, U) -> (B, U)` `reduce_linear`.
3. **Remove 6 duplicate dict keys (`F601`) in `dataprovider/__init__.py:144-149`.**
4. **Commit working-tree B1 (`bias_mode`) & B2 (`attribution_mode`/`energy_mode`) fixes in `tasks/`** — after addressing the `complement_sampled` + `bias_mode="mask"` edge case (`masks.py:465-474`) and wiring `bias_mode` into `ModularityBenchmark.evaluate_mask()` (`benchmarks.py:213`).
5. **Fix `UmapProcessor._create_mapper()` when `UMAP is None` (`stats/processor.py:90`):** Raise a clear `ImportError("pip install 'positronixlab[umap]'")` instead of `TypeError: 'NoneType' object is not callable`.

### P1 — Correctness, Decoupling & Robustness
6. **Port `ModularClusterKmeans` A6a/A6b fixes to `ModularClusterKmeansStream`** (`modularcluster_kmeans_stream.py:77-81, 99-110, 265-268`) and fix `StreamingClusterTracker._relabel_to_match_previous` return annotation (`-> Tuple[torch.Tensor, torch.Tensor]`).
7. **Finish decoupling `PipelineConfig` from `parallelazy`:** Move `PipelineOptions(ManagerOptions)` out of `pipeline_config.py` (or lazy-import `ManagerOptions`) and guard `PipelineConfig` imports under `if TYPE_CHECKING:` in `deeplearning`, `metamodel`, `modularobserver`, and `modularspace`.
8. **Fix `nas/experiment.py` remaining operational risks:** Add `max_consecutive_failures` circuit breaker to `Experiment.run()`, fix `warm_start_from()` `SESSION_FACTORY` cleanup, and fix solo OOM retry off-by-one.
9. **Fix `metamodel` builder & `MetaMetric` bugs:** Unshadow `MetaMetric.custom_metric`, fix `build_model_output_block`, CNN `Flatten`+`BatchNorm`, and `MetaModel.__init__` `device` handling.
10. **Fix `baselines/` edge cases:** Warn when `CVRResNetModel(pretrained=True, use_instance_norm=True)` ignores `pretrained`; fix `NeuroSurgeonWrappedModel.forward` multi-arg bypass; guard `next(model.parameters(), None)` in `neurosurgeon/` & `tasks/benchmarks.py:29`; add friendly `ImportError` hints for `[baselines]`, `[nas]`, and `[ui]` extras (v0.1.3 E5).
11. **Fix `modularspace`, `template`, `modularui`, and `utils` remaining bugs:** `LayerSpace.process_scores()` grad guard, `ClusterTracker` `k_old == 0` / unbounded `history`, `TemplateBase.add_choice(log_scale=...)`, `ModularUI` CSS paths & `visible_layers=[]`, `utils.suppress_output` `KeyError` on `os.dup` failure, and `model_hash` `usedforsecurity=False`.

### P2 — Packaging, Ergonomics & Hygiene (v0.1.3 D/E/F)
12. **Packaging & PyPI polish:** Convert relative links in `README.md` to absolute GitHub URLs (E1); lazy-import heavy single-file deps (`torchsummary`, `pandas`, `requests`, E2); make `template/__init__.py` lazy so `TemplateVariant` doesn't require `ax-platform`; decide whether to enforce `py.typed` with `mypy`/`pyright` in CI (E6).
13. **Performance & SOTA benchmark completeness:** Avoid double-pass in `AtlasDiscoverer.discover*` (D2), add `sample_size` option to CPU silhouette (D3), disable `torch.compile` on CPU by default (D5), and complete remaining SOTA roadmap items (`GradNormTracker`, 6-split Lepori matrix runner, `UntrainedModelBaseline`, `template_sota_*.py`).
14. **Code hygiene:** Replace `print()` in library code with `logging`, fix `zip(..., strict=True)`, `RUF013` implicit `Optional`, and archive/retire legacy `tasks/compositionality/` once superseded.

---

## 4. `AGENTS.md` Sync Status

`AGENTS.md` was last synced at `a069575` (10 commits behind `HEAD` `5740a0c`). Items to update in `AGENTS.md`:
- Update `Last synced` hash after review.
- Note `CUPY_CACHE_DIR=/tmp/cupy_cache` alongside `TRITON_CACHE_DIR` and `TORCHINDUCTOR_CACHE_DIR` for full pytest in read-only-home sandboxes (needed when `cuml`/`cupy` is installed in `.venv`).
- Note `POSITRONIXLAB_COMPILE=0` opt-out toggle in `spacemath/common.py`.
- Note `baselines_analysis.md` and `positronixlab_review_v0.1.3.md` under `reports/`.
- Note `CVRSimCLRDataProvider`, `PermutedMnistDataProvider`, and `SCAN`/`Math` full implementations under `dataprovider/`.
