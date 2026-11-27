# SOTA Integration Architecture — Completed Milestones Archive

> **Archived:** 2026-11-27 (`v0.1.3` / `5740a0c` + working tree).
> **Remaining SOTA roadmap items:** See [SOTA_integration.md](file:///home/cesare/Projects/PositronixLab/reports/SOTA_integration.md).

---

## Completed Deliverables (Phases 1–4)

### Phase 1 — Unified Mask Contract & NeuroSurgeon Bridge (100% Complete)
- [x] `positronixlab/tasks/masks.py`: `SubnetworkMask` (dotted keys, set algebra, `unit`↔`weight` projection, IoU/IoMin/overlap/sparsity, save/load).
- [x] `positronixlab/tasks/masks.py:314-491`: `MaskApplier` context manager with all 5 ablation modes + `bias_mode` + exception-safe `try/finally` weight/bias restoration.
- [x] `positronixlab/baselines/neurosurgeon/{adapter,discoverer}.py`: `NeuroSurgeonDiscoverer` + `HFForwardShim` wrapping `NeuroSurgeon` (bypassing undeclared `torchmetrics` in `Probing`).
- [x] `positronixlab/tasks/atlas_discoverer.py`: `AtlasDiscoverer` on `SubnetworkMask` with clean hook teardown and differential selectivity attribution.
- [x] Head-to-head CPU smoke benchmark: `examples/tasks/example_atlas_vs_sota.py` and `tests/tasks/test_masks_and_discoverers.py`.

### Phase 2 — Csordás Benchmarks & Core Enablers (Completed Subset)
- [x] Compositional data providers: `TupleArithmeticDataProvider`, `AddMulDataProvider`, `ClassRemovalDataProvider`, `PermutedMnistDataProvider`, `SCANDataProvider`, `MathDataProvider` + `baselines/csordas/datasets.py`.
- [x] Generalized `ModularObserver` (`observable_types`, `target_layers`, `exclude_patterns`) and 3D sequence reducers (`reduce_seq_{mean,max,cls,energy}`) in `spacemath/reducers.py`.
- [x] Core `CsordasGumbelDiscoverer` (`baselines/csordas/discoverer.py`) with PyTorch 2.x `parametrize`, Gumbel-Sigmoid STE, and linear logit loss *(advanced options `GradNormTracker`, `n_mask_samples`, and RNG `seed` remain open)*.

### Phase 3 — Lepori Vision (CVR) & Language (SynEval) Benchmarks (Completed Subset)
- [x] `CVROddOneOutDataProvider` (tasks `103–120`), `CVRSimCLRDataProvider`, and `SynevalOddOneOutDataProvider` (tasks `0–19`) with vendored `_cvr_gen/` and `_syneval_gen.py`.
- [x] `OddOneOutHead`, `CVRResNetModel` (`InstanceNorm` ResNet18/50/WideResNet50), `CVRSimCLRModel` (+ `SimCLRLoss`/`pretrain_simclr`), `CVRViTModel`, and `SynEvalBertModel` (`baselines/lepori/models.py`).
- [x] Unified registries: `ModelLoader` (12 architectures) and `DataProviderLoader` (14 datasets) in `baselines/loader.py`.

### Phase 4 — Optional Packaging (Completed Subset)
- [x] `[project.optional-dependencies] baselines = ["NeuroSurgeon", "transformers", "opencv-python-headless", "pillow"]` in `pyproject.toml:65-70`.
