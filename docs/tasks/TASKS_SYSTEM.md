# PositronixLab Tasks System

Downstream evaluation suite for discovering, manipulating, applying, and benchmarking modular subnetworks in trained PyTorch models.

`positronixlab.tasks` provides two layers:
1. **Unified Subnetwork & Benchmark Contract** (top-level `positronixlab/tasks/`): `SubnetworkMask`, `MaskApplier`, `SubnetworkDiscoverer`, `RandomMaskDiscoverer`, PositronixLab's `AtlasDiscoverer`, and `ModularityBenchmark` — enabling apples-to-apples comparisons between Atlas and external SOTA discoverers in `positronixlab.baselines` (Csordás Gumbel-Sigmoid, NeuroSurgeon).
2. **Earlier Compositionality Prototype** (`positronixlab/tasks/compositionality/`): confusion-matrix-based `AtlasCheck`, `WeightmaskCheck`, and `CompositionalityPipeline`.

---

## Architecture (Unified Contract)

```
                   ┌──────────────────────────────────┐
                   │   Trained Model + DataLoaders    │
                   └────────────────┬─────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                         ▼
┌───────────────────┐    ┌─────────────────────┐   ┌───────────────────────┐
│ AtlasDiscoverer   │    │ RandomMaskDiscoverer│   │ baselines Discoverers │
│ (LayerSpace +     │    │ (random baseline)   │   │ • CsordasGumbel       │
│  FAISS K-Means +  │    │                     │   │ • NeuroSurgeon        │
│  unit energy)     │    │                     │   │                       │
└─────────┬─────────┘    └──────────┬──────────┘   └───────────┬───────────┘
          └─────────────────────────┼──────────────────────────┘
                                    ▼
                        ┌───────────────────────┐
                        │    SubnetworkMask     │
                        │  • unit / weight gran.│
                        │  • &, |, -, invert()  │
                        │  • sparsity, iou,     │
                        │    iomin, save/load   │
                        └───────────┬───────────┘
                                    │
                                    ▼
                        ┌───────────────────────┐
                        │      MaskApplier      │
                        │  5 ablation modes +   │
                        │  guaranteed restore   │
                        └───────────┬───────────┘
                                    │
                                    ▼
                        ┌───────────────────────┐
                        │  ModularityBenchmark  │
                        │  • SubnetworkEvalRes. │
                        │  • BenchmarkReport    │
                        │  • JSON/CSV + Ax HPO  │
                        └───────────────────────┘
```

---

## Module Layout

| File | Contents |
|---|---|
| `positronixlab/tasks/__init__.py` | Exports all 12 top-level unified contract symbols |
| `positronixlab/tasks/masks.py` | `Granularity`, `AblationMode`, `SubnetworkMask`, `MaskApplier` |
| `positronixlab/tasks/discoverer.py` | `unpack_batch`, `SubnetworkDiscoverer` (ABC), `RandomMaskDiscoverer` |
| `positronixlab/tasks/atlas_discoverer.py` | `AtlasDiscoverer` (decoupled `fit_atlas`, `discover`, `discover_subtasks`, `attribute_from_clusters`, `attribute_from_unit_energy`, `sweep_fractions`) |
| `positronixlab/tasks/benchmarks.py` | `default_eval_accuracy`, `SubnetworkEvalResult`, `BenchmarkReport`, `ModularityBenchmark` |
| `positronixlab/tasks/compositionality/` | Earlier prototype: `atlas_check.py`, `weightmask_check.py`, `compositionality.py` |
| `examples/tasks/example_atlas_vs_sota.py` | Head-to-head Atlas vs Csordás vs NeuroSurgeon CPU benchmark |
| `examples/tasks/example_sota_benchmark.py` | `ModularityBenchmark` multi-method comparison example |
| `examples/tasks/example_compositionality.py` | Synthetic CPU example for `compositionality/` |

---

## Unified API Reference

### `masks.py`
- **`Granularity`**: `Literal["unit", "weight"]`.
- **`AblationMode`**: `Literal["retain", "zero_ablate", "random_ablate", "randomly_sampled", "complement_sampled"]`.
- **`SubnetworkMask`** (frozen dataclass: `name`, `granularity`, `masks: Dict[str, Tensor]`, `scores=None`, `metadata={}`):
  - Keyed by canonical dotted layer names (e.g. `"fc1"`, `"encoder.layers.0.linear1"`).
  - **Projection**: `to_weight_mask(model)`, `to_unit_mask(min_active_fraction=0.0)`.
  - **Set algebra**: `&`, `|`, `-`, `invert()`.
  - **Metrics**: `l0()`, `max_l0()`, `sparsity()`, `density()`, `iou(other)`, `per_layer_iou(other)`, `iomin(other)`, `per_layer_iomin(other)`, `overlap_fraction(other)`.
  - **Serialization**: `save(path, **kwargs)`, `load(path, **kwargs)`.
- **`MaskApplier(model, mask, mode="retain", exclude_layers=(), seed=None)`**: Leak-free context manager applying any of the 5 `AblationMode`s and guaranteeing restoration of original weights on exit (including on exceptions).

### `discoverer.py`
- **`unpack_batch(batch) -> Tuple[Tensor, Tensor]`**: Unpacks both `(x, y)` and `(x, y, idx)` batches.
- **`SubnetworkDiscoverer(ABC)`**: Abstract interface (`name = "base"`) with `discover(model, dataloader, target_layers=None, mask_name="subnetwork", **kwargs) -> SubnetworkMask`.
- **`RandomMaskDiscoverer(density=0.5, granularity="weight", seed=0)`**: Deterministic random baseline (`name = "random"`).

### `atlas_discoverer.py`
- **`AtlasDiscoverer(SubnetworkDiscoverer)`** (`name = "atlas"`):
  - `__init__(layer_space_config=None, atlas_epochs=3, space_dim=8, space_lr=5e-2, active_q=0.75, contrast_method="softmax_ce", cluster_metric="cosine", k_min=None, k_max=None, top_cluster_fraction=0.5, min_active_clusters=1)`.
  - `fit_atlas(model, dataloader, target_layers=None) -> (positions_by_layer, clusters_by_layer)`: Fits `LayerSpace` + `ModularClusterKmeans` using leak-free hooks (supports 2D Linear, 3D Sequence, and 4D Conv2d activations); caches `last_positions` and `last_clusters`.
  - `discover(model, dataloader, target_layers=None, mask_name="atlas", **kwargs) -> SubnetworkMask`.
  - `discover_subtasks(model, reference_dataloader, subtask_dataloaders, target_layers=None) -> Dict[str, SubnetworkMask]`.
  - Decoupled attribution & sparsity sweeps: `attribute_from_clusters(...)`, `attribute_from_unit_energy(...)`, `sweep_fractions(clusters_by_layer, unit_energy, fractions, ...) -> List[SubnetworkMask]`.

### `benchmarks.py`
- **`default_eval_accuracy(model, dataloader) -> float`**: Default classification accuracy evaluator.
- **`SubnetworkEvalResult`**: Dataclass recording accuracy across all 5 ablation modes, `off_target_zero_ablate`, active/total/masked unit & weight counts/sparsities, `selectivity` property, and `to_flat_dict()`.
- **`BenchmarkReport`**: Dataclass holding `dataset_name`, `model_arch`, `baselines`, `results`, `alignments`, and `metadata`, with `to_dict()`, `summary_metrics()` (for Ax HPO), `save_json(path)`, and `save_csv(path)`.
- **`ModularityBenchmark(eval_fn=default_eval_accuracy, modes=("retain", "zero_ablate"), seed=42)`**:
  - `evaluate_mask(model, mask, target_loader, *, method, subtask, baseline_accuracy=None, off_target_loaders=None) -> SubnetworkEvalResult`.
  - `compare_methods(model, masks_by_method, subtask_loaders, *, dataset_name="unknown", model_arch="unknown", reference_method="atlas", metadata=None) -> BenchmarkReport`: Supports single masks or sparsity-sweep sequences and computes cross-method unit/weight IoU and overlap against `reference_method`.

---

## Earlier Prototype (`compositionality/`)

Retained for confusion-matrix analysis and legacy experiments:
- **`atlas_check.py`**: `Distribution` ABC, `DiscreteDistribution`, `TensorDistribution`, `AtlasCheck` (`run()`, `extract_weightmasks(threshold=0.0)`, `modularity_score(kld_target='y')`).
- **`weightmask_check.py`**: `WeightmaskCheck`, `MultiClassWeightmaskManager`, `WeightMaskCriterion`, `WeightMasksMetrics`.
- **`compositionality.py`**: `StaticMaskApplier`, `CompositionalityCheck`, `CompositionalityPipeline`.

**Prototype caveats** (see [`reports/tasks_analysis.md`](file:///home/cesare/Projects/PositronixLab/reports/tasks_analysis.md)):
1. `AtlasCheck.extract_weightmasks()` returns dotted keys and 1D bool masks, whereas `StaticMaskApplier` expects underscore-sanitized keys and weight-broadcastable shapes.
2. `AtlasCheck` leaves `ModularObserver` forward hooks attached after `run()` — call `atlas.observer.on_forward_layer(None)`.
3. `WeightmaskCheck` runs 1 pass (`num_epochs` ignored), produces `NaN` on empty class slices, and leaves `requires_grad=False`. Prefer `CsordasGumbelDiscoverer` in `positronixlab.baselines` for new work.
