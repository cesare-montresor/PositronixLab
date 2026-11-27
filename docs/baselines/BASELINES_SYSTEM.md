# PositronixLab Baselines System

`positronixlab.baselines` packages external SOTA modularity benchmarks, model architectures, compositional datasets, and weight/neuron-masking discoverers organized by paper, under a single unified contract shared with `positronixlab.tasks`.

It is intentionally decoupled from `positronixlab.pipeline` and `parallelazy` so SOTA models, datasets, and discoverers can be imported and benchmarked standalone.

---

## Architecture

```
                 ┌──────────────────────────────────────────┐
                 │    ModelLoader  /  DataProviderLoader    │
                 │  • Unified registry (csordas:*, lepori:*)│
                 │  • save() / load_bundle() / load()       │
                 └───────────┬──────────────────┬───────────┘
                             │                  │
        ┌────────────────────┼──────────────────┼────────────────────┐
        ▼                    ▼                  ▼                    ▼
┌───────────────┐   ┌─────────────────┐  ┌──────────────┐   ┌──────────────────┐
│ csordas/      │   │ lepori/         │  │ neurosurgeon/│   │ positronixlab.   │
│ (ICLR 2021)   │   │ (NeurIPS 2023)  │  │ (2023/2024)  │   │ tasks            │
│ • 5 models    │   │ • CVR ResNet/ViT│  │ • adapter.py │   │ • SubnetworkMask │
│ • 6 datasets  │   │ • SynEval BERT  │  │ • Discoverer │   │ • MaskApplier    │
│ • Gumbel-STE  │   │ • 8 datasets    │  │ • torchmet.  │   │ • AtlasDiscoverer│
│   Discoverer  │   │ • _cvr_gen/     │  │   bypass     │   │ • Benchmark      │
└───────┬───────┘   └────────┬────────┘  └──────┬───────┘   └────────┬─────────┘
        └────────────────────┴──────────────────┴────────────────────┘
                                      │
                                      ▼
                     SOTAModel + CompositionalDataProvider
                         + SubnetworkDiscoverer -> SubnetworkMask
```

---

## Module Layout

| Path | Contents |
|---|---|
| `positronixlab/baselines/__init__.py` | Public API exports + `csordas`, `lepori`, `neurosurgeon` subpackages |
| `positronixlab/baselines/base.py` | `MASKABLE_LAYER_TYPES`, `get_maskable_layer_names`, `SOTAModelSpec`, `SOTADatasetSpec`, `SOTAModel`; re-exports `CompositionalDataProvider`, `SubtaskSplit` |
| `positronixlab/baselines/loader.py` | `ModelLoader` (12 models, bundle save/load, checkpoint auto-detection), `DataProviderLoader` (14 datasets) |
| `positronixlab/baselines/csordas/` | `models.py` (5 architectures + `strip_csordas_mask_keys`), `datasets.py` (paper builders, `PAPER_CONFIGS`, single-head Csordás wrappers), `discoverer.py` (`CsordasGumbelDiscoverer`) |
| `positronixlab/baselines/lepori/` | `models.py` (`OddOneOutHead`, `CVRDecisionMLP`, `CVRResNetModel`, `CVRViTModel`, `SynEvalBertModel`, `CVRSimCLRModel`, `SimCLRLoss`, `simclr_ranking_metrics`, `pretrain_simclr`, `strip_lepori_l0_keys`), `datasets.py` (CVR, CVR SimCLR & SynEval builders + `PAPER_CONFIGS`), `_cvr_gen/` (procedural CVR generator), `_syneval_gen.py` (SynEval `RawData.zip` split generator) |
| `positronixlab/baselines/neurosurgeon/` | `adapter.py` (`is_neurosurgeon_available`, `HFForwardShim`, `NeuroSurgeonWrappedModel`, `wrap_with_neurosurgeon`, `strip_neurosurgeon_mask_keys`), `discoverer.py` (`NeuroSurgeonDiscoverer`) |

---

## Core Contract (`base.py`)

- **`MASKABLE_LAYER_TYPES`**: `(nn.Linear, nn.Conv2d, nn.Conv1d)` (dynamically appends HuggingFace `Conv1D` when `transformers` is installed).
- **`get_maskable_layer_names(model, include_types=None, exclude_patterns=()) -> List[str]`**: Canonical dotted layer names eligible for masking.
- **`SOTAModelSpec`** / **`SOTADatasetSpec`**: Frozen dataclasses storing registry metadata (`arch`/`name`, `source`, `task_type`, `builder`, `description`, `default_exclude_patterns`).
- **`SOTAModel(nn.Module)`**:
  - Class attributes: `source`, `arch`, `task_type`, `default_exclude_patterns`.
  - `__init__(name=None, device=None, criterion=None)` (defaults to CPU and `nn.CrossEntropyLoss()`).
  - Properties & methods: dynamic `device` (getter/setter), `to(...)`, `loss(x, y, y_hat)`, `maskable_layers(exclude_patterns=None)`, `to_neurosurgeon(...)`, `model_path(path=None)`, `save(path=None, **kwargs)`, `load(path=None, strict=True, map_location=None, **kwargs)`.
- **`CompositionalDataProvider`** & **`SubtaskSplit`**: Re-exported from `positronixlab.dataprovider.dataprovider`.

---

## Registries & Serialization (`loader.py`)

### `ModelLoader`
| Method | Description |
|---|---|
| `register(arch, source, task_type, builder, description="", default_exclude_patterns=())` | Registers a model builder under `arch`. |
| `list_models(source=None) -> List[str]` | Lists registered architectures, optionally filtered by `source` (`"csordas"` or `"lepori"`). |
| `get_spec(arch) -> SOTAModelSpec` | Returns the spec for `arch`. |
| `create(arch, device=None, **kwargs) -> SOTAModel` | Instantiates a model by architecture key. |
| `save(model, path, *, arch=None, model_kwargs=None, dataset_name=None, metrics=None, metadata=None) -> Path` | Saves a self-describing `positronix_sota_bundle_v1` checkpoint bundle (CPU `state_dict` + constructor kwargs + metadata). |
| `load_bundle(path, *, device=None, strict=True, **override_model_kwargs) -> Tuple[SOTAModel, Dict[str, Any]]` | Reconstructs a `SOTAModel` from a `save()` bundle and returns `(model, info)`. |
| `load(arch, checkpoint=None, backbone_checkpoint=None, mlp_checkpoint=None, device=None, strict=False, return_mask=False, **model_kwargs)` | Creates a model and loads weights, auto-detecting checkpoint format (`"neurosurgeon"`, `"lepori"`, `"csordas"`, `"plain"`), stripping mask keys, and optionally returning `(model, SubnetworkMask | None)` when `return_mask=True`. Supports split `backbone_checkpoint` + `mlp_checkpoint` for `CVRResNetModel`. |
| `wrap_neurosurgeon(model, ...)` | Convenience wrapper delegating to `wrap_with_neurosurgeon(...)`. |

**Registered Models (12):**
- **Csordás (`source="csordas"`)**: `"csordas:mlp"` (`CsordasFeedforwardModel`), `"csordas:conv"` (`CsordasConvModel`), `"csordas:lstm"` (`CsordasLSTMModel`), `"csordas:resnet110"` (`CsordasResNet110`), `"csordas:transformer"` (`CsordasTransformerEncDec`, excludes `("embedding", "pos_encoding")`).
- **Lepori (`source="lepori"`)**: `"lepori:cvr_resnet18"`, `"lepori:cvr_resnet50"`, `"lepori:cvr_wideresnet50"` (`CVRResNetModel`), `"lepori:cvr_simclr_resnet18"`, `"lepori:cvr_simclr_resnet50"` (`CVRSimCLRModel`), `"lepori:cvr_vit"` (`CVRViTModel`, excludes `("embeddings",)`), `"lepori:syneval_bert"` (`SynEvalBertModel`, excludes `("embeddings",)`).

### `DataProviderLoader`
- `register(name, source, task_type, builder, description="")`
- `list_datasets(source=None) -> List[str]`
- `create(name, **kwargs) -> CompositionalDataProvider`

**Registered Datasets (14):**
- **Csordás**: `"csordas:tuple_arithmetic"`, `"csordas:add_mul"`, `"csordas:class_removal"`, `"csordas:deepmind_math"`, `"csordas:scan"`, `"csordas:permuted_mnist"`.
- **Lepori**: `"lepori:cvr_inside_contact"`, `"lepori:cvr_inside_count"`, `"lepori:cvr_contact_count"`, `"lepori:cvr_simclr"`, `"lepori:syneval_sv_singular"`, `"lepori:syneval_sv_plural"`, `"lepori:syneval_pro_ant_singular"`, `"lepori:syneval_pro_ant_plural"`.

---

## Paper Implementations

### Csordás et al. (`csordas/`)
- **Models**: `CsordasFeedforwardModel`, `CsordasConvModel`, `CsordasLSTMModel`, `CsordasResNet110`, `CsordasTransformerEncDec`.
- **Datasets**: Thin wrappers over `positronixlab.dataprovider` (`TupleArithmeticDataProvider`, `AddMulDataProvider`, `ClassRemovalDataProvider`) configured with Csordás single-head defaults (`n_digits=1`, `target_mode="integer"`).
- **`CsordasGumbelDiscoverer(mask_loss_weight=1e-4, lr=1e-2, epochs=10, init_logit=2.0, tau=1.0, loss_fn=None)`**: Uses PyTorch 2.x `torch.nn.utils.parametrize` on a `copy.deepcopy(model)` (zero mutation of caller's model), exact Gumbel-Sigmoid STE, and linear logit sparsity penalty `(mask_loss_weight / B) * sum(logits)`. Emits `SubnetworkMask(granularity="weight")`.

### Lepori et al. (`lepori/`)
- **Models**: `OddOneOutHead` (parameter-free cosine-similarity 4-way OOO head: `(B, N, D) -> (B, N)`), `CVRDecisionMLP`, `CVRResNetModel` (5D `(B, N, 3, H, W)` input), `CVRSimCLRModel` (self-supervised contrastive pretraining ResNet with `SimCLRLoss`, `simclr_ranking_metrics`, `to_ooo_model()`, and `pretrain_simclr()`), `CVRViTModel` (`transformers.ViTModel`), `SynEvalBertModel` (`transformers.BertModel`).
- **Datasets**: Re-exports `CVROddOneOutDataProvider` (as `CVROddOneOutSOTAProvider`), `CVRSimCLRDataProvider`, and `SynevalOddOneOutDataProvider` (as `SynEvalOddOneOutSOTAProvider`) from `positronixlab.dataprovider`. Procedural CVR task generation lives in `_cvr_gen/`; SynEval `RawData.zip` split generation lives in `_syneval_gen.py`.
- **Standalone train/eval loops**: `pretrain_simclr()`, `CsordasGumbelDiscoverer.discover()`, and `NeuroSurgeonDiscoverer.discover()` run outside `DeeplearningTrain` and are cataloged in [`docs/deeplearning/DEEPLEARNING_SYSTEM.md`](file:///home/cesare/Projects/PositronixLab/docs/deeplearning/DEEPLEARNING_SYSTEM.md#train--eval-loops-outside-deeplearningtrain).

### NeuroSurgeon (`neurosurgeon/`)
- **`adapter.py`**:
  - `is_neurosurgeon_available() -> bool`
  - `HFForwardShim(model)`: Bridges standard `nn.Module`s to `CircuitModel`'s HuggingFace interface (strips `return_dict=True`, wraps tensor output in `SimpleNamespace(logits=out, loss=None)`).
  - `NeuroSurgeonWrappedModel`: Wraps `CircuitModel` with positional `model(x)` calls, complete non-mask parameter freezing, deepcopy isolation, `temperature` property, `mask_parameters()`, `l0_loss()`, `l0_statistics()`, `set_ablate_mode()`, `use_masks()`, `extract_mask_tensors(binarize=True)`, and `extract_subnetwork_mask(name="neurosurgeon")`.
  - `wrap_with_neurosurgeon(model, target_layers=None, mask_method="continuous_sparsification", mask_unit="weight", mask_hparams=None, freeze_base=True, l0_lambda=1e-8, deepcopy_model=True)` — supports `"continuous_sparsification"`, `"hard_concrete"`, and `"magnitude_pruning"` with `mask_unit` in `{"weight", "neuron"}`.
- **`NeuroSurgeonDiscoverer(mask_method="continuous_sparsification", mask_unit="weight", l0_lambda=1e-4, lr=1e-2, epochs=10, max_temp=200.0, mask_hparams=None, loss_fn=None)`**: Optimizes `mask_parameters()` with Adam on `task_loss + l0_lambda * l0_loss` (with exponential temperature annealing for continuous sparsification; short-circuits for `"magnitude_pruning"` or `epochs <= 0`) and returns `SubnetworkMask` (`"unit"` for `mask_unit="neuron"`, else `"weight"`).

> [!NOTE]
> **NeuroSurgeon `torchmetrics` import quirk**: Upstream `libs/NeuroSurgeon` has an undeclared `torchmetrics` import in `NeuroSurgeon.Probing`. `_import_neurosurgeon()` in `adapter.py` automatically catches `ModuleNotFoundError` for `torchmetrics`, synthesizes the top-level `NeuroSurgeon` package entry, and imports only `NeuroSurgeon.Masking` and `NeuroSurgeon.Models` so all NeuroSurgeon features work without installing `torchmetrics`.

---

## Dependencies & References

- Optional dependency extra in `pyproject.toml`: `baselines = ["NeuroSurgeon", "transformers", "opencv-python-headless", "pillow"]`.
- Design report: [`reports/SOTA_integration.md`](file:///home/cesare/Projects/PositronixLab/reports/SOTA_integration.md).
- Tests: [`tests/baselines/`](file:///home/cesare/Projects/PositronixLab/tests/baselines/).
