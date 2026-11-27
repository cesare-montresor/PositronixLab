# `positronixlab/metamodel` — Open Issues Analysis

> **Status (2026-11-27):** **33 passed, 6 xfailed, 0 failed** (`tests/metamodel/`).
> **Solved issues archive:** Resolved P0 bugs B1, B3, B8, plus B5 (`ConvTranspose2d` autoencoder), partial B2 (lazy `__init__.py`), and `MetaTransform` cleanups moved to [solved/metamodel_solved.md](file:///home/cesare/Projects/PositronixLab/reports/solved/metamodel_solved.md).
> **Note:** Remaining bugs **B2 (partial), B4, B6–B7, B9–B15** remain open, plus v0.1.3 **E2** (`torchsummary` top-level import at `metamodel.py:11`).

---

## 1. Overview & Refactor Status

`positronixlab/metamodel` provides the configurable model builder, loss/metric abstractions, and sample transform wrapper for `positronixlab`. It is currently **mid-refactor**:

| File | Role | Status |
|---|---|---|
| [`__init__.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/metamodel/__init__.py) | Re-exports `BaseModel`, `MetaCriterion`, `MetaMetric`, `MetaModel`, `MetaModelConfig`, `MetaTransform`, `ResNet`. | Active |
| [`config.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/metamodel/config.py) | `MetaModelConfig(BaseConfig)`: clean, composable hyperparameter mixin for model topology, task, criterion, and block parameters. | Active (new foundation) |
| [`metamodel.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/metamodel/metamodel.py) | `BaseModel` (checkpoint/device base), `MetaModel` (`nn_input -> nn_latent -> nn_output` builder), and `ResNet` residual wrapper. | Legacy / Needs Refactor |
| [`metacriterion.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/metamodel/metacriterion.py) | `MetaCriterion` (task-aware loss dispatch + custom loss registry) and `MetaMetric` (accuracy/F1/IoU wrapper). | Legacy / Needs Refactor |
| [`metatransform.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/metamodel/metatransform.py) | `MetaTransform`: string-keyed `torchvision.transforms.Compose` wrapper with `.apply(x)`. | Legacy / Stable (minimal) |

### Config Refactor Gap
`MetaModelConfig` has been cleanly extracted onto `positronixlab.pipeline.config.BaseConfig` (zero `parallelazy` dependency) and is already composed into `ComposedTrialConfig` in `positronixlab/nas/tests/test_nas.py`. However:
- [`metamodel.py:5`](file:///home/cesare/Projects/PositronixLab/positronixlab/metamodel/metamodel.py#L5) and [`metacriterion.py:6`](file:///home/cesare/Projects/PositronixLab/positronixlab/metamodel/metacriterion.py#L6) still do `from positronixlab.pipeline import PipelineConfig` and type-hint `config: PipelineConfig`.
- Because `PipelineConfig` lives in `pipeline/pipeline_config.py` alongside `PipelineOptions(ManagerOptions)`, importing `positronixlab.metamodel` transitively imports `parallelazy`. In restricted/standalone environments without `parallelazy`, even importing `MetaModelConfig` via `positronixlab.metamodel` fails with `ModuleNotFoundError: No module named 'parallelazy'`.
- `MetaModel` only works reliably for **FC classification, FC regression (with caveats on final activation/BN), and FC autoencoders with `output_num_layers >= 2`**. CNN classification/regression, multi-layer CNN autoencoders, RNN, and Transformer modes are broken or unimplemented.

---

## 2. Bugs & Defects

### B1. `MetaMetric.custom_metric` method is shadowed by an instance dictionary (Critical)
- **Location:** [`metacriterion.py:45`](file:///home/cesare/Projects/PositronixLab/positronixlab/metamodel/metacriterion.py#L45), [`metacriterion.py:57-58`](file:///home/cesare/Projects/PositronixLab/positronixlab/metamodel/metacriterion.py#L57-L58)
- **Root Cause:** `__init__` assigns `self.custom_metric = {}`, using the exact same identifier as `def custom_metric(self, name, function):`.
- **Impact:** Calling `metric.custom_metric("name", fn)` always raises `TypeError: 'dict' object is not callable`. Custom metrics cannot be registered through the public API.

### B2. Stale `PipelineConfig` imports transitively couple `metamodel` to `parallelazy` (High)
- **Location:** [`metamodel.py:5`](file:///home/cesare/Projects/PositronixLab/positronixlab/metamodel/metamodel.py#L5), [`metacriterion.py:6`](file:///home/cesare/Projects/PositronixLab/positronixlab/metamodel/metacriterion.py#L6)
- **Root Cause:** Both files import `PipelineConfig` from `positronixlab.pipeline` solely for type annotations instead of importing `MetaModelConfig` from `positronixlab.metamodel.config`.
- **Impact:** Importing anything from `positronixlab.metamodel` triggers `pipeline_config.py -> parallelazy.plugin_data.ManagerOptions`, breaking standalone/sandboxed usage when `parallelazy` is absent or inaccessible.

### B3. `build_model_output_block()` outputs wrong dimension when `output_num_layers == 1` and ignores `output_hidden_dim` (High)
- **Location:** [`metamodel.py:249-259`](file:///home/cesare/Projects/PositronixLab/positronixlab/metamodel/metamodel.py#L249-L259)
- **Root Cause:**
  1. When `output_num_layers == 1`, both `is_first_layer` and `is_last_layer` are `True`. Because `if is_first_layer:` is checked before `elif is_last_layer:`, `output_dim` becomes `self.config.input_hidden_dim` instead of `self.config.output_data_shape`.
  2. On `is_first_layer` (line 252), `output_dim` uses `self.config.input_hidden_dim` instead of `self.config.output_hidden_dim`. Because `last_size` is non-`None` for all subsequent layers, `config.output_hidden_dim` is effectively dead code.
  3. If `input_num_layers == 0`, `last_size` is `None` and `last_size[1]` (line 251) raises `TypeError`.
- **Impact:** Any single-layer output head produces the wrong output shape; tuning `output_hidden_dim` (e.g., in NAS) has zero effect.

### B4. CNN `classification` / `regression` models are broken at runtime (High)
- **Location:** [`metamodel.py:264-276`](file:///home/cesare/Projects/PositronixLab/positronixlab/metamodel/metamodel.py#L264-L276)
- **Root Cause:**
  1. Line 266 sets local `model_type = 'fc'` to create `nn.Linear`, but line 276 calls `self.block_batch_norm(self.config.model_type, output_dim)` using `self.config.model_type` (`'cnn'`), attaching `nn.BatchNorm2d` directly after `nn.Linear`.
  2. No `nn.Flatten()` (or adaptive pooling) exists between 4D CNN `nn_input` outputs `(N, C, H, W)` and 2D FC `nn_output` inputs, and `input_dim` is set to `out_channels` rather than flattened features.
- **Impact:** `forward()` always crashes for `model_type="cnn"` with `task` in `{"classification", "regression"}`.

### B6. Final output block applies activation, BatchNorm, and dropout to predictions/logits (High)
- **Location:** [`metamodel.py:274-283`](file:///home/cesare/Projects/PositronixLab/positronixlab/metamodel/metamodel.py#L274-L283), [`metamodel.py:325-333`](file:///home/cesare/Projects/PositronixLab/positronixlab/metamodel/metamodel.py#L325-L333)
- **Root Cause:** `is_last_layer` is not checked before adding `activation`, `batch_norm`, and `dropout` to the final block of `nn_output`.
- **Impact:** Classification logits with `relu` are clamped non-negative and batch-normalized across samples; regression models with `relu` cannot predict negative values; `output_data_shape=1` with `BatchNorm1d(1)` collapses predictions to zero mean / unit variance.

### B7. `MetaModel.__init__` ignores the `device` argument and forces GPU placement (Medium)
- **Location:** [`metamodel.py:61-72`](file:///home/cesare/Projects/PositronixLab/positronixlab/metamodel/metamodel.py#L61-L72)
- **Root Cause:** `super().__init__()` is called without `name` or `device`, and line 72 unconditionally calls `self.to_gpu()`.
- **Impact:** Passing `device="cpu"` or `device=torch.device("cpu")` is silently ignored on CUDA hosts (moving parameters to CUDA) and prints a noisy `"WARNING: No GPU available, CPU fallback"` on CPU-only hosts.

### B8. `dropout_every == 0` enables dropout on layer 0 instead of disabling it (Medium)
- **Location:** [`metamodel.py:222`](file:///home/cesare/Projects/PositronixLab/positronixlab/metamodel/metamodel.py#L222), [`metamodel.py:279`](file:///home/cesare/Projects/PositronixLab/positronixlab/metamodel/metamodel.py#L279), [`metamodel.py:330`](file:///home/cesare/Projects/PositronixLab/positronixlab/metamodel/metamodel.py#L330)
- **Root Cause:** `first_dropout = dropout_every == 0 and is_first_layer` inserts `nn.Dropout(dropout)` at layer 0 when `dropout_every == 0` (the default in `MetaModelConfig`).
- **Impact:** Default configs silently apply 50% dropout to the first input and output layers; users must discover the undocumented `-1` sentinel to disable dropout.

### B9. Hardcoded `padding=1` in `Conv2d`/`ConvTranspose2d` breaks `cnn_kernel_size != 3` (Medium)
- **Location:** [`metamodel.py:163-166`](file:///home/cesare/Projects/PositronixLab/positronixlab/metamodel/metamodel.py#L163-L166)
- **Root Cause:** `padding=1` is hardcoded instead of `padding=kernel_size // 2` (or `"same"`).
- **Impact:** For `cnn_kernel_size=5` (or `1`), spatial dimensions shrink/grow on every layer, breaking `ResNet` skip-connection addition (`self.module(inputs) + inputs` raises shape mismatch `RuntimeError`).

### B10. `nmum_fop()` typo, `Conv1d` `IndexError`, and inaccurate FLOP formula (Medium)
- **Location:** [`metamodel.py:410-425`](file:///home/cesare/Projects/PositronixLab/positronixlab/metamodel/metamodel.py#L410-L425)
- **Root Cause:** Method is misspelled `nmum_fop`; line 420 indexes `layer.kernel_size[1]` even for `nn.Conv1d` (1-tuple -> `IndexError`), ignores `kernel_size[2]` for `Conv3d`, skips `ConvTranspose2d`, and omits spatial output dimensions `H_out * W_out`.
- **Impact:** Crashes on `Conv1d` and computes weight parameter count rather than true FLOPs for convolutions.

### B11. `layer_size()` fails on `ResNet` blocks and `ConvTranspose*d` (Medium)
- **Location:** [`metamodel.py:390-403`](file:///home/cesare/Projects/PositronixLab/positronixlab/metamodel/metamodel.py#L390-L403)
- **Root Cause:** Unwrapping `ResNet` produces `nn.Sequential` (not the inner `.layer`), which then falls through to `ValueError`. `nn.ConvTranspose{1,2,3}d` is not handled.
- **Impact:** Direct calls to `model.layer_size(block)` fail on residual blocks and transposed convolutions.

### B12. `LAYER_TYPES` advertises unimplemented `'rnn'` and `'transformer'`, and `'decnn'` lacks BatchNorm support (Medium)
- **Location:** [`metamodel.py:58`](file:///home/cesare/Projects/PositronixLab/positronixlab/metamodel/metamodel.py#L58), [`metamodel.py:158-188`](file:///home/cesare/Projects/PositronixLab/positronixlab/metamodel/metamodel.py#L158-L188)
- **Root Cause:** `'rnn'` and `'transformer'` pass `build_model()` validation because they are in `LAYER_TYPES`, then fail in `block_layer()`. `'decnn'` is implemented in `block_layer()` but raises `ValueError` in `block_batch_norm()`.
- **Impact:** Misleading validation errors and inability to use `model_type="decnn"` with `batch_norm=True`.

### B13. `BaseModel.load()` missing default `path=None`, missing `map_location`, and silent missing-file return (Low)
- **Location:** [`metamodel.py:34-46`](file:///home/cesare/Projects/PositronixLab/positronixlab/metamodel/metamodel.py#L34-L46)
- **Root Cause:** `load(self, path)` omits `path=None` (inconsistent with `save(self, path=None)`), calls `torch.load(path, weights_only=True)` without `map_location=self.device`, and returns `None` silently if the file does not exist. `save()` removes the old file before writing (non-atomic).
- **Impact:** Loading CUDA checkpoints on CPU-only machines fails; typos in checkpoint paths silently leave weights uninitialized.

### B14. `MetaModel.print()` breaks on indexed CUDA devices and CNN inputs (Low)
- **Location:** [`metamodel.py:12`](file:///home/cesare/Projects/PositronixLab/positronixlab/metamodel/metamodel.py#L12), [`metamodel.py:91-93`](file:///home/cesare/Projects/PositronixLab/positronixlab/metamodel/metamodel.py#L91-L93)
- **Root Cause:** Hard top-level import of `torchsummary`; passes `device=str(self.device)` (`'cuda:0'` fails `torchsummary`'s `assert device in ['cuda', 'cpu']`) and 1D `input_size=(input_data_shape,)` (breaks CNNs).

### B15. `MetaMetric` mathematical and return-type issues (Low)
- **Location:** [`metacriterion.py:54`](file:///home/cesare/Projects/PositronixLab/positronixlab/metamodel/metacriterion.py#L54), [`metacriterion.py:60-78`](file:///home/cesare/Projects/PositronixLab/positronixlab/metamodel/metacriterion.py#L60-L78)
- **Root Cause:** Regression defaults to `f1_score` (meaningless for continuous targets); `f1_score` and `ioU` operate on raw logits without thresholding/argmax; `return result.detach()` crashes with `AttributeError` if a custom metric returns a Python `float`.

---

## 3. Inconsistencies & Design Issues

1. **Inconsistent sentinel values in `MetaModelConfig`:** `input_reduce_every` defaults to `0` while `output_reduce_every` defaults to `-1` (`config.py:20,29`), yet both check `reduce_every > 1` (preventing halving every layer with `reduce_every == 1`, and risking `dim // 2 == 0` on narrow layers).
2. **Off-by-one reduction cadence:** `build_model_input_block` checks `layer_num % reduce_every == 0` (`metamodel.py:198`), whereas `build_model_output_block` checks `(layer_num + 1) % reduce_every == 0` (`metamodel.py:242`).
3. **`MetaTransform` is not callable:** Implements `.apply(x)` instead of `__call__(x)`, forcing downstream workarounds such as `oddoneout_dataprovider.py:64`. It also hardcodes ImageNet normalization and `ToPILImage()` in `augment_image` (incompatible with grayscale/1D data) and offers no parameterization.
4. **Unused imports & dead code:** `from abc import ABC` (`metacriterion.py:2`), `from torchvision import transforms` (`metamodel.py:7`), unused `last_size` (`metamodel.py:302`), and unused `is_last_layer` (`metamodel.py:200`).
5. **Hard coupling of `MetaModel` to `MetaCriterion` and `torchsummary`:** `MetaModel` constructs `MetaCriterion` internally and imports `torchsummary` eagerly at module import time.
6. **Inconsistent `BaseModel.device` typing:** Default is `torch.device`, but passing `device="cpu"` stores a `str`.

---

## 4. Opportunities & Recommendations (Reusable Functional Modularity Framework)

To evolve `positronixlab/metamodel` into a clean, research-grade backbone for functional modularity experiments (`ModularObserver`, `ModularSpace`, NAS, weight-masking/ablation):

1. **Finish decoupling from `PipelineConfig` / `parallelazy`:**
   - Type-hint `MetaModel` and `MetaCriterion` against `MetaModelConfig` (or a structural `Protocol`), removing `from positronixlab.pipeline import PipelineConfig` from `metamodel.py` and `metacriterion.py`.
   - Lazy-import `torchsummary` inside `print()` (or replace with `torchinfo`).
2. **Fix block builder semantics & expose clean stage boundaries:**
   - Keep the 3-stage `nn_input -> nn_latent -> nn_output` contract (valuable for `ModularObserver` and latent-space interventions), but make `nn_latent` configurable (e.g., bottleneck projection, attention, or custom module injection).
   - Omit activation/BN/dropout on the final output head layer (with an explicit `output_activation: Optional[str] = None` option for bounded outputs / autoencoders).
   - Fix `output_num_layers == 1`, wire `output_hidden_dim`, insert `nn.AdaptiveAvgPool2d(1) + nn.Flatten()` (or shape-inferred flatten) between CNN encoders and FC heads, and compute `padding = kernel_size // 2`.
   - Normalize `dropout_every <= 0` to mean disabled and `reduce_every == 1` to reduce every layer (clamped with `max(1, input_dim // 2)`).
3. **Add first-class modularity & intervention hooks on `BaseModel`:**
   - Standardize a `get_observable_modules()` or named-layer registry so `ModularObserver`, `ModelMapper`, and `tasks/compositionality` (`weightmask_check.py`) can target `BaseModel` subclasses uniformly without fragile `is_linear()` heuristics.
   - Support non-destructive unit/group masking hooks (e.g., forward hooks or `MaskedLinear`) directly on `BaseModel` for causal ablation of discovered Computational Groups (CGs).
4. **Harden `MetaCriterion`, `MetaMetric`, and `MetaTransform`:**
   - Rename `MetaMetric`'s dict to `self.custom_metrics`, fix default regression metric (e.g., `mse`/`mae`/`r2`), and guard `.detach()` for scalar returns.
   - Add `__call__ = apply` to `MetaTransform` and support parameterized transform specs (dicts or callables alongside string shortcuts).
   - Respect `device` in `MetaModel.__init__` (normalize via `torch.device(device)` and avoid side-effect `to_gpu()` calls) and add `map_location=self.device` + atomic temp-file rename to `BaseModel.save()`/`load()`.

---

## 5. Docs / Tests / Examples Summary

- **Docs created:**
  - [`docs/METAMODEL_SYSTEM.md`](file:///home/cesare/Projects/PositronixLab/positronixlab/metamodel/docs/METAMODEL_SYSTEM.md) — architecture overview, module layout, full API reference, and known caveats.
  - [`docs/METAMODEL_USAGE.md`](file:///home/cesare/Projects/PositronixLab/positronixlab/metamodel/docs/METAMODEL_USAGE.md) — quickstart and recipes for `MetaModelConfig`, `MetaModel` (classification & autoencoder), `MetaCriterion`, `MetaTransform`, and checkpointing.
- **Tests created:**
  - [`tests/test_metamodel.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/metamodel/tests/test_metamodel.py) — 39 test cases covering `MetaModelConfig`, `BaseModel`, `ResNet`, `MetaCriterion`, `MetaMetric`, `MetaModel`, and `MetaTransform`, including 12 `@pytest.mark.xfail(strict=True)` regression tests documenting confirmed bugs.
- **Example created:**
  - [`examples/example_metamodel.py`](file:///home/cesare/Projects/PositronixLab/positronixlab/metamodel/examples/example_metamodel.py) — self-contained CPU example exercising config updates, FC classification training, FC autoencoder reconstruction, custom `MetaCriterion`, `MetaTransform`, and checkpoint save/reload.
