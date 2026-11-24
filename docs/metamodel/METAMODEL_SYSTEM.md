# PositronixLab MetaModel System

`positronixlab/metamodel` provides the configurable neural network builder (`BaseModel`, `MetaModel`, `ResNet`), decoupled hyperparameter configuration (`MetaModelConfig`), loss/metric wrappers (`MetaCriterion`, `MetaMetric`), and sample transform pipeline (`MetaTransform`) used across training, observability, and NAS experiments.

## Architecture

`MetaModel` follows a three-stage sequential topology designed for modular inspection (`ModularObserver` / `ModelMapper`) and symmetric autoencoder construction:

```
              MetaModelConfig (BaseConfig)
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
  MetaModel(BaseModel)  MetaCriterion    MetaMetric
   ├─ nn_input   (nn.Sequential of blocks)
   ├─ nn_latent  (nn.Identity placeholder)
   └─ nn_output  (nn.Sequential of blocks)
```

Each block in `nn_input` and `nn_output` is assembled as an `OrderedDict` wrapped in `nn.Sequential` (and optionally wrapped in `ResNet(block)` when `input_dim == output_dim` and residuals are enabled):

```
  input ──► layer (Linear / Conv2d / ConvTranspose2d)
              ──► activation (ReLU / Sigmoid / Tanh / LeakyReLU)
              ──► [batch_norm (BatchNorm1d / BatchNorm2d / GroupNorm)]
              ──► [dropout (Dropout)]
              ──► (+ input if ResNet)
```

For `task="autoencoder"`, `nn_output` mirrors `nn_input` in reverse order (swapping `(in_dim, out_dim)` of each encoder block and mapping `cnn -> decnn`).

## Module Layout

| File | Public Exports | Responsibility |
|---|---|---|
| `positronixlab/metamodel/__init__.py` | `BaseModel`, `MetaCriterion`, `MetaMetric`, `MetaModel`, `MetaModelConfig`, `MetaTransform`, `ResNet` | Package re-exports. |
| `positronixlab/metamodel/config.py` | `MetaModelConfig` | Composable `BaseConfig` dataclass holding architecture, task, criterion, and block hyperparameters. |
| `positronixlab/metamodel/metamodel.py` | `BaseModel`, `MetaModel`, `ResNet` | Base checkpoint/device module, 3-stage configurable model builder, and residual wrapper. |
| `positronixlab/metamodel/metacriterion.py` | `MetaCriterion`, `MetaMetric` | Task-aware loss dispatch and evaluation metric helpers with custom registration hooks. |
| `positronixlab/metamodel/metatransform.py` | `MetaTransform` | String-configured `torchvision.transforms.Compose` wrapper with `.apply(x)`. |

---

# API Reference

## `config.py`

### `MetaModelConfig(BaseConfig)`

Composable dataclass inheriting `BaseConfig` (`signature()`, `to_dict()`, `to_yaml()`, `with_updates()`, `save()`, `load()`, `from_dict()`). Designed to be combined with other domain mixins (e.g. `LayerSpaceConfig`, `TrainConfig`, `DataConfig`, `TrialConfig`).

| Group | Field | Type | Default | Description |
|---|---|---|---|---|
| Identity | `name` (inherited) | `str` | `"default_config"` | Config identifier. |
| Task & Loss | `task` | `str` | `"classification"` | One of `"classification"`, `"regression"`, `"autoencoder"`. |
| | `criterion_type` | `str` | `"cross_entropy"` | Loss key consumed by `MetaCriterion`. |
| | `model_type` | `str` | `"fc"` | Backbone layer type (`"fc"`, `"cnn"`, `"decnn"`). |
| Input Stage | `input_data_shape` | `int` | `784` | Input feature count (FC) or input channels (CNN). |
| | `input_hidden_dim` | `int` | `32` | Output width of the first input block. |
| | `input_num_layers` | `int` | `2` | Number of blocks in `nn_input`. |
| | `input_reduce_every` | `int` | `0` | Halve width every `k` layers when `k > 1` and `layer_num % k == 0`. |
| | `input_dropout_every` | `int` | `0` | Dropout cadence (`0` applies dropout on layer 0; `> 0` every `k` layers; `< 0` disables). |
| | `input_batch_norm` | `bool` | `True` | Include normalization in input blocks. |
| | `input_residual` | `bool` | `True` | Wrap non-first input blocks in `ResNet` when `in_dim == out_dim`. |
| Output Stage | `output_data_shape` | `int` | `10` | Final output dimension (classes / regression targets). |
| | `output_hidden_dim` | `int` | `32` | Intended hidden width for output blocks (see Known Limitations). |
| | `output_num_layers` | `int` | `2` | Number of blocks in `nn_output` (ignored for `"autoencoder"`, which mirrors `input_num_layers`). |
| | `output_reduce_every` | `int` | `-1` | Halve width every `k` layers when `k > 1` and `(layer_num + 1) % k == 0`. |
| | `output_dropout_every` | `int` | `0` | Dropout cadence for output blocks. |
| | `output_batch_norm` | `bool` | `True` | Include normalization in output blocks. |
| | `output_residual` | `bool` | `True` | Wrap non-last output blocks in `ResNet` when `in_dim == out_dim`. |
| Shared Block | `activation` | `str` | `"relu"` | One of `"relu"`, `"sigmoid"`, `"tanh"`, `"leaky_relu"`. |
| | `dropout` | `float` | `0.5` | Dropout probability. |
| | `cnn_kernel_size` | `int` | `3` | Kernel size for `Conv2d` / `ConvTranspose2d`. |

---

## `metamodel.py`

### `BaseModel(nn.Module)`

Minimal base class fulfilling the `DeeplearningBase` / `DeeplearningTrain` contract (`model.device`) and providing checkpoint helpers.

| Member | Signature | Description |
|---|---|---|
| `__init__` | `(name: str = None, device = None)` | Sets `self.name`, `self.timestamp`, `self.basepath = None`, and `self.device` (defaults to CUDA if available, else CPU). |
| `model_path` | `(path = None) -> str` | Resolves absolute path to `<basepath>/<name>.pth` or `./checkpoint/<name>.pth` when `path` is `None`. |
| `save` | `(path = None) -> str` | Saves `state_dict()` to resolved path (creates parent dirs). |
| `load` | `(path) -> Optional[str]` | Loads `state_dict()` with `weights_only=True` if file exists; returns `None` otherwise. |

### `MetaModel(BaseModel)`

Configurable 3-stage model (`nn_input -> nn_latent -> nn_output`) with an embedded `MetaCriterion`.

| Member | Description |
|---|---|
| `TASK_LIST` | `['classification', 'regression', 'autoencoder']` |
| `LAYER_TYPES` | `{'fc': nn.Linear, 'cnn': nn.Conv2d, 'decnn': nn.ConvTranspose2d, 'rnn': nn.RNN, 'transformer': nn.TransformerEncoder}` (only `'fc'`, `'cnn'`, `'decnn'` are implemented in `block_layer`). |
| `__init__(name, config, device=None)` | Builds `criterion`, `nn_input`, `nn_latent`, `nn_output`, then calls `to_gpu()`. |
| `forward(x)` | Passes `x` through `nn_input -> nn_latent -> nn_output`. |
| `loss(x, y, y_hat)` | Delegates to `self.criterion.loss(x, y, y_hat)`. |
| `to_gpu(num=None)` / `to_cpu()` | Moves parameters and updates `self.device`. |
| `build_model()` | Validates `task`/`model_type`, creates `MetaCriterion`, and builds all three stages. |
| `last_layer()` / `last_layer_size()` | Inspects the most recently added layer in `nn_output` (falling back to `nn_input`). |
| `layer_size(layer)` | Returns `(in_dim, out_dim)` for `nn.Linear` and `nn.Conv{1,2,3}d`. |
| `get_model_size() -> float` | Total parameter + buffer memory in MB. |
| `nmum_fop() -> int` | Rough MAC/parameter count across `Linear` and `Conv*d` layers (note typo in method name). |
| `print()` | Calls `torchsummary.summary(...)` and prints model size. |

### `ResNet(nn.Module)`

Residual wrapper: `ResNet(module)` wraps an `nn.Module` (or `OrderedDict`, converted to `nn.Sequential`) and computes `self.module(inputs) + inputs`.

---

## `metacriterion.py`

### `MetaCriterion(config)`

- Reads `config.task` and `config.criterion_type`.
- Built-in `criterion_type` values: `cross_entropy`, `mse`, `mae` (`l1_loss`), `huber`, `bce_with_logits`, `bce`.
- `custom_loss(name, function)` registers `function(x, y, y_hat) -> Tensor`.
- When `task == "autoencoder"`, `loss(x, y, y_hat)` substitutes `y = x`.

### `MetaMetric(config, metric_type=None)`

- Defaults `metric_type` by `config.task`: `"classification" -> "accuracy"`, `"segmentation" -> "ioU"`, anything else `-> "f1_score"`.
- `metric(x, y, y_hat) -> Tensor` computes the detached metric tensor.

---

## `metatransform.py`

### `MetaTransform(transform_list)`

Builds a `torchvision.transforms.Compose` pipeline from a list of string keys and exposes `apply(x)`.

| Key | Underlying Transform |
|---|---|
| `to_tensor` | `transforms.ToTensor()` |
| `normalize` | `transforms.Normalize(mean=[0.5], std=[0.5])` |
| `normalize_image` | `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` |
| `flatten` | `transforms.Lambda(lambda x: torch.flatten(x, start_dim=0))` (picklable top-level `_flatten`) |
| `to_float` | `transforms.Lambda(_to_float)` |
| `to_long` | `transforms.Lambda(_to_long)` |
| `augment_image` | `ToPILImage -> RandomHorizontalFlip -> RandomVerticalFlip -> RandomRotation(360) -> ColorJitter(0.2, 0.2, 0.2, 0.1) -> ToTensor -> ImageNet Normalize` |

---

## Known Limitations & Caveats

See [`analysis.md`](../analysis.md) for the complete defect catalog. Key current caveats:

- **`MetaModel.__init__` ignores `device`**: always invokes `to_gpu()` on construction; call `model.to_cpu()` explicitly for CPU execution.
- **`output_num_layers=1`**: sets output dimension to `input_hidden_dim` instead of `output_data_shape` (use `output_num_layers >= 2`).
- **`output_hidden_dim` is unused**: the first output block uses `input_hidden_dim`.
- **Final output block applies activation & batch norm**: be aware that `nn_output`'s last layer currently includes `activation` and `batch_norm`.
- **`dropout_every=0`**: enables dropout on layer 0 rather than disabling it; pass `-1` to disable dropout.
- **CNN classification/regression & multi-layer CNN autoencoders**: currently broken (missing flatten + `BatchNorm2d` after `Linear` for classification/regression; `layer_size` lacks `ConvTranspose2d` for multi-layer CNN autoencoders).
- **`MetaMetric.custom_metric`**: instance dict shadows the `custom_metric(name, fn)` method.
- **`MetaTransform`**: uses `.apply(x)` and does not implement `__call__`.
