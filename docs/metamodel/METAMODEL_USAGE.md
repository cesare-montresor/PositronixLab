# MetaModel Usage Guide

Practical guide for configuring and running `MetaModelConfig`, `MetaModel`, `MetaCriterion`, `MetaMetric`, and `MetaTransform`.

## 1. Configuring with `MetaModelConfig`

`MetaModelConfig` inherits from `BaseConfig`, supporting immutable updates, deterministic signatures, and YAML/JSON serialization:

```python
from positronixlab.metamodel import MetaModelConfig

cfg = MetaModelConfig(
    name="mnist_fc",
    task="classification",
    criterion_type="cross_entropy",
    model_type="fc",
    input_data_shape=784,
    input_hidden_dim=64,
    input_num_layers=3,
    output_data_shape=10,
    output_num_layers=2,
    input_dropout_every=-1,   # -1 disables dropout (0 enables on layer 0)
    output_dropout_every=-1,
    activation="relu",
)

# Immutable variant + deterministic hash
wider_cfg = cfg.with_updates(input_hidden_dim=128)
assert cfg.signature() != wider_cfg.signature()

# Round-trip serialization
cfg.save("/tmp/mnist_fc.yaml")
loaded = MetaModelConfig.load("/tmp/mnist_fc.yaml")
```

### Composing with Other Domain Configs (NAS / Pipelines)

```python
from dataclasses import dataclass
from positronixlab.metamodel import MetaModelConfig
from positronixlab.deeplearning.config import TrainConfig

@dataclass
class ExperimentConfig(MetaModelConfig, TrainConfig):
    pass
```

## 2. Building and Running a `MetaModel`

```python
import torch
from positronixlab.metamodel import MetaModel, MetaModelConfig, MetaMetric

cfg = MetaModelConfig(
    task="classification",
    criterion_type="cross_entropy",
    input_data_shape=16,
    input_hidden_dim=32,
    input_num_layers=2,
    output_data_shape=4,
    output_num_layers=2,
    input_dropout_every=-1,
    output_dropout_every=-1,
)

model = MetaModel("classifier", cfg)
model.to_cpu()  # Always call to_cpu() explicitly when running on CPU

x = torch.randn(8, 16)
y = torch.randint(0, 4, (8,))

y_hat = model(x)                       # shape: (8, 4)
loss = model.loss(x, y, y_hat)         # scalar cross-entropy loss
acc = MetaMetric(cfg).metric(x, y, y_hat)
```

> **Tip:** Always use `output_num_layers >= 2` for classification and regression. With `output_num_layers == 1`, the current builder outputs `input_hidden_dim` instead of `output_data_shape`.

## 3. Autoencoder Mode

When `task="autoencoder"`, `nn_output` automatically mirrors `nn_input` in reverse order, and `MetaCriterion.loss(x, y, y_hat)` uses `x` as the target:

```python
ae_cfg = MetaModelConfig(
    task="autoencoder",
    criterion_type="mse",
    model_type="fc",
    input_data_shape=32,
    input_hidden_dim=16,
    input_num_layers=2,
    input_reduce_every=0,
    input_dropout_every=-1,
    output_dropout_every=-1,
)

ae = MetaModel("fc_ae", ae_cfg)
ae.to_cpu()

x = torch.randn(8, 32)
recon = ae(x)                          # shape: (8, 32)
loss = ae.loss(x, None, recon)         # MSE(recon, x)
```

## 4. Custom Losses with `MetaCriterion`

```python
import torch
from positronixlab.metamodel import MetaCriterion, MetaModelConfig

cfg = MetaModelConfig(task="regression", criterion_type="weighted_mse")
criterion = MetaCriterion(cfg)
criterion.custom_loss("weighted_mse", lambda x, y, y_hat: ((y_hat - y) ** 2 * 2.0).mean())

loss = criterion.loss(x=None, y=torch.zeros(4, 2), y_hat=torch.ones(4, 2))
```

You can also register a custom loss directly on `model.criterion` after `MetaModel` construction (or use a built-in `criterion_type` first and switch `model.criterion.criterion_type`).

## 5. Sample Transforms with `MetaTransform`

`MetaTransform` builds a `torchvision.transforms.Compose` pipeline from string names and is callable directly (`t(sample)`) or via `t.apply(sample)`:

```python
import torch
from positronixlab.metamodel import MetaTransform

# Note: 'normalize' requires >=3D tensor (C, H, W), so place it BEFORE 'flatten'
t = MetaTransform(["normalize", "flatten", "to_float"])
sample = torch.rand(1, 28, 28)
vec = t.apply(sample)                  # shape: (784,), dtype: float32
```

Available transform names: `to_tensor`, `normalize`, `normalize_image`, `flatten`, `to_float`, `to_long`, `augment_image`.

## 6. Saving and Loading Checkpoints

`BaseModel` (and therefore `MetaModel`) provides `model_path()`, `save()`, and `load()`:

```python
model.basepath = "/tmp/checkpoints"
saved_path = model.save()              # /tmp/checkpoints/classifier.pth

reloaded = MetaModel("classifier", cfg)
reloaded.to_cpu()
reloaded.load(saved_path)
```

## 7. Runnable Example

Run the bundled CPU example from the repository root:

```bash
.venv/bin/python examples/metamodel/example_metamodel.py
```
