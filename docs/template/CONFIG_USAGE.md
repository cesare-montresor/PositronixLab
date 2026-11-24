# Template Module Usage Guide

## Quick Start

### Loading a Built-in Template

```python
from positronixlab.template import TemplateFactory

# List registered experiments
print(list(TemplateFactory.EXPERIMENTS.keys()))
# ['covertype', 'higgs', 'mnist_fc', 'mnist_cnn', 'fmnist_fc', 'fmnist_cnn']

# Instantiate by name
template = TemplateFactory.FromName("mnist_fc", name="mnist_fc_exp")

print("Tunable parameters:", list(template.params.keys()))
print("Fixed constants:", template.get_fixed())
```

### Building a Custom Search Space with `TemplateBase`

All builder methods return `self`, so calls can be chained fluently:

```python
from positronixlab.template import TemplateBase
from positronixlab.template.template_base import ParameterType

template = (
    TemplateBase("custom_mlp")
    .add_fixed("dataset_name", "mnist")
    .add_fixed("input_data_shape", 784)
    .add_fixed("output_data_shape", 10)
    .add_range("learning_rate", [1e-4, 1e-2], ParameterType.FLOAT, scaling="log")
    .add_range("input_hidden_dim", [64, 512], ParameterType.INT, step_size=64)
    .add_range("output_hidden_dim", [32, 256], ParameterType.INT, step_size=32)
    .add_choice("activation", ["relu", "leaky_relu"], ParameterType.STRING)
    .add_choice("input_batch_norm", [True, False], ParameterType.BOOL)
    .add_derived("total_hidden_dim", "input_hidden_dim + output_hidden_dim", ParameterType.INT)
    .add_constraint("output_hidden_dim <= input_hidden_dim")
)
```

---

## Converting to Ax 1.3 `ParameterConfig` Objects

```python
from ax.api.client import Client

client = Client(random_seed=42)
client.configure_experiment(
    name=template.name,
    parameters=template.to_parameter_configs(),
    parameter_constraints=template.get_constraints() or None,
)
client.configure_optimization(objective="accuracy")

trials = client.get_next_trials(max_trials=1)
trial_index, sampled_params = next(iter(trials.items()))

# Merge fixed constants with Ax-sampled parameters (sampled params take precedence)
merged = {**template.get_fixed(), **sampled_params}
print(f"Trial {trial_index}:", merged)
```

---

## Modifying, Merging, and Removing Parameters

```python
from positronixlab.template import TemplateBase, TemplateFactory

base = TemplateFactory.FromName("mnist_fc")

# Converting a tunable parameter into a fixed constant removes it from base.params
base.add_fixed("dropout", 0.25)
assert "dropout" not in base.params
assert base.fixed["dropout"] == 0.25

# Converting a fixed constant into a tunable parameter removes it from base.fixed
base.add_choice("criterion_type", ["cross_entropy", "label_smoothing"], "str")
assert "criterion_type" not in base.fixed
assert "criterion_type" in base.params

# Remove from both params and fixed
base.remove("modular_space_loss_alignment")

# Merge an overlay template
overlay = (
    TemplateBase("overlay")
    .add_range("learning_rate", [1e-4, 5e-3], "float", scaling="log")
    .add_constraint("output_hidden_dim <= input_hidden_dim")
)
base.merge(overlay, overwrite=True)
```

---

## Registering a Custom Template in `TemplateFactory`

```python
from positronixlab.template import TemplateFactory
from positronixlab.template.template_base import TemplateFC


class TemplateIris(TemplateFC):
    def __init__(self, name=None):
        super().__init__(name)
        self.add_fixed("dataset_name", "iris")
        self.add_fixed("input_data_shape", 4)
        self.add_fixed("output_data_shape", 3)
        self.add_range("batch_size", [8, 32], "int")


TemplateFactory.AddTemplate("iris_fc", TemplateIris)
iris_tpl = TemplateFactory.FromName("iris_fc")
```

---

## Working with Variants (`TemplateVariant`, `TemplateVariantList`, `TemplateVariantFactory`)

```python
from positronixlab.template import TemplateFactory
from positronixlab.template.template_factory import TemplateVariantFactory
from positronixlab.template.template_variant import TemplateVariant, TemplateVariantList

template = TemplateFactory.FromName("mnist_fc")

# Standard 4-variant modularity matrix (a, b, c, d)
variants = TemplateVariantFactory.create_standard_modularity_variants(base_name="mod")
variants.validate_all(template)

for variant in variants:
    print(variant.name, variant.get_fixed_features(), "-", variant.description)

# Define a custom variant
custom = TemplateVariant(
    name="relu_bn",
    fixed_features={"activation": "relu", "input_batch_norm": True},
    description="ReLU with input batch normalization",
)
custom.validate(template)
```

---

## Serialization (YAML & Dict)

```python
from positronixlab.template import TemplateBase, TemplateFactory

template = TemplateFactory.FromName("covertype", name="covertype_run")

# Save to YAML
saved_path = template.save("/tmp/covertype_run.yaml")

# Reload from YAML
reloaded = TemplateBase.load(saved_path)
assert reloaded.to_dict() == template.to_dict()

# Dict round-trip (useful for JSON logging or cloning)
clone = TemplateBase.from_dict(template.to_dict())
```

> **Note:** Keep `fixed` values to YAML-safe primitives (`int`, `float`, `str`, `bool`, `None`, `list`, `dict`). Avoid Python tuples in `fixed`, as `save()` uses `yaml.dump` while `load()` uses `yaml.safe_load`.

---

## Integration with `TrialConfig` and `Experiment`

When running NAS with `positronixlab.nas`:

```python
from positronixlab.nas import Experiment, TrialConfig
from positronixlab.template import TemplateFactory

template = TemplateFactory.FromName("mnist_fc")

# Option 1: Materialize a single TrialConfig manually
cfg = TrialConfig.from_trial(
    params={"learning_rate": 0.001, "batch_size": 64},
    fixed=template.get_fixed(),
    trial_index=0,
)

# Option 2: Pass template directly to nas.Experiment
# exp = Experiment(name="mnist_hpo", template=template, objective="accuracy", eval_fn=...)
```

---

## Runnable Example

Run the included self-contained CPU example:

```bash
.venv/bin/python positronixlab/template/examples/example_template.py
```
