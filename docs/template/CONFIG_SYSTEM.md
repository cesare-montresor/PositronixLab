# Template & Search-Space System (`positronixlab/template`)

## Overview

`positronixlab/template` provides lightweight, **Ax 1.3+** search-space builders, reusable dataset templates, a factory registry, and fixed-feature ablation variants.

The core design separates:
- **Tunable & derived parameters (`template.params`)** — converted into Ax 1.3 `*ParameterConfig` objects (`RangeParameterConfig`, `ChoiceParameterConfig`, `DerivedParameterConfig`).
- **Linear parameter constraints (`template.constraints`)** — passed to Ax `Client.configure_experiment(parameter_constraints=...)`.
- **Fixed non-searchable constants (`template.fixed`)** — excluded from the Ax search space to avoid noise, then merged with Ax-sampled trial parameters when materializing a `TrialConfig` / `PipelineConfig`.

---

## Architecture

### 1. `TemplateBase` (`template_base.py`)

Foundational fluent builder for Ax 1.3 search spaces:

- `add_range(name, bounds, parameter_type=ParameterType.FLOAT, scaling=None, step_size=None, log_scale=False)` — adds a `RangeParameterConfig` spec (`scaling` resolves to `"log"` if `log_scale=True` else `"linear"`).
- `add_choice(name, values, parameter_type=ParameterType.STRING, is_ordered=None, dependent_parameters=None, log_scale=None)` — adds a `ChoiceParameterConfig` spec. Automatically defaults `is_ordered = len(values) <= 2 or parameter_type in ("int", "float")` to avoid Ax parameter warnings. *(Note: `log_scale` is not supported by Ax 1.3 `ChoiceParameterConfig`; avoid setting it.)*
- `add_derived(name, expression_str, parameter_type=ParameterType.FLOAT)` — adds a `DerivedParameterConfig` spec (linear SymPy expression).
- `add_constraint(constraint)` — appends a deduplicated linear inequality string (e.g. `"output_hidden_dim <= input_hidden_dim"`).
- `add_fixed(name, value, parameter_type=None)` — stores a fixed constant in `self.fixed` and removes any tunable parameter with the same name (likewise, `add_range`/`add_choice`/`add_derived` remove any fixed constant with the same name).
- `get_fixed(keys=None)` — returns a dict of fixed constants (all or filtered by `keys`).
- `remove(name)` — removes a parameter from both `self.params` and `self.fixed`.
- `merge(other, overwrite=True)` — merges another `TemplateBase`'s `params`, `fixed`, and `constraints` into `self`.
- `to_parameter_configs()` — converts `self.params` into `List[Union[RangeParameterConfig, ChoiceParameterConfig, DerivedParameterConfig]]`.
- `get_constraints()` — returns a copy of `self.constraints`.
- `to_dict()` / `from_dict(data)` — dictionary serialization.
- `save(filepath=None)` / `load(filepath)` — YAML serialization (default path: `configs/template_<name>.yaml`).

### 2. Template Inheritance Hierarchy (`template_base.py`)

```
TemplateBase
└── TemplateDL          # Core DL training, input/output MetaModel hierarchy, dataset & task defaults
    └── TemplateAtlas   # Modularity flags + V1 ModularSpace hyperparameters & loss weights
        ├── TemplateFC  # model_type="fc"
        └── TemplateCNN # model_type="cnn", cnn_kernel_size, modular_cnn_locality
```

### 3. Dataset-Specific Templates

| Class | File | Base | `dataset_name` | `input_data_shape` | `output_data_shape` | `nas_min_acc` |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| `TemplateMnistFC` | `template_mnist.py` | `TemplateFC` | `'mnist'` | `784` | `10` | `0.95` |
| `TemplateMnistCNN` | `template_mnist.py` | `TemplateCNN` | `'mnist'` | `28` | `10` | `0.95` |
| `TemplateFMnistFC` | `template_fmnist.py` | `TemplateFC` | `'fmnist'`* | `784` | `10` | `0.85` |
| `TemplateFMnistCNN` | `template_fmnist.py` | `TemplateCNN` | `'fmnist'`* | `28` | `10` | `0.95`* |
| `TemplateCovertype` | `template_covertype.py` | `TemplateFC` | `'covertype'` | `54` | `7` | `0.70` |
| `TemplateHiggs` | `template_higgs.py` | `TemplateFC` | `'higgs'` | `28` | `2` | `0.70` |

*\*Known issues: `DataProviderFactory` expects `'fashion_mnist'` rather than `'fmnist'`, and `TemplateFMnistCNN` currently inherits `0.95` from `TemplateDL` instead of `0.85` (see `analysis.md`).*

### 4. `TemplateFactory` & `TemplateVariantFactory` (`template_factory.py`)

- `TemplateFactory.EXPERIMENTS`: maps `'covertype'`, `'higgs'`, `'mnist_fc'`, `'mnist_cnn'`, `'fmnist_fc'`, `'fmnist_cnn'` to their template classes.
- `TemplateFactory.FromName(experiment, name=None) -> TemplateBase`: instantiates a registered template (raises `ValueError` for unknown keys).
- `TemplateFactory.AddTemplate(experiment, template_cls)`: registers a custom template class at runtime.
- `TemplateVariantFactory.create_standard_modularity_variants(base_name="variant") -> TemplateVariantList`: creates the canonical 4-variant ablation matrix:
  - `<base>_a`: `exp_metric_modularity=False`, `exp_modularizer=False` (Baseline)
  - `<base>_b`: `exp_metric_modularity=True`, `exp_modularizer=False` (Metric Only)
  - `<base>_c`: `exp_metric_modularity=False`, `exp_modularizer=True` (Modularizer Only)
  - `<base>_d`: `exp_metric_modularity=True`, `exp_modularizer=True` (Full Modularity)

### 5. `TemplateVariant` & `TemplateVariantList` (`template_variant.py`)

- `TemplateVariant(name, fixed_features=None, description="")`: represents an ablation variant fixing a subset of tunable parameters.
  - `validate(template)` — verifies every key in `fixed_features` exists in `template.params`.
  - `get_fixed_features()` — returns a shallow copy of `fixed_features`.
  - `matches_trial_parameters(parameters)` — checks whether an evaluated parameter dict matches the variant's fixed features.
  - `to_dict()` / `from_dict(data)` — dictionary serialization.
- `TemplateVariantList`: ordered registry of `TemplateVariant` objects (`add_variant`, `get_variant`, `list_variants`, `items`, `values`, `validate_all`, `to_dict`, `from_dict`, `__len__`, `__iter__`).

---

## Integration with `nas` and `pipeline`

In the refactored `positronixlab/nas` workflow:

1. `Experiment.__init__(name, template, objective, ...)` passes `template.to_parameter_configs()` and `template.get_constraints()` directly to Ax 1.3 `Client.configure_experiment(...)`.
2. For each generated trial, `TrialConfig.from_trial(params=ax_params, fixed=template.get_fixed(), ...)` merges `template.fixed` with Ax-sampled `params` (Ax parameters take precedence) into a typed `TrialConfig` subclass (composed from `MetaModelConfig`, `LayerSpaceConfig`, `TrainConfig`, `DataConfig`, or `PipelineConfigV1`/`PipelineConfigV2`).
3. `Experiment` snapshots the search space to `<exp_dir>/template.yaml` via `template.save(...)`.

---

## File Structure

```
positronixlab/template/
├── __init__.py               # Public exports
├── template_base.py          # ParameterType, TemplateBase, TemplateDL/Atlas/CNN/FC
├── template_variant.py       # TemplateVariant, TemplateVariantList
├── template_factory.py       # TemplateFactory, TemplateVariantFactory
├── template_mnist.py         # TemplateMnistFC, TemplateMnistCNN
├── template_fmnist.py        # TemplateFMnistFC, TemplateFMnistCNN
├── template_covertype.py     # TemplateCovertype
├── template_higgs.py         # TemplateHiggs
├── analysis.md               # Module audit, bugs, and recommendations
├── docs/
│   ├── CONFIG_SYSTEM.md      # This file
│   └── CONFIG_USAGE.md       # Usage guide
├── examples/
│   └── example_template.py   # Runnable CPU example
└── tests/
    ├── __init__.py
    └── test_template.py      # Unit tests
```

## Dependencies

- `ax-platform` (`>=1.3`, specifically `ax.api.configs`)
- `PyYAML` (`yaml`)
- Python standard library (`copy`, `enum`, `os`, `typing`)
