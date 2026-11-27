# `positronixlab/template` — Open Issues Analysis

> **Status (2026-11-27):** **28 passed, 0 xfailed, 0 failed** (`tests/template/`).
> **Solved issues archive:** Resolved P0/P1 bugs #1–#6 moved to [solved/template_solved.md](file:///home/cesare/Projects/PositronixLab/reports/solved/template_solved.md).
> **Note:** Remaining low/medium issues (#7–#11) remain open, plus v0.1.3 **E5** (top-level `ax.api.configs` import without `[nas]` hint).

---

## 1. Overview & Refactor Status

The `positronixlab/template` module provides lightweight, Ax 1.3-compatible search-space builders, dataset-specific default templates, a template factory registry, and experiment variant definitions for ablation studies.

| File | Responsibility |
| :--- | :--- |
| `positronixlab/template/__init__.py` | Public exports: `TemplateFactory`, `TemplateBase`, `TemplateFMnistCNN`, `TemplateFMnistFC`, `TemplateMnistCNN`, `TemplateMnistFC`, `TemplateCovertype`, `TemplateHiggs`. |
| `positronixlab/template/template_base.py` | `ParameterType`, `ParameterConfig`, `TemplateBase` (Ax 1.3 search-space builder with `params`, `fixed`, `constraints`, YAML serialization), and inheritance hierarchy `TemplateDL` → `TemplateAtlas` → `TemplateCNN` / `TemplateFC`. |
| `positronixlab/template/template_variant.py` | `TemplateVariant` and `TemplateVariantList` for defining fixed-feature ablation variants and matching prior trial parameters. |
| `positronixlab/template/template_factory.py` | `TemplateFactory` (`EXPERIMENTS` registry, `AddTemplate`, `FromName`) and `TemplateVariantFactory` (`create_standard_modularity_variants`). |
| `positronixlab/template/template_mnist.py` | `TemplateMnistFC`, `TemplateMnistCNN`. |
| `positronixlab/template/template_fmnist.py` | `TemplateFMnistFC`, `TemplateFMnistCNN`. |
| `positronixlab/template/template_covertype.py` | `TemplateCovertype`. |
| `positronixlab/template/template_higgs.py` | `TemplateHiggs`. |

**Refactor status:**
- `TemplateBase` was cleanly modernized for **Ax 1.3+** (`RangeParameterConfig`, `ChoiceParameterConfig`, `DerivedParameterConfig`, linear `constraints`, and strict separation of tunable `self.params` from non-searchable `self.fixed`). All 6 factory templates convert cleanly via `to_parameter_configs()` and pass `ax.api.client.Client.configure_experiment()`.
- The **subclass hierarchy** (`TemplateDL` → `TemplateAtlas` → `TemplateCNN`/`TemplateFC`) and **dataset templates** are still aligned with the **V1 (`PipelineConfigV1` / `ModularSpace`)** parameter schema and have **not** been updated for `PipelineConfigV2` / `LayerSpaceConfig` or the refactored `nas.Experiment` + `TrialConfig` flow.
- `TemplateVariant`, `TemplateVariantList`, and `TemplateVariantFactory` are currently **orphaned** (only referenced by `positronixlab/nas/legacy/experiment.py`; the refactored `nas.Experiment` uses `warm_start_from(..., filter_fn=...)` and has not yet re-integrated variants).
- Prior to this review, `positronixlab/template/tests/` contained only an empty `__init__.py`, `examples/` did not exist, and `docs/CONFIG_{SYSTEM,USAGE}.md` documented the pre-refactor `positronixlab.config.template` API.

---

## 2. Bugs & Defects

### Medium Severity

7. **`TemplateVariant.validate()` performs incomplete validation**
   - **Location:** `positronixlab/template/template_variant.py:25-34`
   - **Root cause:** Only checks `if key not in template.params:`. Does not verify that the fixed value lies within `bounds` (range) or `values` (choice), does not reject fixing a `derived` parameter, and silently succeeds when `template` lacks a `.params` attribute (`if hasattr(template, "params"):`).
   - **Impact:** Invalid variant values (e.g., wrong type, out-of-range, or non-existent choice) pass validation and only fail later inside Ax.

### Low Severity

8. **Mutable dict aliasing in `TemplateVariant` and `TemplateBase.get_fixed()`**
   - **Location:** `positronixlab/template/template_variant.py:22, 53, 61`; `positronixlab/template/template_base.py:142`
   - **Root cause:** `TemplateVariant.to_dict()` returns `self.fixed_features` directly without copying, `__init__`/`from_dict()` retain the caller's dict reference, and `TemplateBase.get_fixed()` performs only a shallow `dict(self.fixed)` copy.
   - **Impact:** External mutations to returned or input dicts silently mutate internal template/variant state.

9. **`TemplateBase.add_fixed()` accepts an unused `parameter_type` argument**
   - **Location:** `positronixlab/template/template_base.py:128-137`
   - **Root cause:** `parameter_type` is neither stored nor used for validation/coercion (leftover from the pre-refactor API).

10. **`TemplateFactory` hardcodes `TemplateFactory.EXPERIMENTS` instead of `cls.EXPERIMENTS` and has an inaccurate type hint**
    - **Location:** `positronixlab/template/template_factory.py:23-29`
    - **Root cause:** `AddTemplate` annotates `template_cls: TemplateBase` instead of `Type[TemplateBase]`, and both `AddTemplate` and `FromName` reference `TemplateFactory.EXPERIMENTS` directly rather than `cls.EXPERIMENTS` (breaking subclass registries).

11. **Transitive import, redundant overrides, and copy-paste comments in dataset templates**
    - **Location:** `template_mnist.py:4, 12, 21, 22`; `template_fmnist.py:13, 23, 24`; `template_covertype.py:13`
    - **Details:** `template_mnist.py:4` imports `TemplateCNN, TemplateFC` from `template_fmnist` instead of `template_base`. Both CNN templates re-declare `cnn_kernel_size` identically to `TemplateCNN`. `template_covertype.py:13` and `template_fmnist.py:13, 23` contain stale `# Default input data shape for MNIST` comments on `dataset_name`.

---

## 3. Inconsistencies & Design Issues

### Parameter Alignment with `PipelineConfigV1`, `PipelineConfigV2`, `TrialConfig`, and Domain Mixins

Comparing all 40 keys in `TemplateCNN` (31 tunable + 9 fixed) against the refactored config dataclasses:

| Target Config | Keys in Template but **missing** from Config (silently dropped by `from_dict`) | Keys in Config but **missing** from Template |
| :--- | :--- | :--- |
| **`PipelineConfigV1`** (`pipeline_config_v1.py`) | `template_name`, `modular_cnn_locality` | `name`, `seed`, `device`, `space_epochs`, `optimizer`, `epochs`, `modular_overspace_size` |
| **`PipelineConfigV2`** (`pipeline_config_v2.py`) | `template_name`, `modular_space_loss_alignment` | `name`, `seed`, `device`, `space_epochs`, `optimizer`, `epochs`, `modular_overspace_size`, plus **all 13 V2 `LayerSpace` knobs**: `modular_space_scoring`, `modular_space_reducer`, `modular_space_apply_activation`, `modular_space_weighting`, `modular_space_contrast`, `modular_space_loss_global`, `modular_space_loss_cluster`, `modular_space_loss_sigreg`, `modular_space_tau`, `modular_space_freq_alpha`, `modular_space_sigreg_method`, `modular_space_sigreg_target`, `modular_space_sigreg_dim` |
| **Composed mixins** (`MetaModelConfig` + `LayerSpaceConfig` + `TrainConfig` + `DataConfig` + `TrialConfig`) | `exp_metric_modularity`, `exp_modularizer`, `nas_min_acc`, `nas_min_mod_score`, `template_name`, `modular_space_loss_alignment` | Same V2 `LayerSpace` knobs + `optimizer`, `epochs`, `space_epochs`, `modular_overspace_size`, `seed`, `device`, `experiment_name`, `trial_index`, `trial_dir`, `gpu_id` |

### Other Design & API Inconsistencies

- **Incomplete `__init__.py` exports:** `positronixlab/template/__init__.py` does not export `ParameterType`, `ParameterConfig`, `TemplateDL`, `TemplateAtlas`, `TemplateFC`, `TemplateCNN`, `TemplateVariant`, `TemplateVariantList`, or `TemplateVariantFactory`.
- **Naming convention split:** `TemplateFactory.AddTemplate` and `TemplateFactory.FromName` use PascalCase, whereas `TemplateBase`, `TemplateVariant`, `TemplateVariantList`, and `TemplateVariantFactory.create_standard_modularity_variants` use `snake_case`.
- **Missing dataset templates:** `DataProviderFactory` supports 11 datasets (`iris`, `mnist`, `fashion_mnist`, `covertype`, `higgs`, `cifar10`, `cifar100`, `scan`, `deepmind_math`, `cvr_odd_one_out`, `syneval_odd_one_out`), whereas `TemplateFactory.EXPERIMENTS` covers only 4 datasets (6 entries).
- **Orphaned variant subsystem:** `TemplateVariant` / `TemplateVariantList` / `TemplateVariantFactory` have no active consumer in the refactored `nas.Experiment` (though `TemplateVariant.matches_trial_parameters` could be passed as `filter_fn` to `Experiment.warm_start_from()` once Bug #2 is fixed).
- **`scaling` vs `log_scale` in `add_range()`:** `TemplateDL` uses `scaling="log"` (`template_base.py:282`), while `TemplateCovertype` uses `log_scale=True` (`template_covertype.py:36`). Both work, but if both are passed simultaneously (`scaling="linear", log_scale=True`), `scaling` silently wins.
- **No `clone()` helper:** Unlike `BaseConfig` / pre-refactor `TemplateBase`, there is no `.clone()` method (callers must use `copy.deepcopy(template)` or `TemplateBase.from_dict(template.to_dict())`).

---

## 4. Opportunities & Recommendations

1. **Fix the 3 high-severity bugs first:**
   - Remove `log_scale` forwarding to `ChoiceParameterConfig` in `TemplateBase.add_choice()` / `to_parameter_configs()` (or deprecate the parameter cleanly).
   - Fix `TemplateVariant.matches_trial_parameters()` to require `k in parameters and parameters[k] == v`.
   - Change `dataset_name` in `TemplateFMnistFC` and `TemplateFMnistCNN` from `'fmnist'` to `'fashion_mnist'` (and set `nas_min_acc=0.85` on `TemplateFMnistCNN`).
2. **Harden `TemplateBase` & `TemplateVariant`:**
   - Use `yaml.safe_dump` in `TemplateBase.save()` and validate `bounds` (`len == 2`, `bounds[0] < bounds[1]`) and unknown `p_kind` in `to_parameter_configs()`.
   - Enhance `TemplateVariant.validate()` to check value membership/bounds and reject derived parameters; deep-copy dicts in `to_dict()` / `from_dict()` / `get_fixed()`.
   - Add a `TemplateBase.clone()` convenience method and export `ParameterType`, `TemplateDL`/`Atlas`/`FC`/`CNN`, `TemplateVariant`, `TemplateVariantList`, `TemplateVariantFactory` in `__init__.py`.
3. **Add a V2 / `LayerSpace` template mixin or subclass (e.g. `TemplateAtlasV2`):**
   - Mirror `LayerSpaceConfig` / `PipelineConfigV2` fields (`modular_space_scoring`, `modular_space_contrast`, `modular_space_loss_cluster`, `modular_space_loss_sigreg`, `modular_space_sigreg_method`, etc.) so Ax sweeps can target `LayerSpace` directly without manually adding 13 parameters.
4. **Reconnect `TemplateVariant` to `nas.Experiment`:**
   - Add a helper such as `template.apply_variant(variant) -> TemplateBase` (fixing the variant's `fixed_features` in a cloned template) and pair `variant.matches_trial_parameters` with `Experiment.warm_start_from(..., filter_fn=variant.matches_trial_parameters)`.
5. **Expand `TemplateFactory` coverage & modernize naming:**
   - Add templates for `iris` and `cifar10`/`cifar100`, fix `cls.EXPERIMENTS` and `Type[TemplateBase]` typing, and optionally provide `snake_case` aliases (`from_name`, `add_template`, `list_templates`).

---

## 5. Docs / Tests / Examples Summary

- **Docs updated:**
  - `positronixlab/template/docs/CONFIG_SYSTEM.md` — Rewritten to accurately document `TemplateBase` (Ax 1.3 `ParameterConfig` conversion, `params` vs `fixed` vs `constraints`), `TemplateDL`/`Atlas`/`FC`/`CNN`, `TemplateFactory`, `TemplateVariant` / `TemplateVariantList` / `TemplateVariantFactory`, dataset templates, and alignment with `TrialConfig` / `PipelineConfigV1` / `PipelineConfigV2`.
  - `positronixlab/template/docs/CONFIG_USAGE.md` — Rewritten with accurate, runnable examples for `TemplateBase`, `TemplateFactory`, `TemplateVariant` / `TemplateVariantFactory`, YAML persistence, and Ax 1.3 `Client` / `TrialConfig` integration.
- **Tests created:**
  - `positronixlab/template/tests/test_template.py` — Comprehensive unit test suite covering `TemplateBase` (range, choice, derived, fixed, constraints, remove, merge, `to_parameter_configs`, YAML/dict round-trips), inheritance hierarchy, all 6 `TemplateFactory` templates + Ax 1.3 `Client.configure_experiment()` integration, `TemplateVariant` / `TemplateVariantList` / `TemplateVariantFactory`, plus `@pytest.mark.xfail(strict=True)` regression tests for confirmed bugs.
- **Example created:**
  - `positronixlab/template/examples/example_template.py` — Self-contained runnable CPU example demonstrating template creation, factory lookup, customization, variant matrix creation, Ax 1.3 `Client.configure_experiment()` + trial generation, fixed+sampled parameter merging, and YAML round-trip.
