# `positronixlab/template` — Solved Issues Archive

> **Archived:** 2026-11-27.
> **Active open issues:** See [open/template_analysis.md](file:///home/cesare/Projects/PositronixLab/reports/open/template_analysis.md).

---

## Resolved Bugs

| ID | Summary | Resolution Evidence |
| :--- | :--- | :--- |
| **#2 (P0, 2026-11-27)** | `TemplateVariant.matches_trial_parameters()` returned `True` when fixed keys were missing | Fixed to `if k not in parameters or parameters[k] != v: return False` (`template_variant.py:46`); xfail removed and passing. |
| **#3 (P0, 2026-11-27)** | `TemplateFMnistFC`/`CNN` set `dataset_name='fmnist'` instead of `'fashion_mnist'` | Updated both to `'fashion_mnist'` (`template_fmnist.py:13, 24`); both parametrized xfails removed and passing. |
| **#5 (2026-11-27)** | `TemplateFMnistCNN` inherited `nas_min_acc=0.95` instead of `0.85` | Added `self.add_fixed('nas_min_acc', 0.85, "float")` (`template_fmnist.py:22`); xfail removed and passing. |
| **#1 (P1, 2026-11-27)** | `TemplateBase.to_parameter_configs()` crashed on `add_choice(..., log_scale=...)` in Ax 1.3 | Stopped forwarding `log_scale` to `ChoiceParameterConfig` (`template_base.py:199`); xfail removed and passing. |
| **#4 (P1, 2026-11-27)** | `TemplateBase.save()` / `load()` failed YAML round-trip on tuples | Switched `save()` to `yaml.safe_dump(to_plain_python(self.to_dict()), f, sort_keys=False)` (`template_base.py:241`); xfail removed and passing. |
| **#6 (P1, 2026-11-27)** | `TemplateBase.to_parameter_configs()` silently dropped unrecognized parameter kinds | Added `else: raise ValueError(...)` (`template_base.py:210`); xfail removed and passing. |
