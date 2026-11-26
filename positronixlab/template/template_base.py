from __future__ import annotations

import copy
from enum import Enum
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union
import warnings
import yaml

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=r".*torch\.jit\.script.*deprecated.*", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"^ax(\.|$)")
    from ax.api.configs import (
        ChoiceParameterConfig,
        DerivedParameterConfig,
        RangeParameterConfig,
    )


class ParameterType(str, Enum):
    INT = "int"
    FLOAT = "float"
    STRING = "str"
    BOOL = "bool"


ParameterConfig = Union[
    RangeParameterConfig,
    ChoiceParameterConfig,
    DerivedParameterConfig,
]


class TemplateBase:
    """
    Lightweight Ax 1.3 search-space builder.

    Separates tunable parameters (`self.params`) and linear constraints
    (`self.constraints`) from fixed non-searchable constants (`self.fixed`)
    so the Ax search space stays clean and free of noise.
    """

    CONFIG_PATH = "configs/"

    def __init__(self, name: Optional[str] = None):
        self.name: str = name if name is not None else self.__class__.__name__.lower()
        self.params: Dict[str, Dict[str, Any]] = {}
        self.fixed: Dict[str, Any] = {}
        self.constraints: List[str] = []

    # -------------------------------------------------------------------------
    # Search Space Building Methods (Tunable & Derived Parameters)
    # -------------------------------------------------------------------------
    def add_range(
        self,
        name: str,
        bounds: Sequence[Union[int, float]],
        parameter_type: Union[ParameterType, str] = ParameterType.FLOAT,
        scaling: Optional[str] = None,
        step_size: Optional[Union[int, float]] = None,
        log_scale: bool = False,
    ) -> "TemplateBase":
        p_type = parameter_type.value if isinstance(parameter_type, ParameterType) else str(parameter_type)
        resolved_scaling = scaling if scaling is not None else ("log" if log_scale else "linear")
        spec: Dict[str, Any] = {
            "name": name,
            "type": "range",
            "bounds": list(bounds),
            "parameter_type": p_type,
            "scaling": resolved_scaling,
        }
        if step_size is not None:
            spec["step_size"] = step_size
        self.fixed.pop(name, None)
        self.params[name] = spec
        return self

    def add_choice(
        self,
        name: str,
        values: Sequence[Any],
        parameter_type: Union[ParameterType, str] = ParameterType.STRING,
        is_ordered: Optional[bool] = None,
        dependent_parameters: Optional[Mapping[Any, Sequence[str]]] = None,
        log_scale: Optional[bool] = None,
    ) -> "TemplateBase":
        p_type = parameter_type.value if isinstance(parameter_type, ParameterType) else str(parameter_type)
        if is_ordered is None:
            # Explicit default avoids AxParameterWarning (2-value choices are always ordered in Ax)
            is_ordered = len(values) <= 2 or p_type in ("int", "float")
        spec: Dict[str, Any] = {
            "name": name,
            "type": "choice",
            "values": list(values),
            "parameter_type": p_type,
            "is_ordered": is_ordered,
        }
        if dependent_parameters is not None:
            spec["dependent_parameters"] = {k: list(v) for k, v in dependent_parameters.items()}
        if log_scale is not None:
            spec["log_scale"] = log_scale
        self.fixed.pop(name, None)
        self.params[name] = spec
        return self

    def add_derived(
        self,
        name: str,
        expression_str: str,
        parameter_type: Union[ParameterType, str] = ParameterType.FLOAT,
    ) -> "TemplateBase":
        """Register an Ax 1.3 `DerivedParameterConfig` (linear SymPy expression)."""
        p_type = parameter_type.value if isinstance(parameter_type, ParameterType) else str(parameter_type)
        self.fixed.pop(name, None)
        self.params[name] = {
            "name": name,
            "type": "derived",
            "expression_str": expression_str,
            "parameter_type": p_type,
        }
        return self

    def add_constraint(self, constraint: str) -> "TemplateBase":
        """Add a linear inequality parameter constraint (e.g. `'x1 <= x2'`)."""
        if constraint not in self.constraints:
            self.constraints.append(constraint)
        return self

    # -------------------------------------------------------------------------
    # Fixed Parameters (Excluded from Ax search space, merged into TrialConfig)
    # -------------------------------------------------------------------------
    def add_fixed(
        self,
        name: str,
        value: Any,
        parameter_type: Optional[Union[ParameterType, str]] = None,
    ) -> "TemplateBase":
        """Store a fixed non-searchable parameter (removes any tunable param of the same name)."""
        self.params.pop(name, None)
        self.fixed[name] = value
        return self

    def get_fixed(self, keys: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        if keys is not None:
            return {k: self.fixed[k] for k in keys if k in self.fixed}
        return dict(self.fixed)

    def remove(self, name: str) -> "TemplateBase":
        """Remove a parameter from both tunable and fixed dictionaries."""
        self.params.pop(name, None)
        self.fixed.pop(name, None)
        return self

    def merge(self, other: "TemplateBase", overwrite: bool = True) -> "TemplateBase":
        """Merge another template's tunable params, fixed constants, and constraints into this one."""
        for k, spec in other.params.items():
            if overwrite or (k not in self.params and k not in self.fixed):
                self.fixed.pop(k, None)
                self.params[k] = copy.deepcopy(spec)
        for k, val in other.fixed.items():
            if overwrite or (k not in self.fixed and k not in self.params):
                self.params.pop(k, None)
                self.fixed[k] = copy.deepcopy(val)
        for c in other.constraints:
            self.add_constraint(c)
        return self

    # -------------------------------------------------------------------------
    # Ax 1.3 Conversion
    # -------------------------------------------------------------------------
    def to_parameter_configs(self) -> List[ParameterConfig]:
        """Convert tunable/derived parameters into Ax 1.3 `*ParameterConfig` objects."""
        configs: List[ParameterConfig] = []

        for name, spec in self.params.items():
            p_kind = spec.get("type")
            p_type = spec.get("parameter_type") or spec.get("value_type", "float")

            if p_kind == "range":
                scaling = spec.get("scaling")
                if scaling is None:
                    scaling = "log" if spec.get("log_scale", False) else "linear"
                kwargs: Dict[str, Any] = {
                    "name": name,
                    "bounds": tuple(spec["bounds"]),
                    "parameter_type": p_type,
                    "scaling": scaling,
                }
                if spec.get("step_size") is not None:
                    kwargs["step_size"] = spec["step_size"]
                configs.append(RangeParameterConfig(**kwargs))

            elif p_kind == "choice":
                kwargs = {
                    "name": name,
                    "values": list(spec["values"]),
                    "parameter_type": p_type,
                    "is_ordered": spec.get("is_ordered"),
                }
                deps = spec.get("dependent_parameters") or spec.get("dependents")
                if deps is not None:
                    kwargs["dependent_parameters"] = deps
                if spec.get("log_scale") is not None:
                    kwargs["log_scale"] = spec["log_scale"]
                configs.append(ChoiceParameterConfig(**kwargs))

            elif p_kind == "derived":
                configs.append(
                    DerivedParameterConfig(
                        name=name,
                        expression_str=spec["expression_str"],
                        parameter_type=p_type,
                    )
                )

        return configs

    def get_constraints(self) -> List[str]:
        return list(self.constraints)

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "params": copy.deepcopy(self.params),
            "fixed": copy.deepcopy(self.fixed),
            "constraints": list(self.constraints),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemplateBase":
        tpl = cls(name=data.get("name", "template"))
        tpl.params = copy.deepcopy(data.get("params", {}))
        tpl.fixed = copy.deepcopy(data.get("fixed", {}))
        tpl.constraints = list(data.get("constraints", []))
        return tpl

    def save(self, filepath: Optional[str] = None) -> str:
        if filepath is None:
            filepath = os.path.join(self.CONFIG_PATH, f"template_{self.name}.yaml")
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, sort_keys=False)
        return filepath

    @classmethod
    def load(cls, filepath: str) -> "TemplateBase":
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)


class TemplateDL(TemplateBase):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

        self.add_fixed("nas_min_acc", 0.95, "float")
        self.add_fixed("task", "classification", "str")
        self.add_fixed("criterion_type", "cross_entropy", "str")
        self.add_fixed("model_type", "fc", "str")
        self.add_range("train_epochs", [1, 10], "int")

        self.add_fixed("template_name", self.name, "str")
        self.add_fixed("dataset_name", "mnist", "str")

        self.add_fixed("input_data_shape", 784, "int")
        self.add_range("input_hidden_dim", [64, 512], "int")
        self.add_range("input_num_layers", [2, 10], "int")
        self.add_range("input_reduce_every", [2, 8], "int")
        self.add_choice("input_dropout_every", [-1, 0, 1], "int")
        self.add_choice("input_batch_norm", [True, False], "bool")
        self.add_choice("input_residual", [True, False], "bool")

        self.add_fixed("output_data_shape", 10, "int")
        self.add_range("output_hidden_dim", [64, 256], "int")
        self.add_range("output_num_layers", [2, 4], "int")
        self.add_range("output_reduce_every", [0, 2], "int")
        self.add_choice("output_dropout_every", [-1, 0, 1], "int")
        self.add_choice("output_batch_norm", [True, False], "bool")
        self.add_choice("output_residual", [True, False], "bool")

        self.add_choice("activation", ["relu", "leaky_relu"], "str")
        self.add_range("dropout", [0.1, 0.5], "float")
        self.add_range("learning_rate", [0.0001, 0.01], "float", scaling="log")
        self.add_range("batch_size", [32, 128], "int")


class TemplateAtlas(TemplateDL):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

        self.add_choice("exp_metric_modularity", [True, False], "bool")
        self.add_choice("exp_modularizer", [True, False], "bool")
        self.add_fixed("nas_min_mod_score", 0.2, "float")

        self.add_range("modular_space_size", [3, 8], "int")
        self.add_range("modular_space_lr", [0.0001, 0.01], "float", scaling="log")
        self.add_range("modular_space_active_q", [0.6, 0.95], "float")
        self.add_choice("modular_space_distance", ["euclidean", "cosine"], "str")
        self.add_range("modular_modularize_decay", [0.8, 0.99], "float")

        self.add_range("modular_space_loss_on", [0.0, 1.0], "float")
        self.add_range("modular_space_loss_off", [0.0, 1.0], "float")
        self.add_range("modular_space_loss_centers", [0.0, 1.0], "float")
        self.add_range("modular_space_loss_batch", [0.0, 1.0], "float")
        self.add_range("modular_space_loss_alignment", [0.0, 1.0], "float")


class TemplateCNN(TemplateAtlas):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

        self.add_fixed("model_type", "cnn", "str")
        self.add_choice("cnn_kernel_size", [3, 5, 7], "int")
        self.add_range("modular_cnn_locality", [0.8, 0.99], "float")


class TemplateFC(TemplateAtlas):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

        self.add_fixed("model_type", "fc", "str")
        self.add_range("input_hidden_dim", [64, 512], "int")
