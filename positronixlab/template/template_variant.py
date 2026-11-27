"""
TemplateVariant: Classes for managing variant definitions and fixed parameter sets.
"""

from typing import Any


class TemplateVariant:
    """
    Represents a specific variant of a template with fixed hyperparameters.
    In NAS, variants fix architectural choices (e.g. modularity on/off, residual connections)
    while sharing the broader continuous and discrete search space.
    """

    def __init__(
        self,
        name: str,
        fixed_features: dict[str, Any] | None = None,
        description: str = "",
    ):
        self.name = name
        self.fixed_features: dict[str, Any] = fixed_features or {}
        self.description = description

    def validate(self, template: Any) -> None:
        """
        Validates that all fixed features exist in the base template's parameter space.
        """
        if hasattr(template, "params"):
            for key in self.fixed_features:
                if key not in template.params:
                    raise ValueError(
                        f"Fixed feature '{key}' is not defined in template '{template.name}'."
                    )

    def get_fixed_features(self) -> dict[str, Any]:
        """Returns the dictionary for Ax FixedFeatures."""
        return dict(self.fixed_features)

    def matches_trial_parameters(self, parameters: dict[str, Any]) -> bool:
        """
        Checks if a given set of evaluated parameters matches this variant's fixed features.
        Used to identify common phase prior trials that can bootstrap this variant.
        """
        for k, v in self.fixed_features.items():
            if k not in parameters or parameters[k] != v:
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "fixed_features": self.fixed_features,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TemplateVariant":
        return cls(
            name=data["name"],
            fixed_features=data.get("fixed_features", {}),
            description=data.get("description", ""),
        )

    def __repr__(self) -> str:
        return f"TemplateVariant(name='{self.name}', fixed_features={self.fixed_features})"


class TemplateVariantList:
    """Manages a registry of TemplateVariant instances for an experiment."""

    def __init__(self):
        self.variants: dict[str, TemplateVariant] = {}

    def add_variant(self, variant: TemplateVariant) -> None:
        """Adds a variant to the list."""
        if variant.name in self.variants:
            raise ValueError(f"Variant '{variant.name}' already exists.")
        self.variants[variant.name] = variant

    def get_variant(self, name: str) -> TemplateVariant:
        """Retrieves a variant by name."""
        if name not in self.variants:
            raise KeyError(f"Variant '{name}' does not exist.")
        return self.variants[name]

    def list_variants(self) -> list[str]:
        """Lists all registered variant names."""
        return list(self.variants.keys())

    def items(self):
        return self.variants.items()

    def values(self):
        return self.variants.values()

    def __len__(self) -> int:
        return len(self.variants)

    def __iter__(self):
        return iter(self.variants.values())

    def validate_all(self, template: Any) -> None:
        """Validates all variants against the provided template."""
        for variant in self.variants.values():
            variant.validate(template)

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return {name: v.to_dict() for name, v in self.variants.items()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TemplateVariantList":
        vl = cls()
        for _, item in data.items():
            vl.add_variant(TemplateVariant.from_dict(item))
        return vl
