from typing import Any

from positronixlab.template import TemplateBase

class TemplateVariant:
    """Represents a specific variant of a template with fixed features."""
    
    def __init__(self, name: str, template: TemplateBase, fixed_features: dict[str, Any] = None):
        self.name = name
        self.template = template
        self.fixed_features = fixed_features or {}
        self._validate_features()

    def _validate_features(self):
        """Ensures all fixed features actually exist in the base template."""
        for key in self.fixed_features:
            if key not in self.template.params:
                raise ValueError(f"Feature '{key}' is not defined in the template '{self.template.name}'.")

    def get_fixed_features_dict(self) -> dict[str, Any]:
        """Returns the dictionary for Ax FixedFeatures."""
        return self.fixed_features
    
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'template_name': self.template.name,
            'fixed_features': self.fixed_features
        }

