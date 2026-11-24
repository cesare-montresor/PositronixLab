

from positronixlab.template.template_base import TemplateBase
from positronixlab.template.template_fmnist import TemplateFMnistCNN, TemplateFMnistFC
from positronixlab.template.template_mnist import TemplateMnistCNN, TemplateMnistFC
from positronixlab.template.template_covertype import TemplateCovertype
from positronixlab.template.template_higgs import TemplateHiggs
from positronixlab.template.template_variant import TemplateVariant, TemplateVariantList


class TemplateFactory:
    
    EXPERIMENTS = {
        'covertype': TemplateCovertype,
        'higgs': TemplateHiggs,
        'mnist_fc': TemplateMnistFC,
        'mnist_cnn': TemplateMnistCNN,
        'fmnist_fc': TemplateFMnistFC,
        'fmnist_cnn': TemplateFMnistCNN
    }

    @classmethod
    def AddTemplate(cls, experiment:str, template_cls: TemplateBase):
        TemplateFactory.EXPERIMENTS[experiment] = template_cls
    

    @classmethod
    def FromName(cls, experiment:str, name:str=None) -> TemplateBase:
        template_class = TemplateFactory.EXPERIMENTS.get(experiment, None)
        
        if template_class is None:
            raise ValueError(f"Unknown experiment: {experiment}")
        
        return template_class(name=name)
        


class TemplateVariantFactory:
    
    
    @classmethod
    def create_standard_modularity_variants(
        cls, base_name: str = "variant"
    ) -> "TemplateVariantList":
        """
        Creates the canonical 4-variant matrix for modularity experiments:
        - _a: No modularity metric, No modularizer
        - _b: Modularity metric, No modularizer
        - _c: No modularity metric, Modularizer enabled
        - _d: Modularity metric, Modularizer enabled
        """
        vl = TemplateVariantList()
        vl.add_variant(
            TemplateVariant(
                name=f"{base_name}_a",
                fixed_features={"exp_metric_modularity": False, "exp_modularizer": False},
                description="Baseline (No Modularity metric, No Modularizer)",
            )
        )
        vl.add_variant(
            TemplateVariant(
                name=f"{base_name}_b",
                fixed_features={"exp_metric_modularity": True, "exp_modularizer": False},
                description="Metric Only (Modularity metric enabled, No Modularizer)",
            )
        )
        vl.add_variant(
            TemplateVariant(
                name=f"{base_name}_c",
                fixed_features={"exp_metric_modularity": False, "exp_modularizer": True},
                description="Modularizer Only (Modularizer active, Metric disabled)",
            )
        )
        vl.add_variant(
            TemplateVariant(
                name=f"{base_name}_d",
                fixed_features={"exp_metric_modularity": True, "exp_modularizer": True},
                description="Full Modularity (Both Metric and Modularizer enabled)",
            )
        )
        return vl


