

from positronixlab.template.template_base import TemplateBase
from positronixlab.template.template_fmnist import TemplateFMnistCNN, TemplateFMnistFC
from positronixlab.template.template_mnist import TemplateMnistCNN, TemplateMnistFC
from positronixlab.template.template_covertype import TemplateCovertype
from positronixlab.template.template_higgs import TemplateHiggs



__all__ = [
    "TemplateFMnistCNN",
    "TemplateFMnistFC",
    "TemplateMnistCNN",
    "TemplateMnistFC",
    "TemplateCovertype",
    "TemplateHiggs",
    "TrialConfig",
]