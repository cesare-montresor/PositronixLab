

from modularnet.template.template_base import TemplateBase
from modularnet.template.template_fmnist import TemplateFMnistCNN, TemplateFMnistFC
from modularnet.template.template_mnist import TemplateMnistCNN, TemplateMnistFC
from modularnet.template.template_covertype import TemplateCovertype
from modularnet.template.template_higgs import TemplateHiggs



__all__ = [
    "TemplateFMnistCNN",
    "TemplateFMnistFC",
    "TemplateMnistCNN",
    "TemplateMnistFC",
    "TemplateCovertype",
    "TemplateHiggs",
    "TrialConfig",
]