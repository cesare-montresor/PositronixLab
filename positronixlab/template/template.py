

from positronixlab.template.template_base import TemplateBase
from positronixlab.template.template_fmnist import TemplateFMnistCNN, TemplateFMnistFC
from positronixlab.template.template_mnist import TemplateMnistCNN, TemplateMnistFC
from positronixlab.template.template_covertype import TemplateCovertype
from positronixlab.template.template_higgs import TemplateHiggs


class Template:
    
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
        Template.EXPERIMENTS[experiment] = template_cls
    

    @classmethod
    def FromName(cls, experiment:str):
        template_class = Template.EXPERIMENTS.get(experiment, None)
        
        if template_class is None:
            raise ValueError(f"Unknown experiment: {experiment}")
        
        template = template_class()

        return template()
        


    


