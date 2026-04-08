

from modularnet.config.base_exp_template import BaseExpTemplate


class ExpTemplate:
    
    EXPERIMENTS = [ 'covertype', 'higgs', 'mnist_fc', 'fmnist_fc', 'mnist_cnn', 'fmnist_cnn' ]
    

    @staticmethod
    def FromName(experiment:str):
        if experiment not in BaseExpTemplate.EXPERIMENTS:
            raise ValueError(f"Unknown experiment: {experiment}")
    

        if experiment == 'mnist_fc':
            template = TemplateMnistFC()
        elif experiment == 'fmnist_fc':
            template = TemplateFMnistFC()
        elif experiment == 'mnist_cnn':
            template = TemplateMnistCNN()
        elif experiment == 'fmnist_cnn':
            template = TemplateFMnistCNN()
        elif experiment == 'covertype':
            template = TemplateCovertype()
        elif experiment == 'higgs':
            template = TemplateHiggs()
        
        
        return template


    


