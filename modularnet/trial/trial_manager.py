



from modularnet.template.template_base import TemplateBase
from modularnet.template.template import Template


class TrialManager:
    def __init__(self, template: TemplateBase=None, options: dict=None):
        self.template = template
        self.options = options

    