



from positronixlab.template.template_base import TemplateBase
from positronixlab.template.template_factory import TemplateFactory


class TrialManager:
    def __init__(self, template: TemplateBase=None, options: dict=None):
        self.template = template
        self.options = options

    