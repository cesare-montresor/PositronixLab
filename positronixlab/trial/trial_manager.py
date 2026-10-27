



from positronixlab.template.template_base import TemplateBase


class TrialManager:
    def __init__(self, template: TemplateBase=None, options: dict=None):
        self.template = template
        self.options = options

    