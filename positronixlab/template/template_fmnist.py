     


from positronixlab.template.template_base import TemplateCNN, TemplateFC


class TemplateFMnistFC(TemplateFC):

    def __init__(self, name: str | None = None):
        super().__init__(name)   
        
        self.add_fixed('nas_min_acc', 0.85, "float")
        self.add_fixed('dataset_name', 'fashion_mnist', "str")
        


class TemplateFMnistCNN(TemplateCNN):

    def __init__(self, name: str | None = None):
        super().__init__(name)   
        
        self.add_fixed('nas_min_acc', 0.85, "float")
        self.add_fixed('input_data_shape', 28, "int")
        self.add_fixed('dataset_name', 'fashion_mnist', "str")
        self.add_choice('cnn_kernel_size', [3,5,7], "int")
