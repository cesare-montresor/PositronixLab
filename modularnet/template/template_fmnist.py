     


from modularnet.template.template_base import TemplateFC, TemplateCNN


class TemplateFMnistFC(TemplateFC):

    def __init__(self, name: str=None):
        super().__init__(name)   
        
        self.add_fixed('nas_min_acc', 0.85, "float")
        self.add_fixed('dataset_name', 'fmnist', "str")  # Default input data shape for MNIST
        #self.add_fixed('modular_space_name', name, "str")  # Default modular space name


class TemplateFMnistCNN(TemplateCNN):

    def __init__(self, name: str=None):
        super().__init__(name)   
        
        self.add_fixed('input_data_shape', 28, "int")
        self.add_fixed('dataset_name', 'fmnist', "str")  # Default input data shape for MNIST
        #self.add_fixed('modular_space_name', name, "str")  # Default modular space name
        self.add_choice('cnn_kernel_size', [3,5,7], "int")
