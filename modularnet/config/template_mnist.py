


from modularnet.config.template_fmnist import TemplateCNN, TemplateFC


class TemplateMnistFC(TemplateFC):

    def __init__(self, name: str=None):
        super().__init__(name)      
         
        self.add_fixed('dataset_name', 'mnist', "str")  # Default input data shape for MNIST
        #self.add_fixed('modular_space_name', name, "str")  # Default modular space name
        

class TemplateMnistCNN(TemplateCNN):

    def __init__(self, name: str=None):
        super().__init__(name)    
        
        self.add_fixed('input_data_shape', 28, "int")
        self.add_fixed('dataset_name', 'mnist', "str")  # Default input data shape for MNIST
        #self.add_fixed('modular_space_name', name, "str")  # Default modular space name
        self.add_choice('cnn_kernel_size', [3,5,7], "int")
