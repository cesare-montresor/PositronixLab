
from positronixlab.template.template_base import TemplateFC


class TemplateHiggs(TemplateFC):

    def __init__(self, name:str = None):
        super().__init__(name)   


        self.add_fixed('nas_min_acc', 0.70, "float")  # Baseline XGBoost/RF accuracy
        
        
        self.add_range('train_epochs', [15, 100], "int")  # Large dataset needs more epochs
        self.add_fixed('dataset_name', 'higgs', "str")
        
        self.add_fixed('input_data_shape', 28, "int")  # 28 features (21 low-level + 7 high-level)
        self.add_range('input_hidden_dim', [128, 1024], "int")  # Larger for complex physics data
        
        self.add_range('input_num_layers', [3, 10], "int")  # Deep networks work well
        self.add_range('input_reduce_every', [2, 5], "int")
        self.add_choice('input_dropout_every', [-1, 0, 1, 2], "int")  # More dropout options
        
        self.add_fixed('output_data_shape', 2, "int")  # Binary: signal vs background
        self.add_range('output_hidden_dim', [64, 512], "int")
        self.add_range('output_num_layers', [2, 6], "int")
        self.add_range('output_reduce_every', [0, 3], "int")
        self.add_choice('output_dropout_every', [-1, 0, 1], "int")
        
        self.add_range('batch_size', [128, 512], "int") 
        self.add_choice('activation', ['relu', 'leaky_relu', 'elu'], "str")  # More activation options

        