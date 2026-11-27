

from positronixlab.template.template_base import TemplateFC


class TemplateCovertype(TemplateFC):

    def __init__(self, name: str | None = None):
        super().__init__(name)   
        

        
        self.add_fixed('dataset_name', 'covertype', "str")  # Default input data shape for MNIST
        self.add_range('train_epochs', [10, 50], "int")  # Covertype may need more epochs
        


        self.add_fixed('nas_min_acc', 0.70, "float")  # Covertype baseline ~70-75% accuracy ( sota 80% )
        

        self.add_fixed('input_data_shape', 54, "int") 
        self.add_range('input_hidden_dim', [64, 512], "int")  # Wider range for smaller input
        self.add_range('input_num_layers', [2, 8], "int")  # Slightly fewer layers
        self.add_range('input_reduce_every', [2, 4], "int")  # Less aggressive reduction
        self.add_choice('input_dropout_every', [-1, 0, 1], "int")  # Added every layer option
        
        self.add_fixed('output_data_shape', 7, "int")  # 7 forest cover types
        self.add_range('output_hidden_dim', [32, 256], "int")  # Wider range
        self.add_range('output_num_layers', [2, 5], "int")  # Allow deeper output
        self.add_range('output_reduce_every', [0, 3], "int")
        self.add_choice('output_dropout_every', [-1, 0, 1], "int")
        
        
        self.add_choice('activation', ['relu', 'leaky_relu'], "str")  # Added leaky_relu option
        self.add_range('dropout', [0.0, 0.4], "float")  # Slightly less dropout
        self.add_range('learning_rate', [0.0001, 0.01], "float", log_scale=True)  # Wider LR range
        self.add_range('batch_size', [64, 256], "int")  # Larger batches for tabular data



