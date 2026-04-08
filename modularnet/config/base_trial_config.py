
class BaseTrialConfig:
    """
    A class to hold options for the MetaModel.
    This class can be extended to include various configuration parameters.
    """
    def __init__(self, options):
        self.options = options

        self.exp_metric_modularity = options.get('exp_metric_modularity', True)
        self.exp_modularizer = options.get('exp_modularizer', True)

        
        self.task = options.get('task', "classification")  # Default task is classificationù
        self.criterion_type = options.get('criterion_type', "mse")  # Default task is classification
        self.model_type = options.get('model_type', "fc")  # Default model type is fully connected (fc)
        self.train_epochs = options.get('train_epochs', 10)  # Default number of training epochs
        self.space_epochs = options.get('space_epochs', 1)  # no training. Default number of space epochs ( optimized for unchaning model weights)

        self.dataset_name = options.get('dataset_name', 'fashion_mnist')  # Default input data shape for MNIST
        self.modular_space_name = options.get('modular_space_name', 'fashion_mnist_cnn')  # Default modular space name
        self.modular_space_size = options.get('modular_space_size', 6)  # Default modular space size
        self.modular_space_lr = options.get('modular_space_lr', 0.01)
        self.modular_space_active_q = options.get('modular_space_active_q', 0.80) # Default modular space size
        self.modular_space_distance = options.get('modular_space_distance', 'cosine')  # Default modular space size
        self.modular_modularize_decay = options.get('modular_modularize_decay', 0.95)  # Default modular space size
        

        self.modular_space_loss_on = options.get('modular_space_loss_on', 1)
        self.modular_space_loss_off = options.get('modular_space_loss_off', 1)
        self.modular_space_loss_centers = options.get('modular_space_loss_centers', 1)  
        self.modular_space_loss_batch = options.get('modular_space_loss_batch', 1)  
        self.modular_space_loss_alignment = options.get('modular_space_loss_alignment', 1) 
        
        self.input_data_shape = options.get('input_data_shape', 784) 
        self.input_hidden_dim = options.get('input_hidden_dim', 32) 

        self.input_num_layers = options.get('input_num_layers', 2)
        self.input_reduce_every = options.get('input_reduce_every', 0)  # Default input size reduced every N layer. <1 no reduction
        self.input_dropout_every = options.get('input_dropout_every', 0) # 0 first only, -1 none, N>0 every N layer)
        self.input_batch_norm = options.get('input_batch_norm', True)
        self.input_residual = options.get('input_residual', True)


        self.output_data_shape = options.get('output_data_shape', 10) # -1: auto
        self.output_hidden_dim = options.get('output_hidden_dim', 32) # -1: auto
        self.output_num_layers = options.get('output_num_layers', 2) # -1: auto
        self.output_reduce_every = options.get('output_reduce_every', -1)  # Default input size reduced every N layer. <1 no reduction
        self.output_dropout_every = options.get('output_dropout_every', 0) # 0 first only, -1 none, N>0 every N layer)
        self.output_batch_norm = options.get('output_batch_norm', True)
        self.output_residual = options.get('output_residual', True)


        
        self.activation = options.get('activation', 'relu')
        self.dropout = options.get('dropout', 0.5)
        self.learning_rate = options.get('learning_rate', 0.001)
        self.batch_size = options.get('batch_size', 32)
        self.epochs = options.get('epochs', 10)


        self.nas_min_acc = options.get('nas_min_acc', 0.8)
        self.nas_min_mod_score = options.get('nas_min_mod_score', 0.2)
        #self.optimizer = options.get('optimizer', 'adam')
        self.cnn_kernel_size = options.get('cnn_kernel_size', 3)

    def signature(self):
        sign = signature_settings(self)
        return sign

    def save(self, filepath):
        """
        Save the options to a YAML file.
        """
        with open(filepath, 'w') as f:
            yaml.dump(self.options, f, sort_keys=True)

    @classmethod
    def load(cls, filepath):
        """
        Load options from a YAML file and return a MetaModelOptions instance.
        """
        with open(filepath, 'r') as f:
            options = yaml.safe_load(f)
        return cls(**options)
    
