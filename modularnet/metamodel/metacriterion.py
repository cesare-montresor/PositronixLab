
from torch import nn


class MetaCriterion:
    """
    A class to hold options for the MetaCriterion.
    This class can be extended to include various configuration parameters.
    """
    def __init__(self, options):
        self.custom_losses = {}
        self.task =  options.task  # Default task is regression
        self.criterion_type = options.criterion_type

    def custom_loss(self, name, function):
        self.custom_losses[name] = function
    
    def loss(self, x, y, y_hat):
        if self.task == 'autoencoder': y = x

        name = self.criterion_type
        if name == 'cross_entropy':
            return nn.functional.cross_entropy(y_hat, y)        
        elif name == 'mse':
            return nn.functional.mse_loss(y_hat, y)
        elif name == 'mae':
            return nn.functional.l1_loss(y_hat, y)
        elif name == 'huber':
            return nn.functional.huber_loss(y_hat, y)
        elif name == 'bce_with_logits':
            return nn.functional.binary_cross_entropy_with_logits(y_hat, y)
        elif name == 'bce':
            return nn.functional.binary_cross_entropy(y_hat, y)
        elif name in self.custom_losses:
            return self.custom_losses[name](y_hat, y)
        else:
            raise ValueError(f"Unsupported criterion type: {name}")
        


