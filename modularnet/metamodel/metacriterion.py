
from abc import ABC

from torch import nn

from modularnet.trial.trial_config import TrialConfig


class MetaCriterion:
    """
    A class to hold options for the MetaCriterion.
    This class can be extended to include various configuration parameters.
    """
    def __init__(self, config: TrialConfig):
        self.custom_losses = {}
        self.task =  config.task  # Default task is regression
        self.criterion_type = config.criterion_type

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
            return self.custom_losses[name](x, y, y_hat)
        else:
            raise ValueError(f"Unsupported criterion type: {name}")
        
class MetaMetric:
    def __init__(self, config: TrialConfig, metric_type: str = None):
        self.custom_metric = {}
        self.task =  config.task  # Default task is regression
        self.metric_type = metric_type 
        if self.metric_type is None:
            if self.task == 'classification':
                self.metric_type = 'accuracy'
            elif self.task == 'segmentation':
                self.metric_type = 'ioU'
            else:
                self.metric_type = 'f1_score'  # Default metric for regression
          # Use task as metric type for simplicity

    def custom_metric(self, name, function):
        self.custom_metric[name] = function
    
    def metric(self, x, y, y_hat):
        result = None 
        if self.metric_type == 'f1_score':
            torch_f1_score = (2 * (y_hat * y).sum()) / ((y_hat + y).sum() + 1e-8)
            result = torch_f1_score.detach()
        elif self.metric_type == 'accuracy':
            torch_accuracy = (y_hat.argmax(dim=1) == y).float().mean()
            result = torch_accuracy
        elif self.metric_type == 'ioU':
            intersection = (y_hat * y).sum()
            union = y_hat.sum() + y.sum() - intersection
            iou = intersection / (union + 1e-8)
            result = iou
        elif self.metric_type in self.custom_metric:
            result = self.custom_metric[self.metric_type](x, y, y_hat)
        else:
            raise ValueError(f"Unsupported metric type: {self.metric_type}")

        return result.detach()