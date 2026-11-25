import logging
from typing import Callable

from torch.utils.data import DataLoader
from torch import nn


from positronixlab.deeplearning.deeplearning_base import DeeplearningBase

from positronixlab.pipeline import PipelineConfig


logger = logging.getLogger(__name__)



class DeeplearningEval:

    def __init__(self, config: PipelineConfig, model: nn.Module, dl: DataLoader, draft: bool = False):
        self.config = config
        self.model = model
        self.device = model.device
        self.dl = dl
        self.draft = draft
        

        self.evaluator = DeeplearningBase(self.config, self.model, self.dl, draft=self.draft)

        self.history = []
    

    def add_metric(self, name: str,  criterion=None, minimize: bool = True):
        self.evaluator.add_metric(name, criterion=criterion, minimize=minimize)

    # ---- callbacks: forwarded to trainer / validator, or both ----

    def on_prediction(self, callback: Callable):
        self.evaluator.on_prediction(callback)

    def on_batch_start(self, callback: Callable, with_data: bool = False):
        self.evaluator.on_batch_start(callback, with_data=with_data)

    def on_batch_end(self, callback: Callable, with_metrics: bool = False):
        self.evaluator.on_batch_end(callback, with_metrics=with_metrics)

    def on_epoch_start(self, callback: Callable):
        self.evaluator.on_epoch_start(callback)

    def on_epoch_end(self, callback: Callable):
        self.evaluator.on_epoch_end(callback)


    def sanity_check(self):
        if len(self.evaluator.metrics) == 0:
            raise ValueError("No metrics found. Please add at least one metric to perform evaluation.")

        
    def evaluate(self, epoch_total: int) -> dict:
        if self.draft: epoch_total = 1
        self.sanity_check()
        
        for epoch in range(epoch_total):
            eval_metrics, *_ = self.evaluator.process_epoch(self.dl, epoch, epoch_total, draft=self.draft)
            
            self.history.append(eval_metrics)
            
            metrics_txt = " | ".join(f"{k}:{v:.5f}" for k, v in eval_metrics.items())
            print(f"\n[{epoch + 1}/{epoch_total}] Eval {metrics_txt}")


        return self.history

    
