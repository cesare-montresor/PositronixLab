import logging
from typing import Any, List

import torch
from torch import nn
from torch.utils.data import DataLoader


from collections import Counter

from tqdm import tqdm
from positronixlab.pipeline import PipelineConfig


logger = logging.getLogger(__name__)


class DeeplearningBase:
    
    def __init__(self, 
                 config: PipelineConfig, 
                 model: nn.Module, 
                 dataloader: DataLoader, 
                 optimizer: torch.optim.Optimizer = None,
                 draft=False,
                ):
        self.config = config
        
        self.model = model
        self.device = model.device
        self.dataloader = dataloader

        self.task = config.task
        self.draft = draft
        
        self.optimizer = optimizer

        self.batch_with_data = False
        self.batch_with_metrics = False

        self.metrics = {}

        self._on_prediction = None
        self._batch_start = None
        self._batch_end = None
        self._epoch_start = None
        self._epoch_end = None



    @property
    def is_training(self):
        return self.optimizer is not None

    
    def add_metric(self, name: str, criterion=None, minimize: bool = True,  backward: bool = False):
        self.metrics[name] = {
            "name": name,
            "best_idx": 0,
            "history": [],
            "minimize": minimize,
            "criterion": criterion,
            "backward": backward
        }
    

    def process_batch(self, epoch: int, epoch_total: int, batch_idx: int, batch_total: int, data_batch: List[torch.Tensor]):  
        callback_data = data_batch if self.batch_with_data else None
        self._batch_did_start(epoch, epoch_total, batch_idx, batch_total, callback_data)
                
        self.model.train(self.is_training)
        
        if len(data_batch) == 2: # standard case: x, y
            x, y = data_batch
        elif len(data_batch) == 3: # data track index for continual learning downstram tasks
            x, y, idx = data_batch


        x = x.to(device=self.device)
        y = y.to(device=self.device)

        
        if self.is_training: self.optimizer.zero_grad()
        y_hat = self.model(x)

        self._did_prediction(epoch, epoch_total, batch_idx, batch_total, data_batch, y_hat)

        metrics = {}
        with torch.set_grad_enabled(self.is_training):
            for name, metric in self.metrics.items():      
                criterion = metric.get("criterion") # LET IT RAISE ERROR IF NOT FOUND
                loss = criterion(x, y, y_hat)

                if self.is_training and metric.get("backward"): 
                    loss.backward()

                metrics[name] = float(loss.data.item())

        if self.is_training:
            self.optimizer.step()

        self._batch_did_end(epoch, epoch_total, batch_idx, batch_total, metrics if self.batch_with_metrics else None)
        return metrics
            

    def process_epoch(self, dataloader: DataLoader, epoch: int, epoch_total: int, draft: int = False):
        draft_batches = 20
        batch_total = len(dataloader)
        batch_cnt = 0
        epoch_metrics = Counter({name: 0.0 for name in self.metrics.keys()})
        batch_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epoch_total}", unit="batch")
                    
        self._epoch_did_start(epoch, epoch_total)
        with torch.set_grad_enabled(self.is_training):
            
            for batch_idx, data_batch in enumerate(batch_bar):        
                batch_metrics = self.process_batch(epoch, epoch_total, batch_idx, batch_total, data_batch)
                for name, value in batch_metrics.items():
                    epoch_metrics.setdefault(name, 0.0)
                    epoch_metrics[name] += value

                batch_cnt += 1
                metrics_txt = " | ".join([f"{name}:{value:.5f}" for name, value in batch_metrics.items()])
                desc = f"E {epoch+1}/{epoch_total} | {metrics_txt}"
                batch_bar.set_description(desc)
                batch_bar.update()

                
                if draft:
                    draft_batches -= 1
                    if draft_batches< 0: 
                        print('Draft train: Early breaking.')
                        draft_batches = 20
                        break


        for name, value in batch_metrics.items():
            epoch_metrics.setdefault(name, 0.0)
            epoch_metrics[name] /= batch_cnt

        for name, value in epoch_metrics.items():
            metric = self.metrics[name]
            history = metric["history"]
            
            best_idx = metric["best_idx"]
            cur_idx = len(history) 
            history.append(value)

            if metric["minimize"]:
                if value < history[best_idx]:
                    metric["best_idx"] = cur_idx
            else:
                if value > history[best_idx]:
                    metric["best_idx"] = cur_idx
        
        self._epoch_did_end(epoch, epoch_total, epoch_metrics)

        return dict(epoch_metrics), batch_idx, batch_total

    def on_prediction(self, callback ):
        self._on_prediction = callback
    
    def on_batch_start(self, callback, with_data=False):
        self._batch_start = callback
        self.batch_with_data = with_data

    def on_batch_end(self, callback, with_metrics=False):
        self._batch_end = callback
        self.batch_with_metrics = with_metrics

    def on_epoch_start(self, callback):
        self._epoch_start = callback

    def on_epoch_end(self, callback):
        self._epoch_end = callback

    def _did_prediction(self, epoch: int, epoch_total: int, batch_idx: int, batch_total: int, batch_data: Any, prediction: torch.Tensor):
        if self._on_prediction is not None:
            self._on_prediction(epoch, epoch_total, batch_idx, batch_total, batch_data, prediction)

    def _batch_did_start(self, epoch: int, epoch_total: int, batch_idx: int, batch_total: int, data: Any = None):
        if self._batch_start is not None:
            self._batch_start(epoch, epoch_total, batch_idx, batch_total, data)

    def _batch_did_end(self, epoch: int, epoch_total: int, batch_idx: int, batch_total: int, metrics: dict = None):
        if self._batch_end is not None:
            self._batch_end(epoch, epoch_total, batch_idx, batch_total, metrics)
    
    def _epoch_did_start(self, epoch: int, epoch_total: int):
        if self._epoch_start is not None:
            self._epoch_start(epoch, epoch_total)

    def _epoch_did_end(self, epoch: int, epoch_total: int, metrics: dict):
        if self._epoch_end is not None:
            self._epoch_end(epoch, epoch_total, metrics)

