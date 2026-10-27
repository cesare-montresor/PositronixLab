import logging
from typing import Any, Callable, List

import torch
from torch import nn
from torch.utils.data import DataLoader
import time
import numpy as np
from sklearn.metrics import confusion_matrix

from collections import Counter

from tqdm import tqdm
from positronixlab.dataprovider.dataprovider import DataProvider
from positronixlab.trial.trial_config import TrialConfig


logger = logging.getLogger(__name__)


class DeeplearningBase:
    
    def __init__(self, 
                 config: TrialConfig, 
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
        self._batch_did_start(epoch, epoch_total, batch_idx, batch_total)

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
    
    def on_batch_start(self, callback):
        self._batch_start = callback
        
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



# class DeepLearningCommon:
#     """
#     Orchestrates train / validate / inference on top of DeeplearningBase.

#     - train() and validate() delegate batching/looping to DeeplearningBase
#       (self.trainer / self.validator), which also carry the metric criteria
#       and batch/epoch callbacks.
#     - inference() is a lean standalone loop (no optimizer, no metric criteria
#       required) used for final evaluation: accuracy, confusion matrix, timing.
#     """

#     def __init__(self, config: TrialConfig, model: nn.Module, dataprovider: DataProvider, draft: bool = False):
#         self.config = config
#         self.model = model
#         self.device = model.device
#         self.dp = dataprovider
#         self.draft = draft

#         self.optim = None
#         self.optim = self._get_optimizer()

#         self.early_stop: EarlyStopping = None
#         self.early_stop_metric: str = None  # key in val metrics dict used to drive early stopping

#         self.trainer = DeeplearningBase(self.config, self.model, self.dp.train, optimizer=self.optim, draft=self.draft)
#         self.validator = DeeplearningBase(self.config, self.model, self.dp.val, optimizer=None, draft=self.draft)

#         self.history = {"train": [], "val": []}

#     def add_early_stopping(self, max_attempts: int = 3, min_gain: float = 0.01, metric: str = "loss"):
#         self.early_stop = EarlyStopping(max_attempts=max_attempts, min_gain=min_gain)
#         self.early_stop_metric = metric

#     def _get_optimizer(self):
#         if self.optim is not None:
#             return self.optim
#         if self.config.optimizer == "adamw":
#             self.optim = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
#         elif self.config.optimizer == "sgd":
#             self.optim = torch.optim.SGD(self.model.parameters(), lr=self.config.learning_rate)
#         elif self.config.optimizer == "adam":
#             self.optim = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
#         else:
#             raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

#         return self.optim

#     def add_metric(self, name: str, minimize: bool = True, criterion=None, backward: bool = False):
#         self.trainer.add_metric(name, minimize=minimize, criterion=criterion, backward=backward)
#         self.validator.add_metric(name, minimize=minimize, criterion=criterion, backward=backward)

#     # ---- callbacks: forwarded to trainer / validator, or both ----

#     def on_prediction(self, callback: Callable, phase: str = "both"):
#         if phase in ("train", "both"):
#             self.trainer.on_prediction(callback)
#         if phase in ("val", "both"):
#             self.validator.on_prediction(callback)

#     def on_batch_start(self, callback: Callable, phase: str = "both"):
#         if phase in ("train", "both"):
#             self.trainer.on_batch_start(callback)
#         if phase in ("val", "both"):
#             self.validator.on_batch_start(callback)

#     def on_batch_end(self, callback: Callable, with_metrics: bool = False, phase: str = "both"):
#         if phase in ("train", "both"):
#             self.trainer.on_batch_end(callback, with_metrics=with_metrics)
#         if phase in ("val", "both"):
#             self.validator.on_batch_end(callback, with_metrics=with_metrics)

#     def on_epoch_start(self, callback: Callable, phase: str = "both"):
#         if phase in ("train", "both"):
#             self.trainer.on_epoch_start(callback)
#         if phase in ("val", "both"):
#             self.validator.on_epoch_start(callback)

#     def on_epoch_end(self, callback: Callable, phase: str = "both"):
#         if phase in ("train", "both"):
#             self.trainer.on_epoch_end(callback)
#         if phase in ("val", "both"):
#             self.validator.on_epoch_end(callback)

#     # ---- phases ----


#     def sanity_check(self, must_backward: bool = True):
#         backward_found = not must_backward
#         for metric in self.trainer.metrics.values():
#             if metric.get("backward", False):
#                 backward_found = True
#                 break

#         if not backward_found:
#             raise ValueError("No metric with backward=True found. Please add a metric with backward=True to perform training.")
#         if len(self.trainer.metrics) == 0:
#             raise ValueError("No metrics found. Please add at least one metric to perform training.")

        
#     def train(self, epoch_total: int) -> dict:
#         if self.draft: epoch_total = 1
#         self.sanity_check()
        
#         for epoch in range(epoch_total):
#             train_metrics, *_ = self.trainer.process_epoch(self.dp.train, epoch, epoch_total, draft=self.draft)
#             val_metrics = self.validate(epoch, epoch_total)

#             self.history["train"].append(train_metrics)
#             self.history["val"].append(val_metrics)

#             metrics_txt = " | ".join(f"{k}:{v:.5f}" for k, v in val_metrics.items())
#             print(f"\n[{epoch + 1}/{epoch_total}] Val {metrics_txt}")

#             if self.early_stop is not None and self.early_stop_metric in val_metrics:
#                 if self.early_stop.should_stop(val_metrics[self.early_stop_metric]):
#                     print("[EarlyStop] Training stopped.")
#                     break

#         return self.history

#     def validate(self, epoch: int = 0, epoch_total: int = 1) -> dict:
#         self.sanity_check(must_backward=False)
#         val_metrics, *_ = self.validator.process_epoch(self.dp.val, epoch, epoch_total, draft=self.draft)
#         return val_metrics

#     def inference(self, dataloader: DataLoader = None, compute_cm: bool = False, force_eval: bool = True,
#                   on_batch_end: Callable = None) -> tuple[float, float, float, np.ndarray | None]:
#         """
#         Standalone evaluation loop (no optimizer, no registered metric criteria required).
#         Returns (avg_loss, accuracy, avg_forward_time_per_sample, confusion_matrix|None).
#         on_batch_end(batch_idx, batch_total, loss, acc) fires after every batch if provided.
#         """
#         dataloader = dataloader if dataloader is not None else self.dp.val

#         was_training = self.model.training
#         if force_eval and was_training:
#             self.model.eval()

#         total_loss, correct, data_num = 0.0, 0, 0
#         predicted_class, real_class = [], []

#         draft_batches = 20
#         forward_time = time.time()

#         with torch.no_grad():
#             for batch_idx, (x, y) in enumerate(tqdm(dataloader, desc="Inference")):
#                 x = x.to(device=self.device)
#                 y = y.to(device=self.device)

#                 y_hat = self.model(x)
#                 loss = self.model.loss(x, y, y_hat)

#                 total_loss += loss.detach().item()
#                 data_num += len(x)

#                 batch_acc = None
#                 if self.config.task == "classification":
#                     y_hat_cls = torch.argmax(y_hat, dim=1)
#                     batch_correct = torch.sum(y == y_hat_cls).item()
#                     correct += batch_correct
#                     batch_acc = batch_correct / len(x)

#                     if compute_cm:
#                         predicted_class.extend(y_hat_cls.tolist())
#                         real_class.extend(y.tolist())

#                 if on_batch_end is not None:
#                     on_batch_end(batch_idx, len(dataloader), float(loss.item()), batch_acc)

#                 if self.draft:
#                     draft_batches -= 1
#                     if draft_batches < 0:
#                         print('Draft inference: Early breaking.')
#                         break

#         forward_time = time.time() - forward_time

#         cm = None
#         if compute_cm and self.config.task == "classification":
#             cm = confusion_matrix(real_class, predicted_class)

#         total_loss = float(total_loss / data_num)
#         val_acc = float(correct / data_num) if self.config.task == "classification" else None
#         forward_time = float(forward_time / data_num)

#         if force_eval and was_training:
#             self.model.train()

#         return total_loss, val_acc, forward_time, cm




    
class EarlyStopping:
    def __init__(self, max_attempts: int = 3, min_gain: float = 0.01):
        self.max_attempts = max_attempts
        self.min_gain = min_gain
        self.num_attempts = 0
        self.metric_last = 0
        self.total_attempts = 0
        self.num_resets = 0

    def should_stop(self, metric):
        self.total_attempts += 1
        gain = metric - self.metric_last
        
        self.num_attempts += 1
        

        if gain >= self.min_gain:
            self.num_resets += 1
            self.num_attempts = 0 #valid gain: reset counter
            self.metric_last = metric
            logging.info(f'[EarlyStop] New best! +{int(gain*100)}% metric')
        else:
            logging.info(f'[EarlyStop] No sufficient gain: +{int(gain*100)}%  ({self.num_attempts}/{self.max_attempts})')
        
        if self.num_attempts > self.max_attempts:
            logging.info(f'[EarlyStop] No gain. Stopping.')
            return True

        return False