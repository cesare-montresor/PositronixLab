import logging
from typing import Any
import time
import numpy as np
from tqdm.asyncio import tqdm
from sklearn.metrics import confusion_matrix

import torch
from torch.utils.data import DataLoader




from positronixlab.pipeline.pipeline_config_v1 import PipelineConfig
from positronixlab.metamodel.metamodel import MetaModel
from positronixlab.dataprovider.dataprovider import DataProvider
from positronixlab.utils.utils import set_best




logger = logging.getLogger(__name__)



class DeeplearningCommonOld:
    def __init__(self, config: PipelineConfig, model: MetaModel, dp: DataProvider, path_save: str=None, draft=False):
        self.model = model
        self.device = model.device
        self.dp = dp
        self.config = config
        self.draft = draft
        self.train_patience = 3
        self.train_early_stop_gain = 0.01
        self.path_save = path_save

        self.batch_with_data = False

        self.metrics = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            # "forward_time": []
        }
        self.metrics_best = {
            "train_loss": float('inf'),
            "train_acc": 0,
            "val_loss": float('inf'),
            "val_acc": 0,
            # "forward_time": float('inf')
        }

        self.callbacks = {}
    
    def on_batch_start(self, callback, with_data=False, inference=False):
        name = 'on_batch_start' + ('_inference' if inference else '')
        self.callbacks[name] = callback
        self.batch_with_data = with_data
      
    def on_batch_end(self, callback, inference=False):
        name = 'on_batch_end' + ('_inference' if inference else '')
        self.callbacks[name] = callback

    def on_epoch_start(self, callback, inference=False):
        name = 'on_epoch_start' + ('_inference' if inference else '')
        self.callbacks[name] = callback

    def on_epoch_end(self, callback, inference=False):
        name = 'on_epoch_end' + ('_inference' if inference else '')
        self.callbacks[name] = callback

    def _batch_did_start(self, epoch: int, epoch_total: int, batch_idx: int, batch_total: int, data: Any = None, inference=False):
        name = 'on_batch_start' + ('_inference' if inference else '')
        if self.callbacks.get(name) is not None:
            self.callbacks[name](epoch, epoch_total, batch_idx, batch_total, data)

    def _batch_did_end(self, epoch: int, epoch_total: int, batch_idx: int, batch_total: int, inference=False):
            name = 'on_batch_end' + ('_inference' if inference else '')
            if self.callbacks.get(name) is not None:
                self.callbacks[name](epoch, epoch_total, batch_idx, batch_total)
    
    def _epoch_did_start(self, epoch: int, epoch_total: int, inference=False):
        name = 'on_epoch_start' + ('_inference' if inference else '')
        if self.callbacks.get(name) is not None:
            self.callbacks[name](epoch, epoch_total)

    def _epoch_did_end(self, epoch: int, epoch_total: int, metrics, inference=False):
        name = 'on_epoch_end' + ('_inference' if inference else '')
        if self.callbacks.get(name) is not None:
            self.callbacks[name](epoch, epoch_total, metrics)

    def train(self, lr=0.001, epoch_total=20, autoencoder=False):  
        
        optim = torch.optim.AdamW(self.model.parameters(), lr=lr)

        draft_batches = 20

        early_stop = EarlyStopping(max_attempts=self.train_patience, min_gain=self.train_early_stop_gain)


        if self.draft: epoch_total = 1

        for epoch in tqdm(range(epoch_total), desc=f"Step TRAIN"):
            self._epoch_did_start(epoch, epoch_total)
            train_loss_total = 0

            train_acc_total = 0
            train_num = 0
            train_loss = 0
            train_acc = 0
            
            batch_total = len(self.dp.train)
            batch_bar = tqdm(self.dp.train)
            for batch_idx, data_batch in enumerate(batch_bar):
                self._batch_did_start(epoch, epoch_total, batch_idx, batch_total, data_batch if self.batch_with_data else None)

                if len(data_batch) == 2: # standard case: x, y
                    x, y = data_batch
                elif len(data_batch) == 3: # data track index for continual learning downstram tasks
                    x, y, idx = data_batch
                
                x = x.to(device=self.device)
                y = y.to(device=self.device)
                optim.zero_grad()
                y_hat = self.model(x)
                #print('.', end='', flush=True)
                loss = self.model.loss(x, y, y_hat)

                
                loss.backward()
                
                optim.step()
                
                #if self.space.batch_queue.qsize() > 500:
                #    self.space.wait_for_queues()
                
                with torch.no_grad():
                    train_loss_total += loss.item()
                    train_num += len(x)
                    if self.config.task == "classification":
                        y_hat_cls = torch.argmax(y_hat,dim=1)
                        train_acc_total += torch.sum(y == y_hat_cls)
                
                train_loss = float(train_loss_total/train_num)
                train_acc =  float(train_acc_total/train_num)

                desc = f"E {epoch+1}/{epoch_total} | L: {train_loss:.5f} | A: {train_acc:.2f}"
                batch_bar.set_description(desc)
                batch_bar.update()

                self._batch_did_end( epoch, train_loss, train_acc)

                if self.draft:
                    draft_batches -= 1
                    if draft_batches< 0: 
                        print('Draft train: Early breaking.')
                        draft_batches = 20
                        break
                
                
                
            print("") #new line after batch bar

            # self.space.observing = False
            val_loss, val_acc, forward_time, _ = self.inference( self.dp.val, autoencoder=autoencoder)
            
            # self.space.observing = True

            self.metrics["train_loss"].append(train_loss)
            self.metrics["train_acc"].append(train_acc)
            self.metrics["val_loss"].append(val_loss)
            self.metrics["val_acc"].append(val_acc)
            # self.metrics["forward_time"].append(forward_time)

            set_best(self.metrics_best, "train_loss", train_loss)
            set_best(self.metrics_best, "train_acc", train_acc, min=False)
            set_best(self.metrics_best, "val_loss", val_loss)
            set_best(self.metrics_best, "val_acc", val_acc, min=False)

            
            print(  f"\n[{epoch+1}/{epoch_total} | Train L:{train_loss:.5f} A:{train_acc:.2f} | Val L:{val_loss:.5f} A:{val_acc:.2f} | T: {forward_time:.2f}s ]")
            

            metrics = {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
            
            self._epoch_did_end(epoch, epoch_total, metrics)

            should_stop = early_stop.should_stop(val_acc)
            if should_stop: break

            
            
        if self.path_save is not None:
            self.model.save()


    def inference(self, dataloader:DataLoader, compute_cm=False, cluster_class=False, autoencoder = False, force_eval=True) -> tuple[float, float, float, np.ndarray|None]:
        print("STEP INFERENCE:\n")
        bs, bn, dn = dataloader.batch_size, len(dataloader), len(dataloader.dataset)
        print(f"DataLoader: batch_size: {bs}, num_batches:{bn}, num_data:{dn}")
        if compute_cm: print("compute_cm;", end=' ')
        if cluster_class: print("cluster_class;", end=' ')
        if autoencoder: print("autoencoder;", end=' ')
        if force_eval: print("force_eval;", end=' ')
        print("")

        val_acc = 0
        predicted_class = []
        real_class = []

        total_loss = 0
        data_num = 0
        forward_time = time.time()
        
        
        draft_batches = 20

        #old_queue_state = self.enable_loss_queue
        y_batch_queue = []  

        if cluster_class:
            print("Preparing class_cluster_attribution...")
            #self.space.wait_for_queues()
            #self.space.enable_loss_queue = True
            #if self.space.last_clustering_all is None:
            #    raise ValueError("No clustering available for class_cluster_attribution")
            #
            #clusters_label = torch.unique(self.space.last_clustering_all)
            #self.space.last_cluster_stats = {lbl.item():{} for lbl in clusters_label}

        was_training = self.model.training
        if force_eval and was_training:
            self.model.eval()

        with torch.no_grad():
            for x, y in tqdm(dataloader, desc="Step INFERENCE"): 
                x = x.to(device=self.device)
                y = y.to(device=self.device)
                
                y_hat = self.model(x)
                
                loss = self.model.loss(x, y, y_hat)

                total_loss += loss.detach().item()
                data_num += len(x)

                if self.config.task == "classification":
                    y_hat_cls = torch.argmax(y_hat, dim=1)
                    if cluster_class:
                        y_batch_queue.append( (y, y_hat_cls) )
                        #space_loss = self.space.loss_queue.get()
                        #self.space.class_cluster_attribution(space_loss, y, y_hat_cls)

                    val_acc += torch.sum(y == y_hat_cls)
                    if compute_cm:
                        predicted_class.extend(y_hat_cls.tolist())
                        real_class.extend(y.tolist())

                if self.draft:
                    draft_batches -= 1
                    if draft_batches< 0: 
                        print('Draft validation: Early breaking.')
                        draft_batches = 20
                        break

        forward_time = time.time() - forward_time

        cm = None
        if not autoencoder and compute_cm:
            cm = confusion_matrix(real_class, predicted_class)
            #print("Confusion Matrix:\n", cm)

        total_loss = float(total_loss/data_num)
        val_acc = float(val_acc/data_num)
        forward_time = float(forward_time/data_num)


        #if cluster_class:
        #    self.space.enable_loss_queue = old_queue_state



        if force_eval and was_training:
            self.model.train()

        return total_loss, val_acc, forward_time, cm




# class DeepLearningCommon:
#     """
#     Orchestrates train / validate / inference on top of DeeplearningBase.

#     - train() and validate() delegate batching/looping to DeeplearningBase
#       (self.trainer / self.validator), which also carry the metric criteria
#       and batch/epoch callbacks.
#     - inference() is a lean standalone loop (no optimizer, no metric criteria
#       required) used for final evaluation: accuracy, confusion matrix, timing.
#     """

#     def __init__(self, config: PipelineConfig, model: nn.Module, dataprovider: DataProvider, draft: bool = False):
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




    


class DeeplearningCommonOld1:
    def __init__(self, config: PipelineConfig, model: MetaModel, dp: DataProvider, path_save: str=None, draft=False):
        self.model = model
        self.device = model.device
        self.dp = dp
        self.config = config
        self.draft = draft
        self.train_patience = 3
        self.train_early_stop_gain = 0.01
        self.path_save = path_save

        self.batch_with_data = False

        self.metrics = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            # "forward_time": []
        }
        self.metrics_best = {
            "train_loss": float('inf'),
            "train_acc": 0,
            "val_loss": float('inf'),
            "val_acc": 0,
            # "forward_time": float('inf')
        }

        self.callbacks = {}
    
    def on_batch_start(self, callback, with_data=False, inference=False):
        name = 'on_batch_start' + ('_inference' if inference else '')
        self.callbacks[name] = callback
        self.batch_with_data = with_data
      
    def on_batch_end(self, callback, inference=False):
        name = 'on_batch_end' + ('_inference' if inference else '')
        self.callbacks[name] = callback

    def on_epoch_start(self, callback, inference=False):
        name = 'on_epoch_start' + ('_inference' if inference else '')
        self.callbacks[name] = callback

    def on_epoch_end(self, callback, inference=False):
        name = 'on_epoch_end' + ('_inference' if inference else '')
        self.callbacks[name] = callback

    def _batch_did_start(self, epoch: int, epoch_total: int, batch_idx: int, batch_total: int, data: Any = None, inference=False):
        name = 'on_batch_start' + ('_inference' if inference else '')
        if self.callbacks.get(name) is not None:
            self.callbacks[name](epoch, epoch_total, batch_idx, batch_total, data)

    def _batch_did_end(self, epoch: int, epoch_total: int, batch_idx: int, batch_total: int, inference=False):
            name = 'on_batch_end' + ('_inference' if inference else '')
            if self.callbacks.get(name) is not None:
                self.callbacks[name](epoch, epoch_total, batch_idx, batch_total)
    
    def _epoch_did_start(self, epoch: int, epoch_total: int, inference=False):
        name = 'on_epoch_start' + ('_inference' if inference else '')
        if self.callbacks.get(name) is not None:
            self.callbacks[name](epoch, epoch_total)

    def _epoch_did_end(self, epoch: int, epoch_total: int, metrics, inference=False):
        name = 'on_epoch_end' + ('_inference' if inference else '')
        if self.callbacks.get(name) is not None:
            self.callbacks[name](epoch, epoch_total, metrics)

    def train(self, lr=0.001, epoch_total=20, autoencoder=False):  
        
        optim = torch.optim.AdamW(self.model.parameters(), lr=lr)

        draft_batches = 20

        early_stop = EarlyStopping(max_attempts=self.train_patience, min_gain=self.train_early_stop_gain)


        if self.draft: epoch_total = 1

        for epoch in tqdm(range(epoch_total), desc=f"Step TRAIN"):
            self._epoch_did_start(epoch, epoch_total)
            train_loss_total = 0

            train_acc_total = 0
            train_num = 0
            train_loss = 0
            train_acc = 0
            
            batch_total = len(self.dp.train)
            batch_bar = tqdm(self.dp.train)
            for batch_idx, data_batch in enumerate(batch_bar):
                self._batch_did_start(epoch, epoch_total, batch_idx, batch_total, data_batch if self.batch_with_data else None)

                if len(data_batch) == 2: # standard case: x, y
                    x, y = data_batch
                elif len(data_batch) == 3: # data track index for continual learning downstram tasks
                    x, y, idx = data_batch
                
                x = x.to(device=self.device)
                y = y.to(device=self.device)
                optim.zero_grad()
                y_hat = self.model(x)
                #print('.', end='', flush=True)
                loss = self.model.loss(x, y, y_hat)

                
                loss.backward()
                
                optim.step()
                
                #if self.space.batch_queue.qsize() > 500:
                #    self.space.wait_for_queues()
                
                with torch.no_grad():
                    train_loss_total += loss.item()
                    train_num += len(x)
                    if self.config.task == "classification":
                        y_hat_cls = torch.argmax(y_hat,dim=1)
                        train_acc_total += torch.sum(y == y_hat_cls)
                
                train_loss = float(train_loss_total/train_num)
                train_acc =  float(train_acc_total/train_num)

                desc = f"E {epoch+1}/{epoch_total} | L: {train_loss:.5f} | A: {train_acc:.2f}"
                batch_bar.set_description(desc)
                batch_bar.update()

                self._batch_did_end( epoch, train_loss, train_acc)

                if self.draft:
                    draft_batches -= 1
                    if draft_batches< 0: 
                        print('Draft train: Early breaking.')
                        draft_batches = 20
                        break
                
                
                
            print("") #new line after batch bar

            # self.space.observing = False
            val_loss, val_acc, forward_time, _ = self.inference( self.dp.val, autoencoder=autoencoder)
            
            # self.space.observing = True

            self.metrics["train_loss"].append(train_loss)
            self.metrics["train_acc"].append(train_acc)
            self.metrics["val_loss"].append(val_loss)
            self.metrics["val_acc"].append(val_acc)
            # self.metrics["forward_time"].append(forward_time)

            set_best(self.metrics_best, "train_loss", train_loss)
            set_best(self.metrics_best, "train_acc", train_acc, min=False)
            set_best(self.metrics_best, "val_loss", val_loss)
            set_best(self.metrics_best, "val_acc", val_acc, min=False)

            
            print(  f"\n[{epoch+1}/{epoch_total} | Train L:{train_loss:.5f} A:{train_acc:.2f} | Val L:{val_loss:.5f} A:{val_acc:.2f} | T: {forward_time:.2f}s ]")
            

            metrics = {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
            
            self._epoch_did_end(epoch, epoch_total, metrics)

            should_stop = early_stop.should_stop(val_acc)
            if should_stop: break

            
            
        if self.path_save is not None:
            self.model.save()


    def inference(self, dataloader:DataLoader, compute_cm=False, cluster_class=False, autoencoder = False, force_eval=True) -> tuple[float, float, float, np.ndarray|None]:
        print("STEP INFERENCE:\n")
        bs, bn, dn = dataloader.batch_size, len(dataloader), len(dataloader.dataset)
        print(f"DataLoader: batch_size: {bs}, num_batches:{bn}, num_data:{dn}")
        if compute_cm: print("compute_cm;", end=' ')
        if cluster_class: print("cluster_class;", end=' ')
        if autoencoder: print("autoencoder;", end=' ')
        if force_eval: print("force_eval;", end=' ')
        print("")

        val_acc = 0
        predicted_class = []
        real_class = []

        total_loss = 0
        data_num = 0
        forward_time = time.time()
        
        
        draft_batches = 20

        #old_queue_state = self.enable_loss_queue
        y_batch_queue = []  

        if cluster_class:
            print("Preparing class_cluster_attribution...")
            #self.space.wait_for_queues()
            #self.space.enable_loss_queue = True
            #if self.space.last_clustering_all is None:
            #    raise ValueError("No clustering available for class_cluster_attribution")
            #
            #clusters_label = torch.unique(self.space.last_clustering_all)
            #self.space.last_cluster_stats = {lbl.item():{} for lbl in clusters_label}

        was_training = self.model.training
        if force_eval and was_training:
            self.model.eval()

        with torch.no_grad():
            for x, y in tqdm(dataloader, desc="Step INFERENCE"): 
                x = x.to(device=self.device)
                y = y.to(device=self.device)
                
                y_hat = self.model(x)
                
                loss = self.model.loss(x, y, y_hat)

                total_loss += loss.detach().item()
                data_num += len(x)

                if self.config.task == "classification":
                    y_hat_cls = torch.argmax(y_hat, dim=1)
                    if cluster_class:
                        y_batch_queue.append( (y, y_hat_cls) )
                        #space_loss = self.space.loss_queue.get()
                        #self.space.class_cluster_attribution(space_loss, y, y_hat_cls)

                    val_acc += torch.sum(y == y_hat_cls)
                    if compute_cm:
                        predicted_class.extend(y_hat_cls.tolist())
                        real_class.extend(y.tolist())

                if self.draft:
                    draft_batches -= 1
                    if draft_batches< 0: 
                        print('Draft validation: Early breaking.')
                        draft_batches = 20
                        break

        forward_time = time.time() - forward_time

        cm = None
        if not autoencoder and compute_cm:
            cm = confusion_matrix(real_class, predicted_class)
            #print("Confusion Matrix:\n", cm)

        total_loss = float(total_loss/data_num)
        val_acc = float(val_acc/data_num)
        forward_time = float(forward_time/data_num)


        #if cluster_class:
        #    self.space.enable_loss_queue = old_queue_state



        if force_eval and was_training:
            self.model.train()

        return total_loss, val_acc, forward_time, cm



