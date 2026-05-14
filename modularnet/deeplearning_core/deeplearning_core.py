import torch
from torch.utils.data import DataLoader
import time
import numpy as np
from sklearn.metrics import confusion_matrix



from tqdm.asyncio import tqdm
from modularnet.common.utils import set_best
from modularnet.config.trial_config import TrialConfig
from modularnet.dataprovider.dataprovider import DataProvider
from modularnet.metamodel.metamodel import MetaModel
from modularnet.modularspace.modularspace import ModularSpace

from typing_extensions import override
import torch
from typing import Literal, Optional

from modularnet.modularcluster.modularcluster_kmeans import ModularClusterKmeans
from modularnet.plugin.plugin import Plugin


class DeeplearningCore:
    def __init__(self, config: TrialConfig, model: MetaModel, dp: DataProvider, space: ModularSpace, draft=False):
        self.model = model
        self.dp = dp
        self.space = space
        self.config = config
        self.draft = draft

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


    def setup( self, name: str, num_workers_data: int = 0, debug: bool = False,):
                
        # Create model first to determine device
        model = MetaModel(name, self.config, device=device)

        if debug: model.print()

        device  = model.device
        # Pass model device to dataset
        dataset = DataProvider.from_config(self.config, device=device, num_workers=num_workers_data)

        return model, dataset




    def train(self, lr=0.001, epochs=20, autoencoder=False):  
        
        optim = torch.optim.AdamW(self.model.parameters(), lr=lr)

        draft_batches = 20

        early_stop_cnt = 0
        val_acc_old = 0

        if self.draft: epochs = 1

        for epoch_num in tqdm(range(epochs), desc=f"Step TRAIN"):
            train_loss = 0

            train_acc = 0
            train_num = 0
            avg_train_loss = 0
            avg_train_acc = 0
            
            
            batch_bar = tqdm(self.dp.train)
            for x, y in batch_bar:
                x = x.to(device=self.dp.device)
                y = y.to(device=self.dp.device)
                optim.zero_grad()
                y_hat = self.model(x)
                #print('.', end='', flush=True)
                loss = self.model.loss(x, y, y_hat)

                
                loss.backward()
                
                optim.step()
                
                #if self.space.batch_queue.qsize() > 500:
                #    self.space.wait_for_queues()
                
                with torch.no_grad():
                    train_loss += loss.detach().item()
                    train_num += len(x)
                    if self.config.task == "classification":
                        y_hat_cls = torch.argmax(y_hat,dim=1)
                        train_acc += torch.sum(y == y_hat_cls)
                
                avg_train_loss = float(train_loss/train_num)
                avg_train_acc =  float(train_acc/train_num)

                desc = f"E {epoch_num+1}/{epochs} | L: {avg_train_loss:.5f} | A: {avg_train_acc:.2f}"
                batch_bar.set_description(desc)
                batch_bar.update()
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

            self.metrics["train_loss"].append(avg_train_loss)
            self.metrics["train_acc"].append(avg_train_acc)
            self.metrics["val_loss"].append(val_loss)
            self.metrics["val_acc"].append(val_acc)
            # self.metrics["forward_time"].append(forward_time)

            set_best(self.metrics_best, "train_loss", avg_train_loss)
            set_best(self.metrics_best, "train_acc", avg_train_acc, min=False)
            set_best(self.metrics_best, "val_loss", val_loss)
            set_best(self.metrics_best, "val_acc", val_acc, min=False)

            
            print(  f"\n[{epoch_num+1}/{epochs} | Train L:{avg_train_loss:.5f} A:{avg_train_acc:.2f} | Val L:{val_loss:.5f} A:{val_acc:.2f} | T: {forward_time:.2f}s ]")
            
            self.space.rescale_factor_step()
            self.space.save_space()

            # Early stopping
            gain = val_acc - val_acc_old
            if self.train_patience > 0: # is active
                early_stop_cnt += 1
                if gain >= self.train_early_stop_gain:
                    early_stop_cnt = 0 #valid gain: reset counter
                    val_acc_old = val_acc
                    print(f'[EarlyStop] New best! +{int(gain*100)}% val_acc')
                else:
                    print(f'[EarlyStop] No sufficient gain: +{int(gain*100)}%  ({early_stop_cnt}/{self.train_patience})')
                
                if early_stop_cnt > self.train_patience:
                    print(f'[EarlyStop] No gain. Stopping.')
                    break

            
                
            
            self.space.wait_for_queues()



            

        self.model.save()
        #vis.play()

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

        old_queue_state = self.space.enable_loss_queue
        y_batch_queue = []  

        if cluster_class:
            print("Preparing class_cluster_attribution...")
            self.space.wait_for_queues()
            self.space.enable_loss_queue = True
            if self.space.last_clustering_all is None:
                raise ValueError("No clustering available for class_cluster_attribution")

            clusters_label = torch.unique(self.space.last_clustering_all)
            self.space.last_cluster_stats = {lbl.item():{} for lbl in clusters_label}

        was_training = self.model.training
        if force_eval and was_training:
            self.model.eval()

        with torch.no_grad():
            for x, y in tqdm(dataloader, desc="Step INFERENCE"): 
                x = x.to(device=self.dp.device)
                y = y.to(device=self.dp.device)
                
                y_hat = self.model(x)
                
                loss = self.model.loss(x, y, y_hat)

                total_loss += loss.detach().item()
                data_num += len(x)

                if self.config.task == "classification":
                    y_hat_cls = torch.argmax(y_hat, dim=1)
                    if cluster_class:
                        y_batch_queue.append( (y, y_hat_cls) )
                        space_loss = self.space.loss_queue.get()
                        self.space.class_cluster_attribution(space_loss, y, y_hat_cls)

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


        if cluster_class:
            self.space.enable_loss_queue = old_queue_state



        if force_eval and was_training:
            self.model.train()

        return total_loss, val_acc, forward_time, cm