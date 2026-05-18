import json

import time
from queue import Empty, Queue
import threading

import tqdm
from modularnet..trial_config import TrialConfig
from modularnet.spaceloss.spaceloss import SpaceLoss

import pandas as pd
import torch
from torch import nn
import torch.multiprocessing as mp
from torchgen.context import F

# Set spawn method for CUDA compatibility (if not already set)
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set by pipeline or other module

from collections import OrderedDict
from typing import *
import os
import numpy as np
from scipy.stats import entropy


from modularnet.modularobserver.modularobserver import ModularObserver



class ModularSpace():
    def __init__(self, config: TrialConfig, name:str, module: nn.Module, space_size:int, lr:float=0.1, history:bool=True, debug=False, useLayers=True, useActivations=True, device=None, base_path=None, num_workers=None, autostart=True, modularizer=None):
        self.name = name
        self.config = config
        

        self.module = module
        self.space_size = space_size
        self.debug = debug 

        self.best_cluster_score = None

        
        torch.autograd.set_detect_anomaly(self.debug)


        
        # Get device from module if not specified, ensuring consistency
        if device is not None:
            self.device = torch.device(device)
        else:
            # Try to infer device from the module's parameters
            try:
                model_device = next(module.parameters()).device
                self.device = model_device
            except StopIteration:
                # If module has no parameters, use default device
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.lr = config.modular_space_lr if lr is None else lr
        self.lr_steps = [10000] # [1000,2000,3000]
        self.observing = True
        self.modularizer = True if modularizer is None else modularizer
        self.mos = [] #type: List[ModularObserver]
        self.rescale_gamma = config.modular_modularize_decay
        self.rescale_factor = self.rescale_gamma
        
        self.num_workers = 4 if num_workers is None else num_workers
        self.base_path = './' if base_path is None else base_path
        

        self.space_loss = SpaceLoss(self)
        self.last_space_loss = self.space_loss

        self.mos_units_count = [mo.num_units for mo in self.mos]
        total_units = sum(self.mos_units_count)
        # Allocate one big tensor for all positions
        self.positions_all = torch.randn(total_units, self.space_size, device=self.device, requires_grad=True)
        self.positions_all_last = self.positions_all.detach().clone().cpu()
        
        # Create views over the flat positions tensor for each ModularObserver
        offset = 0
        self.positions = []
        for num_units in self.mos_units_count:
            view = self.positions_all[offset:offset + num_units]
            self.positions.append(view)
            offset += num_units

        self.print_cooldown = 3.0
        self.print_next_time = time.time()

        self.optim = torch.optim.Adam([self.positions_all], lr=self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=self.lr_steps, gamma=0.5)

        self.history = history

        self.activation_queue = Queue()
        # Use threading instead of multiprocessing for GPU tensor sharing
        self.main_thread = None
        self.worker_threads = []
        self.last_batch = None
        self.batch_queue = Queue()
        self.loss_dict = OrderedDict() # type: Dict[str, SpaceLoss]
        self.loss_dict_mutex = threading.Condition( threading.Lock() )

        self.enable_loss_queue = False
        self.loss_queue = Queue()

        self.running = False
        self.optimizing = True
        
        self.batch_ready = threading.Event()
        self.batch_ready.clear()

        self.module.register_forward_pre_hook(self.on_forward)
        # self.module.register_full_backward_hook(self.on_backward)


        self.monitor_thread = None

        self.last_cluster_centers, self.last_clustering_all, self.last_clustering = None, None, None
        self.last_cluster_stats, self.last_cluster_stats_wrong = None, None

        if autostart: self.start()


    def step_space(self, epochs=1, save_history=True):
        print("STEP SPACE")
        #self.model.load()
        self.space.load_space()


        self.model.eval()

        num_batches = len(self.dp.train)
        num = 0 
        for epoch in tqdm(range(epochs) , desc="Step SPACE epoch"):
            for x, y in tqdm(self.dp.train, total=num_batches, desc="Step SPACE batch"):
                x = x.to(device=self.dp.device)
                y = y.to(device=self.dp.device)
                self.model(x)

                #centers_on, centers_off, losses = space.last_centers()
                #self.space.step_batch() #centers_on, centers_off, losses)
                self.space.wait_for_space()

                if save_history: self.space.add_history()

                num += 1

        self.space.save_space()


    def on_forward(self, module, input):
        if not self.observing: return
        last_batch = str(time.time())

        #lock for thread safety
        self.loss_dict_mutex.acquire()
        self.last_batch = last_batch
        self.loss_dict[last_batch] = SpaceLoss(self, batch_id=last_batch)
        self.loss_dict_mutex.release()
        

    def wait_for_batch(self):
        #TODO: block thread until batch is processed
        pass
    
    def wait_for_queues(self):
        #TODO: block thread until all queues are empty
        print("Waiting for queues to empty...") 
        while not self.batch_queue.empty() or not self.activation_queue.empty():
            time.sleep(0.1)
        
    
    def monitor_space_queues(self, start=True):
        if start:
            self.monitor_thread = threading.Thread(target=self.monitor_space_queues, args=(False,) , daemon=True)
            self.monitor_thread.start()
            return self.monitor_thread

        last_empty = False
        while self.running:
            time.sleep(1.0)
            bq, aq, ld = self.batch_queue.qsize(), self.activation_queue.qsize(), len(self.loss_dict)
            lq = self.loss_queue.qsize() if self.loss_queue is not None else -1

            is_empty = (bq + aq + ld + lq) == 0
            if is_empty and last_empty:
                continue
            
            print('QUEUES:', end=' ' if is_empty else '\n')
            if bq > 0: print('- batch_queue:', bq)
            if aq > 0: print('- activation_queue:', aq )
            if ld > 0: print('- loss_dict:', ld )
            if lq != 0: print('- loss_queue:', lq )
            if is_empty: print(' All queues empty.')
            last_empty = is_empty

    def start(self):
        self.reset()
        self.observing = True
        self.optimizing = True
        self.running = True

        self.main_thread = threading.Thread(target=self.space_main, daemon=True)
        self.main_thread.start()
        self.start_workers()    
        self.monitor_thread = self.monitor_space_queues()
        
    def start_workers(self):
        for _ in range(self.num_workers):
            worker = threading.Thread(target=self.space_worker, daemon=True)
            worker.start()
            self.worker_threads.append(worker)
        
    def stop(self):
        self.running = False
        if self.main_thread is not None:
            # Put a sentinel value to unblock the queue
            self.main_thread.join(timeout=2.0)
            self.main_thread = None

        for worker_thread in self.worker_threads:
            worker_thread.join(timeout=2.0)
        self.worker_threads = []

        if self.monitor_thread is not None:
            self.monitor_thread.join(timeout=2.0)
            self.monitor_thread = None

    def space_worker(self):
        
        while self.running:
            batch_id, mos_id, x = self.activation_queue.get()
            while self.running:
                #print('space_worker: activation_queue:', self.activation_queue.qsize() )
                #lock for thread safety
                batch_loss = self.loss_dict.get(batch_id, None)
                if batch_loss is None:
                    time.sleep(0.01)
                    continue
                
                batch_loss.activations_layer(mos_id, x)
                self.is_batch_ready(batch_id, batch_loss)
                break
                
            

    def is_batch_ready(self, batch_id:str, batch_loss:SpaceLoss):
        
        if not batch_loss.can_forward(): 
            return False
        
        self.loss_dict_mutex.acquire()
        batch_loss = self.loss_dict.pop(batch_id, None)
        self.loss_dict_mutex.release()
        
        if batch_loss is None: 
            return False
        
        self.batch_queue.put(batch_loss)
        
        return True

    def get_queue_batch(self) -> SpaceLoss|None:
        try:
            batch_loss = self.batch_queue.get(block=False)
        except Empty:
            batch_loss = None
        return batch_loss


    def space_main(self):
        

        while self.running:
            batch_loss = self.get_queue_batch()
            
            if batch_loss is None:
                time.sleep(0.00001)
                continue
            
            # --- ADDED LINE ---
            try:
                self.step_batch(batch_loss)
                if self.optimizing:
                    self.positions_all_last = self.positions_all.detach().clone().cpu()
                
                if self.enable_loss_queue:
                    self.loss_queue.put(batch_loss)
                    #print("LOSS Q: queue size:", self.loss_queue.qsize())
                
                self.space_loss = batch_loss
                self.last_space_loss = batch_loss
            except Exception as e:
                print(f"Exception in main thread: {e}")
                import traceback
                traceback.print_exc()
                raise e
                
            

                 
            
        

    def save_activations(self, mos:ModularObserver, x:torch.Tensor):
        if not self.observing: return
        # No need for share_memory_() with threading - tensors already share memory
        ax = x.detach()
        mos_id = self.mos_id(mos)
        self.activation_queue.put((self.last_batch, mos_id, ax))
     

    def reset(self):
        self.position_history = []

        """ MOVE TO MODULAR CLUSTER
        self.last_cluster_centers = None
        self.last_clustering = None
        self.last_clustering_all = None
        self.last_cluster_stats = {}
        self.last_cluster_stats_wrong = {}
        """

    def path_checkpoint(self, filename):
        basepath = self.base_path + '/' if self.base_path is not None else ''
        basepath = os.path.join(basepath,'checkpoint')
        os.makedirs(basepath, exist_ok=True)
        path = os.path.join(basepath, filename)
        return path



    def save_space(self, filename=None):
        filename = f'{self.name}_space.pth' if filename is None else filename
        path = self.path_checkpoint(filename)
        if os.path.exists(path): os.remove(path)
        torch.save(self.positions, path)

    def load_space(self, filename=None):
        filename = f'{self.name}_space.pth' if filename is None else filename
        path = self.path_checkpoint(filename)
        if not os.path.exists(path): return False
        space_data = torch.load(path, weights_only=True)
        for pos, data  in zip(self.positions, space_data):
            pos.data += data.data 
        return True

            
    """
    def last_centers(self):
        layers = self.space_loss.layers_loss.values()
        actives = [layer.last_center_on for layer in layers]
        inactives = [layer.last_center_off for layer in layers]
        losses = [layer.last_loss for layer in layers]
        return actives, inactives, losses
    """

    def clear_history(self):
        self.position_history = []

    def add_history(self):
        if (self.history):
            pos = [ pos.clone().tolist() for pos in self.positions]
            self.position_history.append( pos )
    
    def rescale_factor_step(self):
        self.rescale_factor = self.rescale_factor * self.rescale_gamma
        print(f"Rescale factor adjusted to {self.rescale_factor:.4f}")




    def step_batch(self, space_loss:SpaceLoss=None, history=False):
        if space_loss is None: space_loss = self.space_loss
        if not space_loss.can_forward():
            print("step_batch: space loss not complete yet.")
            return
        loss = space_loss.forward()

        if self.optimizing:
            loss.backward()
            
            self.optim.step()
            self.lr_scheduler.step()
            self.optim.zero_grad()
            
        if history: 
            self.add_history()




    

    def rate(self):
        #TODO: produce a signature cluster-based, per datapoint.
        actives = [mo.criterion.last_center_on for mo in self.mos]
        
        #self.add_history()
        
        stats_std = []
        stats_avg = []
        
        bs = len(actives[0])
    
        for i in range(bs):
            goods = []

            for active in actives:
                goods.append(active[i])

            good = torch.vstack(goods).detach()
            
            #unweighted
            std, avg = torch.std_mean(good, dim=0)

            
            
            
            stats_std.append(std.tolist())
            stats_avg.append(avg.tolist())
        
        return stats_avg, stats_std
    
    
    # DEPS: positions
    """
    def cluster_space(self, method='kmeans', save=True, show_plots=False):
        positions_all = self.positions_all_last.numpy()

        
        
        mc_kmeans = ModularClusterKmeans()
        #mc_dbscan = ModularClusterDBScan()
        #mc_hdbscan = ModularClusterHDBScan() 
        ## KMEANS + score

        lbl_kmeans_sc, cluster_centers_sc = mc_kmeans.cluster_kmeans(positions_all, elbow=False)
        scores_kmeans_sc = mc_kmeans.cluster_scores(positions_all, lbl_kmeans_sc)

        ## KMEANS + elbow
        # lbl_kmeans_el, cluster_centers_el = mc_kmeans.cluster_kmeans(positions_all)
        # scores_kmeans_el = mc_kmeans.cluster_scores(positions_all, lbl_kmeans_el)
        
        self.last_cluster_centers = torch.from_numpy(cluster_centers_sc).to(device=self.device)
        self.last_clustering_all = torch.from_numpy(lbl_kmeans_sc).to(device=self.device) + 1
        self.last_clustering = self.last_clustering_all.split( self.mos_units_count )


        if show_plots:
            mc_kmeans.plot_cluster(positions_all, lbl_kmeans_sc, "KMEANS score")
            # mc_kmeans.plot_cluster(positions_all, lbl_kmeans_el, "KMEANS elbow")

            

        
        if self.best_cluster_score is None: self.best_cluster_score = scores_kmeans_sc
        
        silhouette, calinski, davies = scores_kmeans_sc
        silhouette_best, calinski_best, davies_best = self.best_cluster_score
        if silhouette >= silhouette_best:
            self.best_cluster_score = scores_kmeans_sc  
            filename = f'{self.name}_space_best.pth'
            self.save_space(filename)
            filename = f'{self.name}_clusters_best.pth'
            self.save_cluster_map(filename)
        

        #if self.debug:
        
        unique_labels = np.unique_counts(lbl_kmeans_sc)
        ul_cnts = len(unique_labels.counts)
        mean,std = np.mean(unique_labels.counts), np.std(unique_labels.counts)
        stats_sc = (ul_cnts, int(mean), int(std))
        #print(f"Unique labels score: { (ul_cnts, mean, std) }")

        #unique_labels = np.unique_counts(lbl_kmeans_el)
        #ul_cnts = len(unique_labels.counts)
        #mean,std = np.mean(unique_labels.counts), np.std(unique_labels.counts)
        #stats_el = (ul_cnts, int(mean), int(std))
        #print(f"Unique labels elbow: { (ul_cnts, mean, std) }")






        s_dx = silhouette - silhouette_best
        c_dx = calinski - calinski_best
        d_dx = davies - davies_best
        print(f"clusterspace: Improved silhouette score from {silhouette_best:.2f} to {silhouette:.2f} (↑Δs={s_dx:.2f}, ↑Δc={c_dx:.2f}, ↓Δd={d_dx:.2f})")
        
        
        

            
        
        print("clusterspace: position shape", positions_all.shape)
        print("KMEANS (best score) scores: ", scores_kmeans_sc, " tot,avg,std: ", stats_sc )
        #print("KMEANS (elbow) scores: ", scores_kmeans_el, " tot,avg,std: ", stats_el )



       
            
            
        if save: self.save_cluster_map()
    """






    def save_cluster_map(self, filename=None):
        filename = f'{self.name}_clusters.pth' if filename is None else filename
        path = self.path_checkpoint(filename)
        if os.path.exists(path): os.remove(path)
        clustering_data = (self.last_cluster_centers,self.last_clustering_all, self.last_clustering)
        torch.save(clustering_data, path) 
    
    def load_cluster_map(self, filename=None):
        filename = f'{self.name}_clusters.pth' if filename is None else filename
        path = self.path_checkpoint(filename)
        if not os.path.exists(path): return False
        clustering_data = torch.load(path, weights_only=True)
        self.last_cluster_centers, self.last_clustering_all, self.last_clustering = clustering_data
        self.last_clustering_all = self.last_clustering_all.to(device=self.device)
        self.last_clustering = [clustering.to(device=self.device) for clustering in self.last_clustering]
        return True
    
    def save_cluster_stats(self, filename=None):
        filename = f'{self.name}_clusters_stats.pth' if filename is None else filename
        path = self.path_checkpoint(filename)
        if os.path.exists(path): os.remove(path)
        clustering_data = (self.last_cluster_stats, self.last_cluster_stats_wrong)
        torch.save(clustering_data, path) 
    
    def load_cluster_stats(self, filename=None):
        filename = f'{self.name}_clusters_stats.pth' if filename is None else filename
        path = self.path_checkpoint(filename)
        if not os.path.exists(path): return False
        clustering_data = torch.load(path, weights_only=True)
        self.last_cluster_stats, self.last_cluster_stats_wrong = clustering_data
        #self.last_cluster_stats = [clustering.to(device=self.device) for clustering in self.last_cluster_stats]
        #self.last_cluster_stats_wrong = [clustering.to(device=self.device) for clustering in self.last_cluster_stats_wrong]
        return True

    # DEPS: cluster stats
    def cluster_class_stats(self):
        if self.last_cluster_stats is None:
            print("No clustering available.")
            return
        
        clusters_sum = {} 
        classes = set()
        for cluster_num, classes_stats in self.last_cluster_stats.items():
            cluster_num = int(cluster_num)
            class_sum = {}
            for c, cnt in classes_stats.items():
                if c not in class_sum: class_sum[c] = 0
                class_sum[c] += cnt
                classes.add(c)
            clusters_sum[cluster_num] = class_sum   
        
        classes = list(sorted(classes))
        table_header = ['Nr'] + classes
        table_data = []
        for cluster_num, class_sum in clusters_sum.items():
            # Ensure all classes are present in each cluster
            #for c in classes:
            #    if c not in class_sum:
            #        class_sum[c] = 0
            row = [f"#{cluster_num}"] + [class_sum.get(c, 0) for c in classes]
            table_data.append(row)
        cluster_stats = pd.DataFrame(table_data, columns=table_header)
        #print(cluster_stats)
        row_classes_hits = cluster_stats[classes]
        # Compute standard uniform distribution for each row

        uniform_dist = np.ones(len(row_classes_hits.columns)) / len(row_classes_hits.columns)
        # Compute KL divergence for each cluster (row)
        kl_divergences = []
        for idx, row in row_classes_hits.iterrows():
            p = row.values
            p_sum = p.sum()
            if p_sum == 0:
                kl = 0.0
            else:
                p_norm = p / p_sum
                kl = entropy(p_norm, uniform_dist)
            kl_divergences.append(kl)
        kld = np.array(kl_divergences)
        cluster_stats['KL_div'] = kld

        kld_exp = 1-np.exp(-kld) #exponential projection
        cluster_stats['KL_exp'] = kld_exp


        conditions = [
            (kld_exp < 0.25),
            (kld_exp >= 0.25) & (kld_exp <= 0.6),
            (kld_exp > 0.6)
        ]
        choices = np.array(['GENERAL', 'STANDARD', 'SPECIALIZED'])

        # Add the new column
        kl_type = np.select(conditions, choices, default='---')
        cluster_stats['kl_type'] = kl_type


        
        # top classes
        rows = []
        for idx, row in row_classes_hits.iterrows():
            normalized_row = row / row.sum()

            # Sort classes in descending order and compute cumulative sum
            sorted_classes = normalized_row.sort_values(ascending=False)
            cumsum = sorted_classes.cumsum()
            
            # Find the classes that make up the 90% quantile
            top_classes = sorted_classes[cumsum <= 0.90]
            if len(top_classes) == 0:  # Handle case where a single class is >90%
                top_classes = sorted_classes.head(1)
            rows.append(tuple(top_classes.index.tolist()))
            #print(top_classes.index.tolist())
        cluster_stats['top_classes'] = tuple(rows)

        # count neurons:
        class_count = np.unique_counts(self.last_clustering_all.cpu().numpy())
        class_count = {f'#{num}':int(cnt) for num,cnt in zip(*class_count)}
        cluster_stats['CU cnt'] = cluster_stats['Nr'].apply(lambda x: class_count.get(x, 0))
        

        # compute distances
        
        centers_stats = True
        if centers_stats:
            centers = self.last_cluster_centers
            centers_abs = centers.abs().to(dtype=torch.float64)
            centers_min = centers_abs.amin(axis=1).cpu().numpy()
            centers_max = centers_abs.amax(axis=1).cpu().numpy()
            centers_mean = centers_abs.mean(axis=1).cpu().numpy()
            centers_std = centers_abs.std(axis=1).cpu().numpy()
            centers_dist = centers_abs.pow(2).mean(axis=1).sqrt().cpu().numpy()


            cluster_stats['dist min'] = centers_min
            cluster_stats['dist max'] = centers_max
            cluster_stats['dist mean'] = centers_mean
            cluster_stats['dist std'] = centers_std
            cluster_stats['dist euc'] = centers_dist

        

        row_totals = row_classes_hits.sum(axis=1)
        row_totals_sum = row_totals.sum()
        #print(row_totals_sum)
        
        row_totals_perc = row_totals.div(row_totals_sum).mul(100).round(2)
         
        
        #print(row_totals)
        
        #print(cluster_stats)

        #classes_prec = [f'{c} %' for c in classes]
        series_cls_perc = cluster_stats[classes]
        #print(series_cls_perc)
        series_cls_perc = series_cls_perc.div(row_totals, axis=0).mul(100).round(2).fillna(0.0)
        #print(series_cls_perc)
        cluster_stats[classes] = series_cls_perc

        row_totals_std = series_cls_perc.std(axis=1).round(2).fillna(0.0)

        cluster_stats['Total'] = row_totals
        cluster_stats['Total (%)'] =  row_totals_perc
        cluster_stats['Std'] =  row_totals_std

        cluster_stats = cluster_stats.round(2)

        print(cluster_stats)
        cluster_stats.to_csv(f"{self.base_path}/{self.name}_cluster_class_attribution.csv", index=False)
        return cluster_stats
    

    # DEPS: space_loss, modularobserver
    def class_cluster_attribution(self, space_loss, ys, ys_hat):
        idxs_on = [ll.idxs_on.squeeze(2) for ll in space_loss.layers_loss.values()]
       
        ys = ys.unsqueeze(1)
        ys_hat = ys_hat.unsqueeze(1)

        bs = idxs_on[0].shape[0]
        batch_clustering = [ cluster_labels.unsqueeze(0).expand(bs, -1)  for cluster_labels in self.last_clustering]

        clusters_on = torch.cat([ cluster_labels[idx_on] for cluster_labels, idx_on in zip(batch_clustering, idxs_on)])
        ys_on =torch.cat([ (ys * idx_on)[idx_on] for idx_on in idxs_on])

        cluster_class = torch.cat((clusters_on.unsqueeze(1),ys_on.unsqueeze(1)), dim=1)

        unique, counts = torch.unique(cluster_class, return_counts=True, dim=0)
        #cluster_ys_counts = torch.cat( (unique, counts.unsqueeze(1)), dim=1)
        
        for (cluster,ys),counts in zip(unique, counts):
            cluster,ys,counts = int(cluster), int(ys), int(counts)
            
            if cluster not in self.last_cluster_stats: 
                self.last_cluster_stats[cluster] = {}
            if ys not in self.last_cluster_stats[cluster]: 
                self.last_cluster_stats[cluster][ys] = 0
            self.last_cluster_stats[cluster][ys] += counts



    def mask_by_class(self, class_num, threshold=0.0):# -> list[Tensor | Any]:
        masks = [torch.zeros_like(ls) for ls in self.last_clustering]
        for cluster_num, classes_num in self.last_cluster_stats.items():
            class_vals = np.array(list(classes_num.values()))
            total = np.sum(class_vals)
            
            filtered_classed = [k for k,v in classes_num.items() if total > 0 and v/total >= threshold]

            if str(class_num) in filtered_classed: continue
            cluster_layer_masks = [(layer_clustering == cluster_num) for layer_clustering in self.last_clustering]
            masks = [mask + clm for mask, clm in zip(masks,cluster_layer_masks)]
        masks = [(mask <= 0) for mask in masks] # invert bool mask: 0 = True, >0 = False
        
        #print(masks)

        #cns = [ ( (m == False).sum().item(),m.numel() ) for m in masks]
        #print(class_num, cns)

        return masks
    

