from abc import ABC, abstractmethod
from collections import defaultdict
from importlib.metadata import distribution
import logging
import time
from typing import Any, Dict, Optional

from torch import Tensor, nn
import torch

from modularnet.dataprovider.dataprovider import DataProvider
from modularnet.deeplearning.deeplearning_base import DeeplearningBase
from modularnet.modularobserver.modularobserver import ModularObserver
from modularnet.spacemath.spacemath import SpaceMath
from modularnet.trial.trial_config import TrialConfig



logger = logging.getLogger(__name__)

class Distribution(ABC):
    @abstractmethod
    def update(self, data: Any): pass

    @abstractmethod
    def get_distribution(self) -> Any: pass

    @abstractmethod
    def add(self, other: 'Distribution'): pass


class DiscreteDistribution(Distribution):
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.counts = torch.zeros(num_classes, dtype=torch.float32)
        self.total_count = 0

    def update(self, class_labels: Tensor, counts: Tensor=None):
        if counts is None:
            counts = torch.ones_like(class_labels, dtype=torch.float32)
        for label in class_labels:
            self.counts[label] += counts[label]
            self.total_count += counts[label]

    def get_distribution(self) -> Tensor:
        return self.counts / self.total_count if self.total_count > 0 else self.counts

    def add(self, other: 'DiscreteDistribution'):
        if not isinstance(other, DiscreteDistribution):
            raise TypeError("Can only add another DiscreteDistribution.")
        if self.num_classes != other.num_classes:
            raise ValueError("Cannot add distributions with different number of classes.")
        self.counts += other.counts
        self.total_count += other.total_count


class TensorDistribution(Distribution):
    def __init__(self, shape: torch.Size):
        self.shape = shape
        self.counts = torch.zeros(shape, dtype=torch.float32)
        self.total_count = 0

    def update(self, y_hat: Tensor):
        self.counts += y_hat
        self.total_count += 1

    def get_distribution(self) -> Tensor:
        return (self.counts / self.total_count) if self.total_count > 0 else self.counts

    def add(self, other: 'TensorDistribution'):
        if not isinstance(other, TensorDistribution):
            raise TypeError("Can only add another TensorDistribution.")
        if self.shape != other.shape:
            raise ValueError("Cannot add distributions with different shapes.")
        self.counts += other.counts
        self.total_count += other.total_count

class AtlasCheck:
    def __init__(self, 
                 config: TrialConfig, 
                 model: nn.Module, 
                 dataprovider: DataProvider, 
                 clusters: Dict[str, Tensor],
                 ):

        self.config = config
        self.model = model
        self.dataprovider = dataprovider
        self.clusters = clusters
        self.output_shape = None
        
        self.cluster_class_stats: Dict[str, Dict[str, DiscreteDistribution]] # layer, cluster, class, count
    
        self.observer = ModularObserver(config=self.config, model=self.model ) 
        self.layer_names = list(self.observer.layer_names())
        self.latest_scores = {}
 
        self.eval = DeeplearningBase(
            config=self.config,
            model=self.model,
            dataloader=self.dataprovider.train, # all 3 ? all data? or just train? cordas ? 
        )
        
        self.observer.on_forward_layer(self.process_activations)
        self.eval.on_prediction(self._on_prediction)
        self.eval.on_batch_start(self._on_batch_start)

    def get_distribution(self) -> DiscreteDistribution:
        return DiscreteDistribution(num_classes=self.dataprovider.num_classes)
   
    def run(self):
        # Implement the modularity score test logic here
        self.eval.process_epoch(self.dataprovider.train, epoch=0, epoch_total=1, draft=False)
        pass

    
    def _on_batch_start(self, epoch, epoch_total, batch_idx, batch_total, data_batch):
        self.latest_scores = {name: None for name in self.layer_names}

    def _on_prediction(self, epoch, epoch_total, batch_idx, batch_total, data_batch, predictions):
        if len(data_batch) == 2: # standard case: x, y
            xs, ys = data_batch
        elif len(data_batch) == 3: # data track index for continual learning downstram tasks
            xs, ys, idxs = data_batch

        if self.output_shape is None:
            self.output_shape = predictions.shape[1:]  # Exclude batch dimension

        while any(score is None for score in self.latest_scores.values()):
            time.sleep(0.01)  # Wait for all layers to be processed
            pass

        for layer_name, scores in self.latest_scores.items():
            clusters = self.clusters[layer_name]
            if layer_name not in self.cluster_class_stats:
                self.cluster_class_stats[layer_name] = {}
            current_stats = self.cluster_class_stats.get(layer_name)


            partial_stats = self.class_cluster_attribution(scores, clusters, ys)

            
            for cluster, distribution in partial_stats.items():
                if cluster not in current_stats:
                    current_stats[cluster] = distribution
                else:
                    current_stats[cluster].add(distribution)
        

    def process_activations(self, layer: nn.Module, x_in: Tensor, activations: Tensor):
        node = self.observer.by_layer(layer)
        scores = SpaceMath.activations_to_scores(
            x_quantile=self.config.modular_space_active_q,
            activations=activations,
            layer_type=node.layer_type,
            activation_fn=node.activation_fn,
            cnn_locality_weight=0.5,
        )
        self.latest_scores[node.name] = scores
    
    
    def class_cluster_attribution(self, scores: Tensor, clusters: Tensor, ys: Tensor):
        idxs_on = scores > 0.0
        
        ys = ys.unsqueeze(1)

        bs = idxs_on[0].shape[0]
        batch_clustering = clusters.unsqueeze(0).expand(bs, -1) 
        
        clusters_on = torch.cat([ cluster_labels[idx_on] for cluster_labels, idx_on in zip(batch_clustering, idxs_on)])
        ys_on = torch.cat([ (ys * idx_on)[idx_on] for idx_on in idxs_on])

        
        cluster_class = torch.cat((clusters_on.unsqueeze(1),ys_on.unsqueeze(1)), dim=1)
        unique, counts = torch.unique(cluster_class, return_counts=True, dim=0)
        
        cluster_stats = {}
        for (cluster,ys),counts in zip(unique, counts):
            cluster,ys,counts = int(cluster), int(ys), int(counts)
            
            if cluster not in cluster_stats: 
                cluster_stats[cluster] = self.get_distribution()
            
            cluster_stats[cluster].update(ys, counts)

        return cluster_stats


