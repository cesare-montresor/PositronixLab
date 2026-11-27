import logging
import math
import time
from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor, nn

from positronixlab.dataprovider.dataprovider import DataProvider
from positronixlab.deeplearning.deeplearning_base import DeeplearningBase
from positronixlab.modularobserver.modularobserver import ModularObserver
from positronixlab.pipeline import PipelineConfig
from positronixlab.spacemath.spacemath import SpaceMath

# PREV: from torch.distributions.kl import kl_divergence  <- kl_divergence dispatches on torch.distributions.Distribution
#       subclasses (e.g. Categorical), not on raw probability tensors, so calling it on prob_dist/target_dist
#       tensors below raised NotImplementedError. Replaced with a manual epsilon-smoothed KL computation instead.


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
        self.total_count: float = 0.0

    def update(self, class_labels: Any, counts: Tensor | int | float | None = None):
        if isinstance(class_labels, int) or (torch.is_tensor(class_labels) and class_labels.dim() == 0):
            label = int(class_labels)
            count = float(counts) if counts is not None else 1.0
            self.counts[label] += count
            self.total_count += count
            return

        if counts is None or isinstance(counts, (int, float)):
            counts_t = torch.ones_like(class_labels, dtype=torch.float32)
        else:
            counts_t = counts
        for label, count in zip(class_labels, counts_t, strict=False):
            self.counts[label] += count
            self.total_count += float(count)

    def get_distribution(self) -> Tensor:
        return self.counts / self.total_count if self.total_count > 0 else self.counts

    def add(self, other: Distribution):
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

    def add(self, other: Distribution):
        if not isinstance(other, TensorDistribution):
            raise TypeError("Can only add another TensorDistribution.")
        if self.shape != other.shape:
            raise ValueError("Cannot add distributions with different shapes.")
        self.counts += other.counts
        self.total_count += other.total_count

class AtlasCheck:
    def __init__(self, 
                 config: PipelineConfig, 
                 model: nn.Module, 
                 dataprovider: DataProvider, 
                 clusters: dict[str, Tensor],
                 ):

        self.config = config
        self.model = model
        self.dataprovider = dataprovider
        self.clusters = clusters
        self.output_shape = None
        
        self.cluster_class_stats: dict[str, dict[int, DiscreteDistribution]] = {}  # layer, cluster, class, count
    
        self.observer = ModularObserver(config=self.config, model=self.model ) 
        self.observer.map()  # Map the model layers for observation
        self.layer_names = list(self.observer.layer_names())
        self.latest_scores: dict[str, Tensor | None] = {}
        self._num_classes = int(self.dataprovider.num_classes or 0)
        self.y_distribution = DiscreteDistribution(num_classes=self._num_classes)
 
        self.eval = DeeplearningBase(
            config=self.config,
            model=self.model,
            dataloader=self.dataprovider.train, # all 3 ? all data? or just train? cordas ? 
        )
        
        self.eval.on_prediction(self._on_prediction)
        self.eval.on_batch_start(self._on_batch_start)

    def get_distribution(self) -> DiscreteDistribution:
        return DiscreteDistribution(num_classes=self._num_classes)
   
    def run(self):
        # Implement the modularity score test logic here
        self.cluster_class_stats = {}
        self.y_distribution = self.get_distribution()
        self.output_shape = None
        
        self.observer.on_forward_layer(self._process_activations)
        try:
            self.eval.process_epoch(self.dataprovider.train, epoch=0, epoch_total=1, draft=False)
        finally:
            self.observer.on_forward_layer(None)

    
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

        self.y_distribution.update(ys)
        
        for layer_name, scores in self.latest_scores.items():
            if scores is None:
                continue
            clusters = self.clusters[layer_name]
            if layer_name not in self.cluster_class_stats:
                self.cluster_class_stats[layer_name] = {}
            current_stats = self.cluster_class_stats[layer_name]

            partial_stats = self._cluster_class_attribution(scores, clusters, ys)

            
            for cluster, distribution in partial_stats.items():
                if cluster not in current_stats:
                    current_stats[cluster] = distribution
                else:
                    current_stats[cluster].add(distribution)
        

    def _process_activations(self, layer: nn.Module, x_in: Tensor, activations: Tensor):
        node = self.observer.by_layer(layer)
        if node is None:
            return
        act_fn = node.activation_fn if isinstance(node.activation_fn, nn.Module) else None
        scores = SpaceMath.activations_to_scores(
            activations,
            x_quantile=self.config.modular_space_active_q,
            layer_type=node.layer_type,
            activation_fn=act_fn,
            cnn_locality_weight=0.5,
        )
        self.latest_scores[node.name] = scores
    
    
    def _cluster_class_attribution(self, scores: Tensor, clusters: Tensor, ys: Tensor):
        idxs_on = ( scores > 0.0 )                     # [bs, num_features]
        bs, num_features = idxs_on.shape

        batch_clustering = clusters.unsqueeze(0).expand(bs, -1)      # [bs, num_features]
        model_dev: Any = getattr(self.model, "device", None) or next(self.model.parameters()).device
        batch_ys = ys.unsqueeze(1).to(model_dev).expand(-1, num_features)  # [bs, num_features]

        clusters_on = batch_clustering[idxs_on].unsqueeze(1)  # [N, 1]
        ys_on = batch_ys[idxs_on].unsqueeze(1)                # [N, 1]

        cluster_class = torch.cat((clusters_on, ys_on), dim=1)
        unique, counts = torch.unique(cluster_class, return_counts=True, dim=0)

        cluster_stats = {}
        for (cluster, y), count in zip(unique, counts, strict=False):
            cluster, y, count = int(cluster), int(y), int(count)
            if cluster not in cluster_stats:
                cluster_stats[cluster] = self.get_distribution()
            cluster_stats[cluster].update(y, count)

        return cluster_stats

    def _cluster_output_attribution(self, scores: Tensor, clusters: Tensor, y_hats: Tensor):
        """WIP: find an efficent way to attribute 'the raw output vector (or latente sapace)' to each cluster"""
        idxs_on = scores > 0.0
        
        y_hats = y_hats.unsqueeze(1)


        bs = idxs_on.shape[0]
        batch_clustering = clusters.unsqueeze(0).expand(bs, -1) 
        
        clusters_on = torch.cat([ cluster_labels[idx_on] for cluster_labels, idx_on in zip(batch_clustering, idxs_on, strict=False)])
        ys_on = torch.cat([ (y_hats * idx_on)[idx_on] for idx_on in idxs_on])

        
        cluster_class = torch.cat((clusters_on.unsqueeze(1),ys_on.unsqueeze(1)), dim=1)
        unique, counts = torch.unique(cluster_class, return_counts=True, dim=0)
        
        cluster_stats = {}
        for (cluster, y_hat), count in zip(unique, counts, strict=False):
            cluster_id, y_val, cnt = int(cluster), int(y_hat), int(count)
            
            if cluster_id not in cluster_stats: 
                cluster_stats[cluster_id] = self.get_distribution()
            
            cluster_stats[cluster_id].update(y_val, cnt)

        return cluster_stats
        
    def extract_weightmasks(self, threshold: float = 0.0):
        num_masks = self._num_classes
        masks = {}
        for curr_class in range(num_masks):
            class_masks = {}
            for layer_id, layer_cluster in self.cluster_class_stats.items():
                class_mask = torch.zeros(self.clusters[layer_id].shape, dtype=torch.bool)
                for cluster, distribution in layer_cluster.items():
                    prob_dist = distribution.get_distribution() - self.y_distribution.get_distribution()
                    if prob_dist[curr_class] > threshold:
                        class_mask[self.clusters[layer_id] == cluster] = True
                class_masks[layer_id] = class_mask
            masks[curr_class] = class_masks
        return masks
    

    def modularity_score(self, kld_target: str = 'y'):
        stats = {}

        target_dist = None
        if kld_target == 'y':
            target_dist = self.y_distribution.get_distribution()
        elif kld_target == 'uniform':
            target_dist = torch.ones(self._num_classes) / self._num_classes
        else:
            raise ValueError(f"Unknown kld_target: {kld_target}. Use 'y' or 'uniform'.")

        
        for layer_id, layer_clusters in self.cluster_class_stats.items():
            layer_total_items = 0
            layer_purity_sum = 0.0
            layer_entropy_sum = 0.0
            layer_kld_sum = 0.0
            layer_kld_exp_sum = 0.0

            layers_labels = self.clusters[layer_id]
            
            for cluster, distribution_obj in layer_clusters.items():
                prob_dist = distribution_obj.get_distribution()
                cluster_size = int(torch.sum(layers_labels == cluster).item())
                
                if cluster_size == 0:
                    continue
                
                epsilon = 1e-9 # Prevent log(0) / div-by-0
                kld = torch.sum(prob_dist * torch.log((prob_dist + epsilon) / (target_dist + epsilon))).item()
                kld_exp = 1 - math.exp(-kld) #exponential projection
                    
                # 1. Purity: The probability of the most frequent class in this cluster
                purity = torch.max(prob_dist).item()
                
                # 2. Entropy: H = -sum(p * log(p)). Lower entropy = higher modularity.
                entropy = -torch.sum(prob_dist * torch.log(prob_dist + epsilon)).item()
                
                # Weight the metrics by cluster size
                layer_purity_sum += purity * cluster_size
                layer_entropy_sum += entropy * cluster_size
                layer_kld_sum += kld * cluster_size
                layer_kld_exp_sum += kld_exp * cluster_size
                layer_total_items += cluster_size
                
                logger.debug(f"Layer: {layer_id}, Cluster: {cluster}, Size: {cluster_size}, Purity: {purity:.4f}")

            # Compute layer-wide averages
            if layer_total_items > 0:
                avg_purity = layer_purity_sum / layer_total_items
                avg_entropy = layer_entropy_sum / layer_total_items
                avg_kld = layer_kld_sum / layer_total_items
                avg_kld_exp = layer_kld_exp_sum / layer_total_items
            else:
                avg_purity, avg_entropy, avg_kld, avg_kld_exp = 0.0, 0.0, 0.0, 0.0
                
            stats[layer_id] = {
                'modularity_purity': avg_purity,
                'modularity_entropy': avg_entropy,
                'modularity_kld': avg_kld,
                'modularity_kld_exp': avg_kld_exp
            }
            
            logger.info(f"Layer {layer_id} -> Avg Purity: {avg_purity:.4f}, Avg Entropy: {avg_entropy:.4f}, Avg KLD: {avg_kld:.4f}, Avg KLD Exp: {avg_kld_exp:.4f}")

        n_layers = len(stats)
        if n_layers > 0:
            stats['model'] = {
                'modularity_purity': sum(s['modularity_purity'] for s in stats.values()) / n_layers,
                'modularity_entropy': sum(s['modularity_entropy'] for s in stats.values()) / n_layers,
                'modularity_kld': sum(s['modularity_kld'] for s in stats.values()) / n_layers,
                'modularity_kld_exp': sum(s['modularity_kld_exp'] for s in stats.values()) / n_layers,
            }
        else:
            stats['model'] = {
                'modularity_purity': 0.0,
                'modularity_entropy': 0.0,
                'modularity_kld': 0.0,
                'modularity_kld_exp': 0.0,
            }
            
        return stats