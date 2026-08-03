import torch
from typing import Dict

class WeightMasksMetrics:
    @staticmethod
    def compute_iou(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
        """
        Intersection over Union (IoU).
        Measures the exact overlap ratio of weights required by two distinct tasks/classes.
        """
        bin_a = (mask_a > 0).bool()
        bin_b = (mask_b > 0).bool()
        
        intersection = (bin_a & bin_b).sum().float()
        union = (bin_a | bin_b).sum().float()
        
        return (intersection / union).item() if union > 0 else 0.0

    @staticmethod
    def compute_iomin(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
        """
        Intersection over Minimum (IoMin).
        Captures compositional subsetness. Yields 1.0 if the active weights 
        of one task are entirely contained within the active weights of the other.
        """
        bin_a = (mask_a > 0).bool()
        bin_b = (mask_b > 0).bool()
        
        intersection = (bin_a & bin_b).sum().float()
        min_size = min(bin_a.sum().float(), bin_b.sum().float())
        
        return (intersection / min_size).item() if min_size > 0 else 0.0

    @staticmethod
    def compute_sparsity(mask: torch.Tensor) -> float:
        """
        Active weight ratio.
        Measures the percentage of the network's structural capacity actively 
        utilized to solve the isolated task.
        """
        active_params = (mask > 0).sum().float()
        total_params = torch.tensor(mask.numel(), dtype=torch.float, device=mask.device)
        
        return (active_params / total_params).item() if total_params > 0 else 0.0
        
    @staticmethod
    def evaluate_network_modularity(masks_a: Dict[str, torch.Tensor], masks_b: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Aggregates metrics across the entire network architecture by averaging 
        the layer-wise intersections.
        """
        total_iou, total_iomin = 0.0, 0.0
        layers = list(masks_a.keys())
        
        for layer in layers:
            total_iou += WeightMasksMetrics.compute_iou(masks_a[layer], masks_b[layer])
            total_iomin += WeightMasksMetrics.compute_iomin(masks_a[layer], masks_b[layer])
            
        num_layers = len(layers)
        return {
            "network_iou": total_iou / num_layers if num_layers > 0 else 0.0,
            "network_iomin": total_iomin / num_layers if num_layers > 0 else 0.0
        }