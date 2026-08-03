# modularnet/criterions/spatial_loss.py
import torch
import torch.nn as nn
from typing import List, Dict

class WeightMaskCriterion(nn.Module):
    def __init__(self, lambda_l1: float = 1e-4):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.task_loss_fn = nn.CrossEntropyLoss()

    def _compute_overall_spaceloss(self, layer_masks: List[torch.Tensor]) -> torch.Tensor:
        """
        Computes overall_spaceloss by averaging the sparsity penalty 
        across all layers at the current timestep.
        """
        if not layer_masks:
            return torch.tensor(0.0, requires_grad=True, device=layer_masks[0].device if layer_masks else 'cpu')
        
        layer_losses = [torch.norm(torch.sigmoid(m), p=1) for m in layer_masks]
        overall_spaceloss = sum(layer_losses) / len(layer_losses)
        return overall_spaceloss

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, layer_masks: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        task_loss = self.task_loss_fn(logits, targets)
        overall_spaceloss = self._compute_overall_spaceloss(layer_masks)
        
        total_loss = task_loss + (self.lambda_l1 * overall_spaceloss)
        
        return {
            "total_loss": total_loss,
            "task_loss": task_loss,
            "overall_spaceloss": overall_spaceloss
        }