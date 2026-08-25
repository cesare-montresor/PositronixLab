from math import log

from tqdm import tqdm

from modularnet.dataprovider.dataprovider_factory import DataProvider
import logging
import torch
import torch.nn as nn
from typing import Dict, List

logger = logging.getLogger(__name__)

class WeightmaskCheck:
    def __init__(self, model: nn.Module, dataprovider: DataProvider):
        self.model = model
        self.dataprovider = dataprovider

    def run(self, target_layers: Dict[str, nn.Module], num_epochs: int = 5, lr: float = 1e-2) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Optimizes and exports differentiable Gumbel-Sigmoid weight masks per class.
        Returns Dict[class_id, Dict[layer_name, binary_mask_tensor]].
        """
        class_masks = {}
        criterion = WeightMaskCriterion()
        
        for c in range(self.dataprovider.num_classes):
            logger.info(f"Optimizing Csordas baseline masks for Class {c}")
            mask_mgr = WeightmaskManager(self.model, target_layers)
            optimizer = torch.optim.Adam(mask_mgr.masks.parameters(), lr=lr)
            
            mask_mgr.train()
            for epoch in tqdm(range(num_epochs), desc=f"Epoch", leave=False):
                for batch in tqdm(self.dataprovider.train, desc=f"Csordas. Class {c} Batch Processing", leave=False):
                    if len(batch) == 2:
                        x, y = batch
                    elif len(batch) == 3:
                        x, y, idxs = batch
                    
                        # Filter batch strictly for the current isolated task/class
                        mask = (y == c)
                        if torch.sum(mask) <= 1:
                            continue
                        
                    x_c = x[mask].to(self.model.device) 
                    y_c = y[mask].to(self.model.device)
                    
                    optimizer.zero_grad()
                    logits = mask_mgr(x_c)
                    loss_dict = criterion(logits, y_c, mask_mgr.active_layer_masks)
                    loss_dict["total_loss"].backward()
                    optimizer.step()
                logger.info(f"Epoch {epoch+1}/{num_epochs} - Class {c} Loss: {loss_dict['total_loss'].item():.4f}, Task Loss: {loss_dict['task_loss'].item():.4f}, Weightmask Loss: {loss_dict['overall_weightmaskloss'].item():.4f}")
            logger.info(f"Completed optimization for Class {c}. Extracting binary masks...")

            mask_mgr.eval()
            with torch.no_grad():
                class_masks[c] = {
                    name: (param > 0).float() 
                    for name, param in mask_mgr.masks.items()
                }
            mask_mgr.remove_hooks()
            
        return class_masks

class WeightmaskManager(nn.Module):
    def __init__(self, model: nn.Module, target_layers: Dict[str, nn.Module], temp: float = 1.0):
        super().__init__()
        self.model = model
        self.temp = temp
        
        self.masks = nn.ParameterDict()
        for name, module in target_layers.items():
            if hasattr(module, 'weight'):
                safe_name = name.replace('.', '_')
                self.masks[safe_name] = nn.Parameter(torch.ones_like(module.weight))

        self.active_layer_masks: List[torch.Tensor] = []
        self._hook_handles = []
        self._setup_hooks_and_freezing(target_layers)

    def _setup_hooks_and_freezing(self, target_layers: Dict[str, nn.Module]):
        for param in self.model.parameters():
            param.requires_grad = False

        for name, module in target_layers.items():
            safe_name = name.replace('.', '_')
            
            def pre_hook(mod, inputs, s_name=safe_name):
                if hasattr(mod, 'weight'):
                    mod._original_weight = mod.weight
                    mask = self._get_computed_mask(s_name, mod.training)
                    self.active_layer_masks.append(self.masks[s_name])
                    
                    delattr(mod, 'weight')
                    mod.weight = mod._original_weight * mask

            def post_hook(mod, inputs, outputs):
                if hasattr(mod, '_original_weight'):
                    delattr(mod, 'weight')
                    mod.weight = mod._original_weight
                    delattr(mod, '_original_weight')

            self._hook_handles.append(module.register_forward_pre_hook(pre_hook))
            self._hook_handles.append(module.register_forward_hook(post_hook))

    def _get_computed_mask(self, safe_name: str, training: bool) -> torch.Tensor:
        mask_scores = self.masks[safe_name]
        if training:
            u = torch.rand_like(mask_scores)
            gumbel = -torch.log(-torch.log(u + 1e-8) + 1e-8)
            return torch.sigmoid((mask_scores + gumbel) / self.temp)
        else:
            return (mask_scores > 0).float()

    def clear_active_masks(self):
        self.active_layer_masks.clear()

    def compute_spatial_loss(self) -> torch.Tensor:
        if not self.active_layer_masks:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        layer_losses = [torch.norm(torch.sigmoid(m), p=1) for m in self.active_layer_masks]
        return sum(layer_losses) / len(layer_losses)

    def forward(self, *args, **kwargs):
        self.clear_active_masks()
        return self.model(*args, **kwargs)

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

class WeightMaskCriterion(nn.Module):
    def __init__(self, lambda_l1: float = 1e-4):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.task_loss_fn = nn.CrossEntropyLoss()

    def _compute_overall_weightmaskloss(self, layer_masks: List[torch.Tensor]) -> torch.Tensor:
        if not layer_masks:
            return torch.tensor(0.0, requires_grad=True, device=layer_masks[0].device if layer_masks else 'cpu')
        
        layer_losses = [torch.norm(torch.sigmoid(m), p=1) for m in layer_masks]
        overall_weightmaskloss = sum(layer_losses) / len(layer_losses)
        return overall_weightmaskloss

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, layer_masks: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        task_loss = self.task_loss_fn(logits, targets)
        overall_weightmaskloss = self._compute_overall_weightmaskloss(layer_masks)
        total_loss = task_loss + (self.lambda_l1 * overall_weightmaskloss)
        return {
            "total_loss": total_loss,
            "task_loss": task_loss,
            "overall_weightmaskloss": overall_weightmaskloss
        }

class WeightMasksMetrics:
    @staticmethod
    def compute_iou(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
        bin_a = (mask_a > 0).bool()
        bin_b = (mask_b > 0).bool()
        intersection = (bin_a & bin_b).sum().float()
        union = (bin_a | bin_b).sum().float()
        return (intersection / union).item() if union > 0 else 0.0

    @staticmethod
    def compute_iomin(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
        bin_a = (mask_a > 0).bool()
        bin_b = (mask_b > 0).bool()
        intersection = (bin_a & bin_b).sum().float()
        min_size = min(bin_a.sum().float(), bin_b.sum().float())
        return (intersection / min_size).item() if min_size > 0 else 0.0

    @staticmethod
    def compute_sparsity(mask: torch.Tensor) -> float:
        active_params = (mask > 0).sum().float()
        total_params = torch.tensor(mask.numel(), dtype=torch.float, device=mask.device)
        return (active_params / total_params).item() if total_params > 0 else 0.0
        
    @staticmethod
    def evaluate_network_modularity(masks_a: Dict[str, torch.Tensor], masks_b: Dict[str, torch.Tensor]) -> Dict[str, float]:
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
    
    @staticmethod
    def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
        predictions = torch.argmax(logits, dim=1)
        correct = (predictions == targets).sum().item()
        total = targets.size(0)
        return correct / total if total > 0 else 0.0

    @staticmethod
    def compute_sequence_accuracy(predictions: torch.Tensor, targets: torch.Tensor, pad_token_id: int = 0) -> float:
        valid_mask = (targets != pad_token_id)
        elementwise_match = (predictions == targets) | ~valid_mask
        sequence_match = elementwise_match.all(dim=1)
        correct = sequence_match.sum().item()
        total = targets.size(0)
        return correct / total if total > 0 else 0.0