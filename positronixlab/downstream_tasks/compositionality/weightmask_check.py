from math import log

from tqdm import tqdm

from positronixlab.dataprovider.dataprovider_factory import DataProvider
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
        Uses gradient accumulation over class subsets to process the dataset in a single pass.
        """
        num_classes = self.dataprovider.num_classes
        criterion = WeightMaskCriterion()
        
        # Single manager and optimizer. The num_classes dimension inside MultiClassWeightmaskManager 
        # prevents hook conflicts and handles all parameters.
        mask_mgr = MultiClassWeightmaskManager(self.model, target_layers, num_classes)
        optimizer = torch.optim.Adam(mask_mgr.masks.parameters(), lr=lr)
        
        mask_mgr.train()
        device = next(self.model.parameters()).device
        for batch in tqdm(self.dataprovider.train, desc="Batch Processing", leave=False):
            if len(batch) == 2:
                x, y = batch
            elif len(batch) == 3:
                x, y, idxs = batch
            
            optimizer.zero_grad()
            
            for c in range(num_classes):
                mask = (y == c)
                
                x_c = x[mask].to(device) 
                y_c = y[mask].to(device)
                
                # Route the hook to apply the correct mask slice for this class
                mask_mgr.current_class = c
                
                logits = mask_mgr(x_c)
                
                loss_dict = criterion(logits, y_c, mask_mgr.active_layer_masks)
                loss_dict["total_loss"].backward()
                
            # Step the optimizer once per batch after all classes have accumulated their gradients
            optimizer.step()

        logger.info("Completed optimization for all classes. Extracting binary masks...")
        
        mask_mgr.eval()
        class_masks = {}
        with torch.no_grad():
            for c in range(num_classes):
                class_masks[c] = {}
                for name, param in mask_mgr.masks.items():
                    class_masks[c][name] = (param[c] > 0).float()
                    
        mask_mgr.remove_hooks()
            
        return class_masks

class MultiClassWeightmaskManager(nn.Module):
    def __init__(self, model: nn.Module, target_layers: Dict[str, nn.Module], num_classes: int, temp: float = 1.0):
        super().__init__()
        self.model = model
        self.target_layers = target_layers 
        self.temp = temp
        self.num_classes = num_classes
        self.current_class = 0 
        
        self.masks = nn.ParameterDict()
        for name, module in target_layers.items():
            if hasattr(module, 'weight'):
                safe_name = name.replace('.', '_')
                # Grab the device directly from the original weight tensor
                self.masks[safe_name] = nn.Parameter(
                    torch.zeros((num_classes, *module.weight.shape), device=module.weight.device)
                )

        self.active_layer_masks: List[torch.Tensor] = []
        self._hook_handles = []
        self._setup_hooks_and_freezing()

    def _setup_hooks_and_freezing(self):
        for param in self.model.parameters():
            param.requires_grad = False

        for name, module in self.target_layers.items():
            safe_name = name.replace('.', '_')
            
            # 1. Stash the parameter ONCE during setup and delete it from the module
            if hasattr(module, 'weight'):
                module._original_weight_param = module.weight
                delattr(module, 'weight')
            
            def pre_hook(mod, inputs, s_name=safe_name):
                # 2. Slice the mask for the current class
                class_mask_logits = self.masks[s_name][self.current_class]
                mask = self._get_computed_mask(class_mask_logits, mod.training)
                
                self.active_layer_masks.append(class_mask_logits)
                
                # 3. Dynamically assign the masked tensor for this forward pass.
                # Because we don't delete it afterward, it's always safely available.
                mod.weight = mod._original_weight_param * mask

            self._hook_handles.append(module.register_forward_pre_hook(pre_hook))
            # Notice: The fragile post_hook is completely removed.

    def _get_computed_mask(self, mask_scores: torch.Tensor, training: bool) -> torch.Tensor:
        if training:
            u = torch.rand_like(mask_scores)
            gumbel = -torch.log(-torch.log(u + 1e-8) + 1e-8)
            soft_mask = torch.sigmoid((mask_scores + gumbel) / self.temp)
            hard_mask = (soft_mask > 0.5).float()
            return hard_mask - soft_mask.detach() + soft_mask
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
        # 1. Remove the hooks
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()
        
        # 2. Safely restore the original parameters to the model
        for name, module in self.target_layers.items():
            if hasattr(module, '_original_weight_param'):
                # Delete the dynamically assigned tensor if it exists
                if hasattr(module, 'weight'):
                    delattr(module, 'weight')
                # Restore the original parameter
                module.weight = module._original_weight_param
                delattr(module, '_original_weight_param')

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