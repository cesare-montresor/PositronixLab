import torch
import torch.nn as nn
from typing import Dict, List

class WeightMaskManager(nn.Module):
    """
    Manages external Gumbel-Sigmoid masks via forward hooks and computes 
    the averaged spatial L1 sparsity penalty without modifying the base model.
    """
    def __init__(self, model: nn.Module, target_layers: Dict[str, nn.Module], temp: float = 1.0):
        super().__init__()
        self.model = model
        self.temp = temp
        
        # External Storage for Mask Scores (Only parameters to be optimized)
        self.masks = nn.ParameterDict()
        for name, module in target_layers.items():
            if hasattr(module, 'weight'):
                safe_name = name.replace('.', '_')
                self.masks[safe_name] = nn.Parameter(torch.ones_like(module.weight))

        # Dynamic execution tracker for the current forward pass
        self.active_layer_masks: List[torch.Tensor] = []
        
        # Freeze base model and attach hooks
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
        """Must be invoked right before every forward pass to prevent OOM growth."""
        self.active_layer_masks.clear()

    def compute_spatial_loss(self) -> torch.Tensor:
        """Averages the L1 sparsity penalty across all executed layers."""
        if not self.active_layer_masks:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        layer_losses = [torch.norm(torch.sigmoid(m), p=1) for m in self.active_layer_masks]
        return sum(layer_losses) / len(layer_losses)

    def forward(self, *args, **kwargs):
        """Pass-through forward method that automatically resets execution state."""
        self.clear_active_masks()
        return self.model(*args, **kwargs)

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()