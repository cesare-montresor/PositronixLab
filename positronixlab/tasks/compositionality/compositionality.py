import logging
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from positronixlab.trial.trial_config import TrialConfig
from positronixlab.dataprovider.dataprovider import DataProvider
from positronixlab.tasks.compositionality.weightmask_check import WeightmaskCheck
from positronixlab.tasks.compositionality.atlas_check import AtlasCheck

logger = logging.getLogger(__name__)


class StaticMaskApplier:
    """
    Context manager that applies static binary weight masks to target layers
    during forward passes and restores original weights upon exit.
    """
    def __init__(self, model: nn.Module, masks: Dict[str, torch.Tensor]):
        self.model = model
        self.masks = masks
        self._handles = []
        self._orig_weights = {}

    def __enter__(self):
        for name, module in self.model.named_modules():
            safe_name = name.replace('.', '_')
            if safe_name in self.masks and hasattr(module, 'weight') and module.weight is not None:
                mask = self.masks[safe_name].to(module.weight.device)
                self._orig_weights[name] = module.weight

                def make_pre_hook(m_mask, orig_w):
                    def pre_hook(mod, inputs):
                        delattr(mod, 'weight')
                        mod.weight = orig_w * m_mask
                    return pre_hook

                def make_post_hook(orig_w):
                    def post_hook(mod, inputs, outputs):
                        delattr(mod, 'weight')
                        mod.weight = orig_w
                    return post_hook

                self._handles.append(module.register_forward_pre_hook(make_pre_hook(mask, module.weight)))
                self._handles.append(module.register_forward_hook(make_post_hook(module.weight)))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        
        # Guard: ensure all original weights are strictly restored
        for name, module in self.model.named_modules():
            if name in self._orig_weights:
                if not hasattr(module, 'weight') or module.weight is not self._orig_weights[name]:
                    if hasattr(module, 'weight'):
                        delattr(module, 'weight')
                    module.weight = self._orig_weights[name]
        self._orig_weights.clear()


class CompositionalityCheck:
    """
    Evaluates modular compositionality and cross-class interference:
    1. Records untainted baseline Confusion Matrix.
    2. Runs inference per class mask for both Atlas and Csordas.
    3. Computes retention, off-target suppression, and CM comparisons.
    """
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        num_classes: int,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.dataloader = dataloader
        self.num_classes = num_classes
        self.device = device or next(model.parameters()).device

    @torch.no_grad()
    def _evaluate_confusion_matrix(self) -> torch.Tensor:
        """Runs evaluation over the dataloader and returns a (C, C) confusion matrix."""
        cm = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64, device=self.device)
        self.model.eval()

        for batch in self.dataloader:
            if len(batch) == 2:
                xs, ys = batch
            elif len(batch) == 3:
                xs, ys, _ = batch
            else:
                raise ValueError(f"Unsupported batch format of length {len(batch)}")

            xs, ys = xs.to(self.device), ys.to(self.device)
            logits = self.model(xs)
            preds = torch.argmax(logits, dim=-1)

            indices = ys * self.num_classes + preds
            batch_cm = torch.bincount(indices, minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
            cm += batch_cm

        return cm.cpu()

    def compute_untainted_cm(self) -> torch.Tensor:
        logger.info("Computing untainted baseline Confusion Matrix...")
        return self._evaluate_confusion_matrix()

    def evaluate_masked_runs(
        self,
        masks_by_class: Dict[int, Dict[str, torch.Tensor]],
        method_name: str = "Atlas"
    ) -> Dict[int, Dict[str, Any]]:
        results = {}
        for c in range(self.num_classes):
            if c not in masks_by_class:
                logger.warning(f"[{method_name}] Mask for class {c} not found. Skipping.")
                continue

            class_mask = masks_by_class[c]
            with StaticMaskApplier(self.model, class_mask):
                cm_c = self._evaluate_confusion_matrix()

            total_target_samples = cm_c[c].sum().item()
            target_correct = cm_c[c, c].item()
            target_recall = (target_correct / total_target_samples) if total_target_samples > 0 else 0.0

            total_off_target = cm_c.sum().item() - total_target_samples
            off_target_leakage = (cm_c[:, c].sum().item() - target_correct) / total_off_target if total_off_target > 0 else 0.0

            results[c] = {
                "confusion_matrix": cm_c,
                "target_recall": target_recall,
                "off_target_leakage": off_target_leakage,
                "total_accuracy": cm_c.diag().sum().item() / cm_c.sum().item() if cm_c.sum().item() > 0 else 0.0,
            }
            logger.info(f"[{method_name}] Class {c} Subnetwork -> Recall on C: {target_recall:.4f} | Off-Target Leakage: {off_target_leakage:.4f}")

        return results

    

    def run(
        self,
        atlas_masks: Dict[int, Dict[str, torch.Tensor]],
        csordas_masks: Dict[int, Dict[str, torch.Tensor]]
    ) -> Dict[str, Any]:
        untainted_cm = self.compute_untainted_cm()
        untainted_total_acc = untainted_cm.diag().sum().item() / untainted_cm.sum().item()

        logger.info("--- Evaluating Atlas Class Masks ---")
        atlas_results = self.evaluate_masked_runs(atlas_masks, method_name="Atlas")

        logger.info("--- Evaluating Csordas Baseline Masks ---")
        csordas_results = self.evaluate_masked_runs(csordas_masks, method_name="Csordas")

        comparison = {}
        for c in range(self.num_classes):
            if c in atlas_results and c in csordas_results:
                cm_a = atlas_results[c]["confusion_matrix"]
                cm_c = csordas_results[c]["confusion_matrix"]

                # 1. Normalizza per riga (somma delle righe) per ottenere le proporzioni (0-1)
                cm_a_norm = cm_a / (cm_a.sum(dim=1, keepdim=True) + 1e-6)
                cm_c_norm = cm_c / (cm_c.sum(dim=1, keepdim=True) + 1e-6)

                cm_diff = torch.norm((cm_a_norm - cm_c_norm), p='fro').item()
                
                comparison[c] = {
                    "atlas_target_recall": atlas_results[c]["target_recall"],
                    "csordas_target_recall": csordas_results[c]["target_recall"],
                    "atlas_off_target_leakage": atlas_results[c]["off_target_leakage"],
                    "csordas_off_target_leakage": csordas_results[c]["off_target_leakage"],
                    "cm_frobenius_diff": cm_diff,
                }

        return {
            "untainted": {
                "confusion_matrix": untainted_cm,
                "accuracy": untainted_total_acc,
            },
            "atlas": atlas_results,
            "csordas": csordas_results,
            "comparison": comparison,
        }


class CompositionalityPipeline:
    """
    High-level orchestrator to generate masks from both paradigms and run the compositionality evaluation.
    """
    def __init__(self, model: nn.Module, dataprovider: DataProvider, config: TrialConfig):
        self.model = model
        self.dataprovider = dataprovider
        self.config = config

    def run_full_evaluation(
        self, 
        clusters: Dict[str, torch.Tensor], 
        atlas_threshold: float = 0.0,
        csordas_epochs: int = 5,
        csordas_lr: float = 1e-2
    ) -> Dict[str, Any]:


        logger.info("Generating Atlas Cluster Masks...")
        atlas = AtlasCheck(
            config=self.config,
            model=self.model,
            dataprovider=self.dataprovider,
            clusters=clusters
        )
        atlas.run()
        atlas_masks = atlas.extract_weightmasks(threshold=atlas_threshold)
        target_layers = {node.name:node.module_ref for node in atlas.observer.nodes_list}  # Assuming AtlasCheck provides the target layers used for mask generation


        logger.info("Optimizing Csordas Baseline Masks...")
        csordas = WeightmaskCheck(
            model=self.model,
            dataprovider=self.dataprovider
        )
        csordas_masks = csordas.run(target_layers=target_layers, num_epochs=csordas_epochs, lr=csordas_lr)
        
        checker = CompositionalityCheck(
            model=self.model,
            dataloader=self.dataprovider.test,
            num_classes=self.dataprovider.num_classes
        )
        results = checker.run(atlas_masks=atlas_masks, csordas_masks=csordas_masks)

        return results