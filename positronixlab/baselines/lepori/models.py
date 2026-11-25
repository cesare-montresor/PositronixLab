"""Standard-PyTorch & upstream-Transformers models from Lepori et al. (NeurIPS 2023).

Reference: ``libs/Compositional_Subnetworks/``
- Vision CVR ResNet50 / WideResNet50 (with InstanceNorm2d + Decision MLP + OOO head)
- Vision CVR ViT (upstream ``transformers.ViTModel`` + Decision MLP + OOO head)
- Language SynEval BERT (upstream ``transformers.BertModel`` + OOO head)

Avoids the forked ``transformers`` library and PyTorch Lightning harness while
supporting direct loading of ``*_backbone.pt`` / ``*_mlp.pt`` checkpoints saved
by ``Compositional_Subnetworks`` (and extracting ``*.mask_weight`` tensors into
canonical :class:`SubnetworkMask` objects).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from positronixlab.baselines.base import SOTAModel


class OddOneOutHead(nn.Module):
    """Parameter-free cosine-similarity 4-way Odd-One-Out head (Lepori et al., 2023).

    Given embeddings of shape ``(B, N, D)`` (typically ``N = 4``), computes
    L2-normalized embeddings and returns ``-sum_j cos(x_i, x_j)`` as logits of
    shape ``(B, N)`` so the outlier receives the highest logit.
    """

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 3:
            raise ValueError(
                f"OddOneOutHead expects 3D tensor (B, N, D), got shape {tuple(embeddings.shape)}"
            )
        normed = F.normalize(embeddings, dim=-1)
        sim = torch.bmm(normed, normed.transpose(1, 2)).sum(dim=-1)
        return -sim


def strip_lepori_l0_keys(
    state_dict: Dict[str, Any],
    prefix: str = "",
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Separate base weights from Lepori ``L0Linear``/``L0Conv2d`` checkpoints.

    Returns:
        ``(clean_weights, mask_weights)`` where ``mask_weights`` maps canonical
        dotted module names (e.g. ``"backbone.layer1.0.conv1"`` or ``"mlp.fc1"``)
        to their saved ``mask_weight`` logit tensors.
    """
    raw = state_dict
    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        raw = raw["state_dict"]
    elif isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        raw = raw["model"]

    clean_weights: Dict[str, torch.Tensor] = {}
    mask_weights: Dict[str, torch.Tensor] = {}
    pref = f"{prefix}." if prefix else ""

    for key, val in raw.items():
        if not isinstance(val, torch.Tensor):
            continue
        if key.endswith(".mask_weight"):
            mod_name = key[: -len(".mask_weight")]
            mask_weights[f"{pref}{mod_name}"] = val
            continue
        if key.endswith(".random_weight") or key.endswith(".mask"):
            continue
        clean_weights[f"{pref}{key}"] = val

    return clean_weights, mask_weights


class CVRDecisionMLP(nn.Module):
    """Decision MLP matching ``Vision/CVR/models/decisionMLP.py:MLP``."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 2048,
        out_dim: int = 128,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        if hidden_dim > 0:
            self.fc1 = nn.Linear(in_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, out_dim)
        else:
            self.fc1 = nn.Linear(in_dim, out_dim)
            self.fc2 = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fc2 is not None:
            return self.fc2(F.relu(self.fc1(x)))
        return self.fc1(x)


class CVRResNetModel(SOTAModel):
    """CVR 4-way Odd-One-Out ResNet model (Lepori et al., 2023).

    Uses ``torchvision.models`` with ``norm_layer=nn.InstanceNorm2d`` (matching
    ``Vision/CVR/models/resnet.py``) + :class:`CVRDecisionMLP` +
    :class:`OddOneOutHead`.

    Expected input shape: ``(B, 4, 3, H, W)`` (e.g. ``H=W=128``) or ``(B, N, 3, H, W)``.
    Output shape: ``(B, 4)`` logits.
    """

    source = "lepori"
    arch = "lepori:cvr_resnet50"
    task_type = "odd_one_out"

    def __init__(
        self,
        backbone_name: str = "resnet50",
        mlp_hidden_dim: int = 2048,
        mlp_out_dim: int = 128,
        use_instance_norm: bool = True,
        pretrained: bool = False,
        name: Optional[str] = None,
        device: Optional[torch.device | str] = None,
    ) -> None:
        super().__init__(name=name or f"CVR_{backbone_name}", device=device)
        import torchvision.models as tv_models

        self.backbone_name = backbone_name
        self.arch = f"lepori:cvr_{backbone_name}"
        norm_layer = nn.InstanceNorm2d if use_instance_norm else nn.BatchNorm2d

        builders = {
            "resnet18": tv_models.resnet18,
            "resnet34": tv_models.resnet34,
            "resnet50": tv_models.resnet50,
            "wideresnet50": tv_models.wide_resnet50_2,
            "wide_resnet50_2": tv_models.wide_resnet50_2,
        }
        if backbone_name not in builders:
            raise ValueError(
                f"Unknown CVR ResNet backbone '{backbone_name}'. "
                f"Choose from {sorted(builders.keys())}."
            )

        weights = "DEFAULT" if (pretrained and not use_instance_norm) else None
        backbone = builders[backbone_name](weights=weights, norm_layer=norm_layer)
        num_ftrs = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.mlp = CVRDecisionMLP(num_ftrs, mlp_hidden_dim, mlp_out_dim)
        self.ooo_head = OddOneOutHead()
        if device is not None:
            self.to(device)

    def encode_items(self, x: torch.Tensor) -> torch.Tensor:
        """Encode flattened images ``(B*N, C, H, W) -> (B*N, D)``."""
        return self.mlp(self.backbone(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                f"CVRResNetModel expects 5D input (B, N, C, H, W), got {tuple(x.shape)}"
            )
        b, n, c, h, w = x.shape
        emb = self.encode_items(x.reshape(b * n, c, h, w)).reshape(b, n, -1)
        return self.ooo_head(emb)

    def load_lepori_components(
        self,
        backbone_path: Optional[str | Path] = None,
        mlp_path: Optional[str | Path] = None,
        strict: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Load Lepori ``*_backbone.pt`` and ``*_mlp.pt`` checkpoints.

        Returns any extracted ``*.mask_weight`` tensors keyed by canonical dotted
        module path.
        """
        extracted_masks: Dict[str, torch.Tensor] = {}
        if backbone_path is not None:
            raw = torch.load(backbone_path, map_location=self.device, weights_only=True)
            clean, masks = strip_lepori_l0_keys(raw, prefix="backbone")
            # Remove "backbone." prefix when loading directly into self.backbone
            bb_clean = {k[len("backbone.") :]: v for k, v in clean.items()}
            self.backbone.load_state_dict(bb_clean, strict=strict)
            extracted_masks.update(masks)
        if mlp_path is not None:
            raw = torch.load(mlp_path, map_location=self.device, weights_only=True)
            clean, masks = strip_lepori_l0_keys(raw, prefix="mlp")
            mlp_clean = {k[len("mlp.") :]: v for k, v in clean.items()}
            self.mlp.load_state_dict(mlp_clean, strict=strict)
            extracted_masks.update(masks)
        return extracted_masks


class CVRViTModel(SOTAModel):
    """CVR 4-way Odd-One-Out ViT model using upstream ``transformers.ViTModel``."""

    source = "lepori"
    arch = "lepori:cvr_vit"
    task_type = "odd_one_out"
    default_exclude_patterns = ("embeddings",)

    def __init__(
        self,
        image_size: int = 128,
        patch_size: int = 16,
        num_hidden_layers: int = 12,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        mlp_hidden_dim: int = 2048,
        mlp_out_dim: int = 128,
        pretrained_name: Optional[str] = None,
        name: Optional[str] = None,
        device: Optional[torch.device | str] = None,
    ) -> None:
        super().__init__(name=name or "CVR_ViT", device=device)
        from transformers import ViTConfig, ViTModel

        if pretrained_name is not None:
            self.backbone = ViTModel.from_pretrained(pretrained_name)
            embed_dim = self.backbone.config.hidden_size
        else:
            cfg = ViTConfig(
                image_size=image_size,
                patch_size=patch_size,
                num_hidden_layers=num_hidden_layers,
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
            )
            self.backbone = ViTModel(cfg)
            embed_dim = hidden_size

        self.mlp = CVRDecisionMLP(embed_dim, mlp_hidden_dim, mlp_out_dim)
        self.ooo_head = OddOneOutHead()
        if device is not None:
            self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                f"CVRViTModel expects 5D input (B, N, C, H, W), got {tuple(x.shape)}"
            )
        b, n, c, h, w = x.shape
        out = self.backbone(pixel_values=x.reshape(b * n, c, h, w))
        cls_emb = out.last_hidden_state[:, 0, :]
        emb = self.mlp(cls_emb).reshape(b, n, -1)
        return self.ooo_head(emb)


class SynEvalBertModel(SOTAModel):
    """Syntactic 4-way Odd-One-Out BERT model using upstream ``transformers.BertModel``.

    Matches ``Language/SyntacticSubnetworks/model.py:BertClf`` (defaulting to
    BERT-small dimensions: 4 layers, hidden size 512, 8 heads, dropout 0.0).

    Accepts either:
    - a single 4D tensor ``(B, 2, 4, L)`` where dim 1 is ``[input_ids, attention_mask]``, OR
    - two 3D tensors ``(input_ids, attention_mask)`` each of shape ``(B, 4, L)``.
    """

    source = "lepori"
    arch = "lepori:syneval_bert"
    task_type = "odd_one_out"
    default_exclude_patterns = ("embeddings",)

    def __init__(
        self,
        pretrained_name: Optional[str] = None,
        vocab_size: int = 30522,
        hidden_size: int = 512,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 8,
        intermediate_size: int = 2048,
        name: Optional[str] = None,
        device: Optional[torch.device | str] = None,
    ) -> None:
        super().__init__(name=name or "SynEval_BERT", device=device)
        from transformers import BertConfig, BertModel

        if pretrained_name is not None:
            self.backbone = BertModel.from_pretrained(pretrained_name)
        else:
            cfg = BertConfig(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
            )
            self.backbone = BertModel(cfg)

        self.ooo_head = OddOneOutHead()
        if device is not None:
            self.to(device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attention_mask is None and input_ids.ndim == 4 and input_ids.shape[1] == 2:
            attention_mask = input_ids[:, 1, :, :]
            input_ids = input_ids[:, 0, :, :]

        if input_ids.ndim != 3:
            raise ValueError(
                f"SynEvalBertModel expects (B, N, L) input_ids or (B, 2, N, L) stacked tensor, "
                f"got shape {tuple(input_ids.shape)}"
            )
        b, n, seq_len = input_ids.shape
        flat_ids = input_ids.reshape(b * n, seq_len)
        flat_mask = (
            attention_mask.reshape(b * n, seq_len)
            if attention_mask is not None
            else torch.ones_like(flat_ids)
        )
        out = self.backbone(input_ids=flat_ids, attention_mask=flat_mask)
        cls_emb = out.last_hidden_state[:, 0, :].reshape(b, n, -1)
        return self.ooo_head(cls_emb)
