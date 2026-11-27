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

import copy
from collections.abc import Sequence
from pathlib import Path
from typing import Any

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
    state_dict: dict[str, Any],
    prefix: str = "",
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
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

    clean_weights: dict[str, torch.Tensor] = {}
    mask_weights: dict[str, torch.Tensor] = {}
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
        self.fc2: nn.Linear | None
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
        name: str | None = None,
        device: torch.device | str | None = None,
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
        backbone: Any = builders[backbone_name](weights=weights, norm_layer=norm_layer)
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
        backbone_path: str | Path | None = None,
        mlp_path: str | Path | None = None,
        strict: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Load Lepori ``*_backbone.pt`` and ``*_mlp.pt`` checkpoints.

        Returns any extracted ``*.mask_weight`` tensors keyed by canonical dotted
        module path.
        """
        extracted_masks: dict[str, torch.Tensor] = {}
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
        pretrained_name: str | None = None,
        name: str | None = None,
        device: torch.device | str | None = None,
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
        pretrained_name: str | None = None,
        vocab_size: int = 30522,
        hidden_size: int = 512,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 8,
        intermediate_size: int = 2048,
        name: str | None = None,
        device: torch.device | str | None = None,
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
        attention_mask: torch.Tensor | None = None,
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


# ---------------------------------------------------------------------------
# Vision: CVR SimCLR self-supervised pretraining (SimCLRModel.py / SimCLR.py)
# ---------------------------------------------------------------------------


def _simclr_similarity_and_masks(
    embeddings: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute scaled cosine similarity matrix (diagonal masked) and positive-pair mask."""
    if embeddings.ndim != 2 or embeddings.shape[0] < 2 or embeddings.shape[0] % 2 != 0:
        raise ValueError(
            "SimCLR expects 2D embeddings of even batch size (2*N, D), "
            f"got shape {tuple(embeddings.shape)}"
        )
    if temperature <= 0.0:
        raise ValueError(f"SimCLR temperature must be > 0, got {temperature}")

    z = F.normalize(embeddings, dim=-1)
    cos_sim = z @ z.transpose(0, 1)
    n = cos_sim.shape[0]
    self_mask = torch.eye(n, dtype=torch.bool, device=cos_sim.device)
    pos_mask = self_mask.roll(shifts=n // 2, dims=0)
    fill_val = -1e9 if cos_sim.dtype in (torch.float16, torch.bfloat16) else -9e15
    cos_sim = cos_sim.masked_fill(self_mask, fill_val)
    return cos_sim / temperature, pos_mask


class SimCLRLoss(nn.Module):
    """NT-Xent / InfoNCE contrastive loss matching Lepori et al. ``SimCLRModel.py``.

    Expects ``embeddings`` of shape ``(2 * N, D)`` where ``embeddings[i]`` and
    ``embeddings[(i + N) % (2 * N)]`` are the two augmented views of item ``i``.
    Accepts an optional ignored *targets* argument so it plugs directly into
    ``DeeplearningTrain`` or standard ``criterion(preds, targets)`` loops.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError(f"SimCLRLoss temperature must be > 0, got {temperature}")
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del targets
        logits, pos_mask = _simclr_similarity_and_masks(embeddings, self.temperature)
        nll = -logits[pos_mask] + torch.logsumexp(logits, dim=-1)
        return nll.mean()


@torch.no_grad()
def simclr_ranking_metrics(
    embeddings: torch.Tensor,
    temperature: float = 0.07,
) -> dict[str, float]:
    """Compute SimCLR positive-pair ranking metrics (``top1``, ``top5``, ``mean_pos``).

    Matches ``SimCLRModel.info_nce_loss`` logging (``acc_top1``, ``acc_top5``,
    ``acc_mean_pos``).
    """
    logits, pos_mask = _simclr_similarity_and_masks(embeddings, temperature)
    fill_val = -1e9 if logits.dtype in (torch.float16, torch.bfloat16) else -9e15
    comb_sim = torch.cat(
        [logits[pos_mask].unsqueeze(1), logits.masked_fill(pos_mask, fill_val)],
        dim=-1,
    )
    sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
    return {
        "top1": float((sim_argsort == 0).float().mean().item()),
        "top5": float((sim_argsort < 5).float().mean().item()),
        "mean_pos": float((1.0 + sim_argsort.float().mean()).item()),
    }


class CVRSimCLRModel(SOTAModel):
    """CVR SimCLR self-supervised pretraining model (Lepori et al., 2023).

    Reference: ``libs/Compositional_Subnetworks/Vision/CVR/models/SimCLRModel.py``

    Uses a ``torchvision`` ResNet backbone with ``nn.InstanceNorm2d`` (identical
    to :class:`CVRResNetModel`) followed by a 2-layer :class:`CVRDecisionMLP`
    projection head (default ``2048 -> 2048 -> 128``).

    Accepts any of:
    - Stacked tensor ``(B, 2, K, C, H, W)`` from :class:`CVRSimCLRDataProvider`
      (2 views × ``K`` images per item, e.g. ``K=2``),
    - List/tuple of 2 view tensors ``[v0, v1]`` each of shape ``(B, K, C, H, W)``
      or ``(B, C, H, W)``,
    - Stacked 2-view tensor ``(B, 2, C, H, W)``.

    Returns projected embeddings of shape ``(2 * N, mlp_out_dim)`` ordered so
    view 0 occupies ``[:N]`` and view 1 occupies ``[N:]``, directly consumable
    by :class:`SimCLRLoss`.
    """

    source = "lepori"
    arch = "lepori:cvr_simclr_resnet50"
    task_type = "contrastive"

    def __init__(
        self,
        backbone_name: str = "resnet50",
        mlp_hidden_dim: int = 2048,
        mlp_out_dim: int = 128,
        use_instance_norm: bool = True,
        temperature: float = 0.07,
        name: str | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(
            name=name or f"CVR_SimCLR_{backbone_name}",
            device=device,
            criterion=SimCLRLoss(temperature=temperature),
        )
        import torchvision.models as tv_models

        self.backbone_name = backbone_name
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_out_dim = mlp_out_dim
        self.use_instance_norm = use_instance_norm
        self.temperature = temperature
        self.arch = f"lepori:cvr_simclr_{backbone_name}"

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
                f"Unknown CVR SimCLR backbone '{backbone_name}'. "
                f"Choose from {sorted(builders.keys())}."
            )

        backbone: Any = builders[backbone_name](weights=None, norm_layer=norm_layer)
        num_ftrs = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.mlp = CVRDecisionMLP(num_ftrs, mlp_hidden_dim, mlp_out_dim)
        if device is not None:
            self.to(device)

    @staticmethod
    def _flatten_views(
        views: torch.Tensor | Sequence[torch.Tensor],
    ) -> torch.Tensor:
        if isinstance(views, (list, tuple)):
            if len(views) != 2:
                raise ValueError(f"Expected 2 views, got {len(views)}")
            v0, v1 = views[0], views[1]
        elif isinstance(views, torch.Tensor):
            if views.ndim == 6 and views.shape[1] == 2:
                v0, v1 = views[:, 0], views[:, 1]
            elif views.ndim == 5 and views.shape[1] == 2:
                v0, v1 = views[:, 0], views[:, 1]
            else:
                raise ValueError(
                    "CVRSimCLRModel expects (B, 2, K, C, H, W), (B, 2, C, H, W), "
                    f"or a 2-element list/tuple of view tensors; got shape {tuple(views.shape)}"
                )
        else:
            raise TypeError(f"Unsupported views type: {type(views)!r}")

        if v0.ndim == 5:
            b, k, c, h, w = v0.shape
            v0 = v0.reshape(b * k, c, h, w)
            v1 = v1.reshape(b * k, c, h, w)
        elif v0.ndim != 4:
            raise ValueError(
                f"Each SimCLR view must be 4D (B, C, H, W) or 5D (B, K, C, H, W), "
                f"got {tuple(v0.shape)}"
            )
        return torch.cat([v0, v1], dim=0)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode flattened images ``(N, C, H, W) -> (N, mlp_out_dim)``."""
        return self.mlp(self.backbone(images))

    def forward(
        self,
        views: torch.Tensor | Sequence[torch.Tensor],
    ) -> torch.Tensor:
        imgs = self._flatten_views(views)
        return self.encode(imgs)

    def to_ooo_model(
        self,
        *,
        mlp_hidden_dim: int = 2048,
        mlp_out_dim: int = 128,
        device: torch.device | str | None = None,
    ) -> CVRResNetModel:
        """Create a downstream :class:`CVRResNetModel` initialized with this pretrained backbone.

        Matches Lepori et al.'s protocol: transfers ``self.backbone`` weights
        and discards the SimCLR projection MLP.
        """
        target_device = device if device is not None else self.device
        ooo = CVRResNetModel(
            backbone_name=self.backbone_name,
            mlp_hidden_dim=mlp_hidden_dim,
            mlp_out_dim=mlp_out_dim,
            use_instance_norm=self.use_instance_norm,
            pretrained=False,
            device=target_device,
        )
        ooo.backbone.load_state_dict(copy.deepcopy(self.backbone.state_dict()), strict=True)
        return ooo

    def save_lepori_components(self, out_dir: str | Path) -> tuple[Path, Path]:
        """Save ``backbone.pt`` and ``mlp.pt`` matching ``Vision/CVR/SimCLR.py``."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        bb_path = out / "backbone.pt"
        mlp_path = out / "mlp.pt"
        torch.save({k: v.detach().cpu() for k, v in self.backbone.state_dict().items()}, bb_path)
        torch.save({k: v.detach().cpu() for k, v in self.mlp.state_dict().items()}, mlp_path)
        return bb_path, mlp_path

    def load_lepori_components(
        self,
        backbone_path: str | Path | None = None,
        mlp_path: str | Path | None = None,
        strict: bool = True,
    ) -> None:
        """Load ``backbone.pt`` and/or ``mlp.pt`` saved by ``SimCLR.py`` or :meth:`save_lepori_components`."""
        if backbone_path is not None:
            raw = torch.load(backbone_path, map_location=self.device, weights_only=True)
            clean, _ = strip_lepori_l0_keys(raw)
            self.backbone.load_state_dict(clean, strict=strict)
        if mlp_path is not None:
            raw = torch.load(mlp_path, map_location=self.device, weights_only=True)
            clean, _ = strip_lepori_l0_keys(raw)
            self.mlp.load_state_dict(clean, strict=strict)


def _move_views(
    views: torch.Tensor | Sequence[torch.Tensor],
    device: torch.device,
) -> torch.Tensor | list[torch.Tensor]:
    if isinstance(views, torch.Tensor):
        return views.to(device)
    return [v.to(device) for v in views]


def pretrain_simclr(
    model: CVRSimCLRModel,
    dataprovider: Any,
    *,
    epochs: int = 100,
    lr: float = 5e-4,
    weight_decay: float = 0.0,
    temperature: float = 0.07,
    checkpoint_dir: str | Path | None = None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Run Lepori et al.'s SimCLR contrastive pretraining loop on *model*.

    Reference: ``libs/Compositional_Subnetworks/Vision/CVR/SimCLR.py`` and
    ``configs/SimCLR/SimCLR_Config.yaml`` (``AdamW(lr=5e-4, weight_decay=0.0)``,
    ``CosineAnnealingLR(T_max=epochs, eta_min=lr/50)``, ``temperature=0.07``,
    best checkpoint selected by ``val_top5``).

    .. note::
        Standalone pretraining loop outside ``DeeplearningTrain``; tracked in
        ``docs/deeplearning/DEEPLEARNING_SYSTEM.md`` for future trainer unification.
    """
    target_device = torch.device(device) if device is not None else model.device
    model.to(target_device)

    criterion = SimCLRLoss(temperature=temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs), eta_min=lr / 50.0
    )

    train_loader = dataprovider.train if hasattr(dataprovider, "train") else dataprovider
    val_loader = getattr(dataprovider, "val", None)

    history: dict[str, list[dict[str, float]]] = {"train": [], "val": []}
    best_score = -1.0
    best_state: dict[str, torch.Tensor] | None = None

    for _epoch in range(epochs):
        model.train()
        tr_totals = {"loss": 0.0, "top1": 0.0, "top5": 0.0, "mean_pos": 0.0}
        tr_batches = 0
        for batch in train_loader:
            views = _move_views(batch[0], target_device)
            optimizer.zero_grad()
            emb = model(views)
            loss = criterion(emb)
            loss.backward()
            optimizer.step()

            metrics = simclr_ranking_metrics(emb.detach(), temperature=temperature)
            tr_totals["loss"] += float(loss.item())
            for k, v in metrics.items():
                tr_totals[k] += v
            tr_batches += 1

        if tr_batches > 0:
            tr_epoch = {k: v / tr_batches for k, v in tr_totals.items()}
        else:
            tr_epoch = dict(tr_totals)
        history["train"].append(tr_epoch)

        val_epoch: dict[str, float] | None = None
        if val_loader is not None:
            model.eval()
            va_totals = {"loss": 0.0, "top1": 0.0, "top5": 0.0, "mean_pos": 0.0}
            va_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    views = _move_views(batch[0], target_device)
                    emb = model(views)
                    loss = criterion(emb)
                    metrics = simclr_ranking_metrics(emb, temperature=temperature)
                    va_totals["loss"] += float(loss.item())
                    for k, v in metrics.items():
                        va_totals[k] += v
                    va_batches += 1
            if va_batches > 0:
                val_epoch = {k: v / va_batches for k, v in va_totals.items()}
                history["val"].append(val_epoch)

        monitor = val_epoch["top5"] if val_epoch is not None else tr_epoch["top5"]
        if monitor >= best_score:
            best_score = monitor
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
        model.to(target_device)

    saved_paths: tuple[Path, Path] | None = None
    if checkpoint_dir is not None:
        saved_paths = model.save_lepori_components(checkpoint_dir)

    return {
        "model": model,
        "best_val_top5": best_score,
        "history": history,
        "checkpoint_paths": saved_paths,
    }
