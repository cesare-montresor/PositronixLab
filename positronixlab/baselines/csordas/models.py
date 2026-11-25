"""Standard-PyTorch equivalents of models from Csordas et al. (ICLR 2021).

Reference: ``libs/are_neural_nets_modular/models/``
- ``FeedforwardModel`` -> :class:`CsordasFeedforwardModel`
- ``ConvModel``        -> :class:`CsordasConvModel`
- ``LSTMModel``        -> :class:`CsordasLSTMModel`
- ``ResNet110``        -> :class:`CsordasResNet110`
- ``TransformerEncDecModel`` -> :class:`CsordasTransformerEncDec`

All models use standard PyTorch layers (``nn.Linear``, ``nn.Conv2d``,
``nn.BatchNorm2d``, ``nn.LSTM``, ``nn.Transformer``) while preserving parameter
naming conventions so checkpoints trained in ``are_neural_nets_modular`` can be
loaded via :func:`strip_csordas_mask_keys`.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from positronixlab.baselines.base import SOTAModel


def strip_csordas_mask_keys(
    state_dict: Dict[str, Any],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, torch.Tensor]]]:
    """Separate base model weights from Csordas ``MaskedModel`` checkpoints.

    Handles both plain ``state_dict``s and wrapped dicts (``{"model": ...}`` or
    ``MaskedModel`` state dicts containing ``masks.<stage>.<param>`` and
    ``model.<param>`` keys).

    Returns:
        ``(clean_weights, stage_mask_logits)`` where ``stage_mask_logits`` maps
        ``stage_key -> {param_name: logit_tensor}``.
    """
    raw = state_dict
    if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        raw = raw["model"]

    clean_weights: Dict[str, torch.Tensor] = {}
    stage_masks: Dict[str, Dict[str, torch.Tensor]] = {}

    for key, val in raw.items():
        if not isinstance(val, torch.Tensor):
            continue
        if key.startswith("masks."):
            parts = key.split(".", 2)
            if len(parts) == 3:
                stage, param_key = parts[1], parts[2]
                stage_masks.setdefault(stage, {})[param_key] = val
            continue
        clean_key = key[len("model.") :] if key.startswith("model.") else key
        clean_weights[clean_key] = val

    return clean_weights, stage_masks


class CsordasFeedforwardModel(SOTAModel):
    """MLP matching ``are_neural_nets_modular/models/feedforward.py:FeedforwardModel``."""

    source = "csordas"
    arch = "csordas:mlp"
    task_type = "classification"

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        layer_sizes: Optional[List[int]] = None,
        name: Optional[str] = None,
        device: Optional[torch.device | str] = None,
    ) -> None:
        super().__init__(name=name or "CsordasFeedforwardModel", device=device)
        sizes = list(layer_sizes) if layer_sizes is not None else [800, 800, 256]
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.layer_sizes = sizes
        self.layers = nn.ModuleList(
            [
                nn.Linear(in_s, out_s)
                for in_s, out_s in zip([n_inputs] + sizes, sizes + [n_outputs])
            ]
        )
        self.reset_parameters()
        if device is not None:
            self.to(device)

    def reset_parameters(self) -> None:
        last_idx = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            gain = 1.0 if i == last_idx else nn.init.calculate_gain("relu")
            nn.init.xavier_normal_(layer.weight, gain=gain)
            if layer.bias is not None:
                layer.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for i, layer in enumerate(self.layers):
            if i != 0:
                out = F.relu(out)
            out = layer(out)
        return out


class CsordasConvModel(SOTAModel):
    """4-layer CNN matching ``are_neural_nets_modular/models/conv.py:ConvModel``."""

    source = "csordas"
    arch = "csordas:conv"
    task_type = "classification"

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 10,
        dropout: bool = True,
        name: Optional[str] = None,
        device: Optional[torch.device | str] = None,
    ) -> None:
        super().__init__(name=name or "CsordasConvModel", device=device)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.Dropout(0.25 if dropout else 0.0),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5 if dropout else 0.0),
        )
        self.out_layer = nn.Linear(256, out_channels)
        if device is not None:
            self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(self.features(x).mean(dim=(2, 3)))


class CsordasLSTMModel(SOTAModel):
    """LSTM + linear projection matching ``are_neural_nets_modular/models/lstm.py``."""

    source = "csordas"
    arch = "csordas:lstm"
    task_type = "sequence"

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        state_size: int = 256,
        n_layers: int = 2,
        batch_first: bool = True,
        name: Optional[str] = None,
        device: Optional[torch.device | str] = None,
    ) -> None:
        super().__init__(name=name or "CsordasLSTMModel", device=device)
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.state_size = state_size
        self.n_layers = n_layers
        self.batch_first = batch_first
        self.lstm = nn.LSTM(
            n_inputs, state_size, num_layers=n_layers, batch_first=batch_first
        )
        self.output_projection = nn.Linear(state_size, n_outputs)
        if device is not None:
            self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.output_projection(out)


class _CifarBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut: nn.Module = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class CsordasResNet110(SOTAModel):
    """CIFAR ResNet-110 (3 stages x 18 blocks, 16/32/64 ch) matching Csordas et al."""

    source = "csordas"
    arch = "csordas:resnet110"
    task_type = "classification"

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        num_blocks: Tuple[int, int, int] = (18, 18, 18),
        name: Optional[str] = None,
        device: Optional[torch.device | str] = None,
    ) -> None:
        super().__init__(name=name or "CsordasResNet110", device=device)
        self.in_planes = 16
        self.conv1 = nn.Conv2d(
            in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)
        if device is not None:
            self.to(device)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for s in strides:
            blocks.append(_CifarBasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3]).flatten(1)
        return self.linear(out)


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class CsordasTransformerEncDec(SOTAModel):
    """Standard PyTorch encoder-decoder Transformer matching SCAN / DM-Math setup."""

    source = "csordas"
    arch = "csordas:transformer"
    task_type = "seq2seq"
    default_exclude_patterns = ("embedding", "pos_encoding")

    def __init__(
        self,
        n_input_tokens: int,
        n_output_tokens: int,
        state_size: int = 128,
        n_heads: int = 4,
        n_encoder_layers: int = 3,
        n_decoder_layers: int = 3,
        ff_multiplier: float = 2.0,
        dropout: float = 0.1,
        max_len: int = 512,
        name: Optional[str] = None,
        device: Optional[torch.device | str] = None,
    ) -> None:
        super().__init__(name=name or "CsordasTransformerEncDec", device=device)
        self.input_embedding = nn.Embedding(n_input_tokens, state_size)
        self.output_embedding = nn.Embedding(n_output_tokens, state_size)
        self.pos_encoding = _PositionalEncoding(state_size, max_len=max_len)
        self.transformer = nn.Transformer(
            d_model=state_size,
            nhead=n_heads,
            num_encoder_layers=n_encoder_layers,
            num_decoder_layers=n_decoder_layers,
            dim_feedforward=int(state_size * ff_multiplier),
            dropout=dropout,
            batch_first=True,
        )
        self.output_map = nn.Linear(state_size, n_output_tokens)
        if device is not None:
            self.to(device)

    def forward(
        self,
        src: torch.Tensor,
        tgt: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if tgt is None:
            tgt = src
        src_emb = self.pos_encoding(self.input_embedding(src))
        tgt_emb = self.pos_encoding(self.output_embedding(tgt))
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt.size(1), device=tgt.device
        )
        out = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return self.output_map(out)
