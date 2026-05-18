from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class CNN1DClassifier(nn.Module):
    """
    Minimal 1D CNN binary classifier for sequence data.

    Expected input:
        x -> (batch_size, sequence_length, n_features)

    Internal conv format:
        (batch_size, n_features, sequence_length)
    """

    def __init__(
        self,
        input_channels: int,
        conv_channels: int = 32,
        kernel_size: int = 3,
        dropout: float = 0.2,
        num_conv_layers: int = 2,
    ) -> None:
        super().__init__()

        if num_conv_layers < 1:
            raise ValueError("num_conv_layers must be at least 1.")

        padding = kernel_size // 2

        layers = []
        in_ch = input_channels
        for _ in range(num_conv_layers):
            layers.extend([
                nn.Conv1d(in_ch, conv_channels, kernel_size=kernel_size, padding=padding),
                nn.ReLU(),
                nn.BatchNorm1d(conv_channels),
            ])
            in_ch = conv_channels
        layers.append(nn.AdaptiveAvgPool1d(1))
        self.feature_extractor = nn.Sequential(*layers)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(conv_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns raw logits for BCEWithLogitsLoss.
        """
        if x.ndim != 3:
            raise ValueError("CNN1DClassifier expects input shape (batch, seq_len, n_features).")

        x = x.transpose(1, 2)  # (batch, n_features, seq_len)
        features = self.feature_extractor(x).squeeze(-1)
        features = self.dropout(features)
        logits = self.classifier(features).squeeze(-1)
        return logits


def build_cnn1d_model(
    config: Dict[str, Any],
    input_shape: Optional[Tuple[int, ...]] = None,
) -> CNN1DClassifier:
    """
    Build CNN1D model from config.

    input_shape is expected to be:
        (sequence_length, n_features)
    """
    if input_shape is None or len(input_shape) != 2:
        raise ValueError("input_shape must be a tuple: (sequence_length, n_features)")

    _, n_features = input_shape
    cnn_cfg = config.get("cnn1d", {})

    model = CNN1DClassifier(
        input_channels=int(n_features),
        conv_channels=int(cnn_cfg.get("conv_channels", 32)),
        kernel_size=int(cnn_cfg.get("kernel_size", 3)),
        dropout=float(cnn_cfg.get("dropout", 0.2)),
        num_conv_layers=int(cnn_cfg.get("num_conv_layers", 2)),
    )
    return model
