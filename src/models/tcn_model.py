from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        residual = x if self.downsample is None else self.downsample(x)
        return self.activation(out + residual)


class TCNClassifier(nn.Module):
    def __init__(
        self,
        input_channels: int,
        channels: list[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers = []
        in_channels = input_channels
        for i, out_channels in enumerate(channels):
            layers.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=2 ** i,
                    dropout=dropout,
                )
            )
            in_channels = out_channels
        self.network = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(channels[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("TCNClassifier expects input shape (batch, seq_len, n_features).")
        x = x.transpose(1, 2)
        features = self.network(x)
        pooled = self.pool(features).squeeze(-1)
        return self.classifier(pooled).squeeze(-1)


def build_tcn_model(
    config: Dict[str, Any],
    input_shape: Optional[Tuple[int, ...]] = None,
) -> TCNClassifier:
    if input_shape is None or len(input_shape) != 2:
        raise ValueError("input_shape must be a tuple: (sequence_length, n_features)")
    _, n_features = input_shape
    tcn_cfg = config.get("tcn", {})
    channels = tcn_cfg.get("channels", [32, 32, 32])
    channels = [int(c) for c in channels]
    if not channels:
        raise ValueError("tcn.channels must contain at least one channel size.")
    return TCNClassifier(
        input_channels=int(n_features),
        channels=channels,
        kernel_size=int(tcn_cfg.get("kernel_size", 3)),
        dropout=float(tcn_cfg.get("dropout", 0.2)),
    )
