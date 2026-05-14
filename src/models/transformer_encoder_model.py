from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class TransformerEncoderClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        sequence_length: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("transformer_encoder.d_model must be divisible by nhead.")
        self.pooling = pooling
        self.input_projection = nn.Linear(n_features, d_model)
        self.positional_embedding = nn.Parameter(torch.zeros(1, sequence_length, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("TransformerEncoderClassifier expects input shape (batch, seq_len, n_features).")
        if x.shape[1] > self.positional_embedding.shape[1]:
            raise ValueError("Input sequence is longer than the configured positional embedding length.")
        x = self.input_projection(x)
        x = x + self.positional_embedding[:, : x.shape[1], :]
        encoded = self.encoder(x)
        if self.pooling == "last":
            features = encoded[:, -1, :]
        else:
            features = encoded.mean(dim=1)
        features = self.dropout(features)
        return self.classifier(features).squeeze(-1)


def build_transformer_encoder_model(
    config: Dict[str, Any],
    input_shape: Optional[Tuple[int, ...]] = None,
) -> TransformerEncoderClassifier:
    if input_shape is None or len(input_shape) != 2:
        raise ValueError("input_shape must be a tuple: (sequence_length, n_features)")
    sequence_length, n_features = input_shape
    tr_cfg = config.get("transformer_encoder", {})
    return TransformerEncoderClassifier(
        n_features=int(n_features),
        sequence_length=int(sequence_length),
        d_model=int(tr_cfg.get("d_model", 64)),
        nhead=int(tr_cfg.get("nhead", 4)),
        num_layers=int(tr_cfg.get("num_layers", 2)),
        dim_feedforward=int(tr_cfg.get("dim_feedforward", 128)),
        dropout=float(tr_cfg.get("dropout", 0.2)),
        pooling=tr_cfg.get("pooling", "mean"),
    )
