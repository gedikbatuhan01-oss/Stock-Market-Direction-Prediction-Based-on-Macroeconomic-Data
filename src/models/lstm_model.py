from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("LSTMClassifier expects input shape (batch, seq_len, n_features).")
        _, (hidden, _) = self.lstm(x)
        if self.bidirectional:
            features = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            features = hidden[-1]
        features = self.dropout(features)
        return self.classifier(features).squeeze(-1)


def build_lstm_model(
    config: Dict[str, Any],
    input_shape: Optional[Tuple[int, ...]] = None,
) -> LSTMClassifier:
    if input_shape is None or len(input_shape) != 2:
        raise ValueError("input_shape must be a tuple: (sequence_length, n_features)")
    _, n_features = input_shape
    lstm_cfg = config.get("lstm", {})
    return LSTMClassifier(
        input_size=int(n_features),
        hidden_dim=int(lstm_cfg.get("hidden_dim", 64)),
        num_layers=int(lstm_cfg.get("num_layers", 1)),
        dropout=float(lstm_cfg.get("dropout", 0.2)),
        bidirectional=bool(lstm_cfg.get("bidirectional", False)),
    )
