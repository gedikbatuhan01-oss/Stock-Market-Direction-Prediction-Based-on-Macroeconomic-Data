from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class GRUClassifier(nn.Module):
    """
    Minimal GRU binary classifier.

    Input:
        x -> (batch_size, sequence_length, n_features)

    Output:
        logits -> (batch_size,)
    """

    def __init__(
        self,
        input_size: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        effective_dropout = dropout if num_layers > 1 else 0.0

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns raw logits for BCEWithLogitsLoss.
        """
        output, hidden = self.gru(x)
        last_hidden = hidden[-1]
        last_hidden = self.dropout(last_hidden)
        logits = self.classifier(last_hidden).squeeze(-1)
        return logits


def build_gru_model(
    config: Dict[str, Any],
    input_shape: Optional[Tuple[int, ...]] = None,
) -> GRUClassifier:
    """
    Build GRU model from config.

    input_shape is expected to be:
        (sequence_length, n_features)
    """
    if input_shape is None or len(input_shape) != 2:
        raise ValueError("input_shape must be a tuple: (sequence_length, n_features)")

    _, n_features = input_shape
    gru_cfg = config.get("gru", {})

    model = GRUClassifier(
        input_size=int(n_features),
        hidden_dim=int(gru_cfg.get("hidden_dim", 64)),
        num_layers=int(gru_cfg.get("num_layers", 1)),
        dropout=float(gru_cfg.get("dropout", 0.2)),
    )
    return model
