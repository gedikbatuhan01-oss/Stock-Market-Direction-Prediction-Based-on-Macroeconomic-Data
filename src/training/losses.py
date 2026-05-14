from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def compute_pos_weight_from_labels(y: np.ndarray) -> float:
    """
    Compute positive-class weight for BCEWithLogitsLoss.

    pos_weight = n_negative / n_positive
    """
    y = np.asarray(y).astype(np.int64)

    n_positive = int((y == 1).sum())
    n_negative = int((y == 0).sum())

    if n_positive == 0:
        raise ValueError("Cannot compute pos_weight: there are no positive samples.")
    if n_negative == 0:
        raise ValueError("Cannot compute pos_weight: there are no negative samples.")

    return float(n_negative / n_positive)


def build_bce_with_logits_loss(
    y_train: Optional[np.ndarray] = None,
    use_weighted_loss: bool = True,
    device: str = "cpu",
) -> nn.Module:
    """
    Build BCEWithLogitsLoss, optionally with positive-class weighting.
    """
    if use_weighted_loss and y_train is not None:
        pos_weight_value = compute_pos_weight_from_labels(y_train)
        pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    return nn.BCEWithLogitsLoss()
