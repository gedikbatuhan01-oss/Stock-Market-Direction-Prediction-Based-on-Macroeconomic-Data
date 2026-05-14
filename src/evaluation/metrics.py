from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute classification metrics for binary bull/bear prediction.

    Returns a dictionary with:
    - accuracy
    - balanced_accuracy
    - f1
    - precision
    - recall
    - mcc
    - roc_auc (if probabilities are provided and valid)
    - confusion_matrix
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D arrays.")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    results: Dict[str, Any] = {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        if y_prob.ndim != 1:
            raise ValueError("y_prob must be a 1D array when provided.")
        if len(y_prob) != len(y_true):
            raise ValueError("y_prob and y_true must have the same length.")

        results["probability_mean"] = float(np.mean(y_prob))
        results["probability_std"] = float(np.std(y_prob))

        try:
            results["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            results["roc_auc"] = None
    else:
        results["roc_auc"] = None

    return results


def summarize_fold_metrics(fold_metrics: list[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate fold metrics using mean and std for numeric scalar fields.
    """
    if not fold_metrics:
        raise ValueError("fold_metrics must contain at least one fold result.")

    numeric_keys = [
        "accuracy",
        "balanced_accuracy",
        "f1",
        "precision",
        "recall",
        "mcc",
        "roc_auc",
    ]

    summary: Dict[str, Any] = {
        "n_folds": len(fold_metrics),
        "folds": fold_metrics,
    }

    for key in numeric_keys:
        values = [fm[key] for fm in fold_metrics if fm.get(key) is not None]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))
        else:
            summary[f"{key}_mean"] = None
            summary[f"{key}_std"] = None

    return summary
