from __future__ import annotations

from typing import Iterable, List, Tuple
import numpy as np
import pandas as pd


def build_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    labels: pd.Series,
    lookback: int,
) -> Tuple[np.ndarray, np.ndarray, pd.Series]:
    """
    Build chronological feature windows and aligned labels.

    Alignment rule:
    - X window covers [t-lookback+1, ..., t]
    - y is the label at time t (which represents the move from t to t+1)
    """
    if lookback < 1:
        raise ValueError("lookback must be at least 1.")
    if len(df) != len(labels):
        raise ValueError("df and labels must have the same length.")
    if any(col not in df.columns for col in feature_cols):
        missing = [col for col in feature_cols if col not in df.columns]
        raise ValueError(f"Missing feature columns: {missing}")
    if len(df) < lookback:
        raise ValueError("Not enough rows to build at least one sequence.")

    feature_matrix = df[feature_cols].to_numpy(dtype=np.float32)
    label_array = labels.to_numpy()

    if "Date" in df.columns:
        full_timestamps = pd.to_datetime(df["Date"]).reset_index(drop=True)
    else:
        full_timestamps = pd.Series(df.index)

    X_list = []
    y_list = []
    ts_list = []

    for end_idx in range(lookback - 1, len(df)):
        start_idx = end_idx - lookback + 1
        window = feature_matrix[start_idx : end_idx + 1]
        target = label_array[end_idx]
        timestamp = full_timestamps.iloc[end_idx]

        X_list.append(window)
        y_list.append(target)
        ts_list.append(timestamp)

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    timestamps = pd.Series(ts_list, name="timestamp").reset_index(drop=True)

    validate_sequence_alignment(X, y, timestamps)
    return X, y, timestamps


def build_sequences_for_endpoints(
    df: pd.DataFrame,
    feature_cols: List[str],
    labels: pd.Series,
    endpoint_indices: Iterable[int],
    lookback: int,
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """
    Build windows whose label endpoint is explicitly provided.

    This is used for walk-forward validation: validation/test endpoints can use
    historical feature rows from earlier splits, while the label endpoint itself
    remains inside the evaluated split.
    """
    if lookback < 1:
        raise ValueError("lookback must be at least 1.")
    if len(df) != len(labels):
        raise ValueError("df and labels must have the same length.")
    if any(col not in df.columns for col in feature_cols):
        missing = [col for col in feature_cols if col not in df.columns]
        raise ValueError(f"Missing feature columns: {missing}")

    endpoint_list = [int(idx) for idx in endpoint_indices]
    if not endpoint_list:
        raise ValueError("endpoint_indices must contain at least one index.")

    feature_matrix = df[feature_cols].to_numpy(dtype=np.float32)
    label_array = labels.to_numpy()

    if "Date" in df.columns:
        full_timestamps = pd.to_datetime(df["Date"]).reset_index(drop=True)
    else:
        full_timestamps = pd.Series(df.index)

    X_list = []
    y_list = []
    ts_list = []
    kept_endpoint_indices = []

    for end_idx in endpoint_list:
        if end_idx < lookback - 1:
            continue
        if end_idx >= len(df):
            raise ValueError(f"Endpoint index out of bounds: {end_idx}")

        start_idx = end_idx - lookback + 1
        window = feature_matrix[start_idx : end_idx + 1]
        target = label_array[end_idx]
        timestamp = full_timestamps.iloc[end_idx]

        X_list.append(window)
        y_list.append(target)
        ts_list.append(timestamp)
        kept_endpoint_indices.append(end_idx)

    if not X_list:
        raise ValueError("No valid sequences could be built for the provided endpoints.")

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    timestamps = pd.Series(ts_list, name="timestamp").reset_index(drop=True)
    endpoints = pd.Series(kept_endpoint_indices, name="endpoint_index").reset_index(drop=True)

    validate_sequence_alignment(X, y, timestamps)
    return X, y, timestamps, endpoints


def drop_neutral_sequences(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: pd.Series,
    neutral_value: int = -1,
) -> Tuple[np.ndarray, np.ndarray, pd.Series]:
    """Remove samples whose labels are neutral."""
    if not (len(X) == len(y) == len(timestamps)):
        raise ValueError("X, y, and timestamps must have the same number of samples.")

    keep_mask = y != neutral_value

    X_filtered = X[keep_mask]
    y_filtered = y[keep_mask]
    timestamps_filtered = timestamps.loc[keep_mask].reset_index(drop=True)

    return X_filtered, y_filtered, timestamps_filtered


def flatten_sequences(X_seq: np.ndarray) -> np.ndarray:
    """Flatten 3D sequence tensor into 2D matrix for tabular models."""
    if X_seq.ndim != 3:
        raise ValueError("X_seq must be a 3D array of shape (n_samples, lookback, n_features).")
    return X_seq.reshape(X_seq.shape[0], -1)


def validate_sequence_alignment(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: pd.Series,
) -> None:
    """Run basic checks on sequence-label alignment."""
    if X.ndim != 3:
        raise ValueError("X must be 3D: (n_samples, lookback, n_features).")
    if y.ndim != 1:
        raise ValueError("y must be 1D.")
    if len(X) != len(y):
        raise ValueError("Number of samples in X and y must match.")
    if len(y) != len(timestamps):
        raise ValueError("y and timestamps lengths must match.")
    if X.shape[1] < 1 or X.shape[2] < 1:
        raise ValueError("Sequence dimensions must be positive.")
