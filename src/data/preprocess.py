from __future__ import annotations

from typing import List
import pandas as pd


def validate_required_columns(df: pd.DataFrame, required_cols: List[str]) -> None:
    """Raise error if any required columns are missing."""
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


def apply_missing_value_policy(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    Apply missing value handling policy.

    Supported methods:
    - "ffill_then_drop_head": forward-fill, then drop any remaining NaNs
    """
    if method != "ffill_then_drop_head":
        raise ValueError(f"Unsupported missing value policy: {method}")

    cleaned = df.copy()
    cleaned = cleaned.ffill()
    cleaned = cleaned.dropna(axis=0).reset_index(drop=True)
    return cleaned


def select_columns(
    df: pd.DataFrame,
    feature_cols: List[str],
    close_col: str,
    date_col: str,
) -> pd.DataFrame:
    """Keep only required date, close, and feature columns."""
    required_cols = [date_col, close_col] + feature_cols
    validate_required_columns(df, required_cols)

    selected = df[required_cols].copy()
    return selected
