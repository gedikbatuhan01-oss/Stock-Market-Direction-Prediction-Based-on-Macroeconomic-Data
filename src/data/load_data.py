from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_market_data(file_path: str | Path, date_col: str) -> pd.DataFrame:
    """Load CSV, parse date column, sort chronologically, and reset index."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    df = pd.read_csv(file_path, parse_dates=[date_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)

    assert_chronological_order(df, date_col)
    return df


def assert_chronological_order(df: pd.DataFrame, date_col: str) -> None:
    """Raise error if dataframe is not sorted increasingly by date column."""
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in dataframe.")

    dates = df[date_col]
    if not dates.is_monotonic_increasing:
        raise ValueError(
            f"Data is not sorted chronologically by '{date_col}'. "
            f"First date: {dates.iloc[0]}, Last date: {dates.iloc[-1]}"
        )
