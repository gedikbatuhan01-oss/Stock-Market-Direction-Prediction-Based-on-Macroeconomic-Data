from __future__ import annotations

from typing import List, Tuple
import pandas as pd


def split_dev_test(df: pd.DataFrame, test_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe chronologically into development and final test sets.
    """
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1.")

    n_rows = len(df)
    if n_rows < 2:
        raise ValueError("Dataframe must contain at least 2 rows.")

    test_size = max(1, int(round(n_rows * test_ratio)))
    test_size = min(test_size, n_rows - 1)

    split_idx = n_rows - test_size
    dev_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)

    return dev_df, test_df


def make_expanding_folds(
    n_rows: int,
    n_folds: int,
    val_ratio_within_dev: float,
) -> List[Tuple[range, range]]:
    """
    Create chronological expanding-window train/validation index ranges.

    Strategy:
    - validation block size is fixed
    - first train block is the earliest remaining chunk
    - each subsequent fold expands train up to the next validation block
    """
    if n_rows < 3:
        raise ValueError("n_rows must be at least 3.")
    if n_folds < 1:
        raise ValueError("n_folds must be at least 1.")
    if not 0 < val_ratio_within_dev < 1:
        raise ValueError("val_ratio_within_dev must be between 0 and 1.")

    val_size = max(1, int(round(n_rows * val_ratio_within_dev)))

    if n_folds * val_size >= n_rows:
        raise ValueError(
            "Not enough rows for the requested number of folds and validation ratio."
        )

    initial_train_size = n_rows - (n_folds * val_size)
    if initial_train_size < 1:
        raise ValueError("Initial train size must be at least 1.")

    folds: List[Tuple[range, range]] = []

    for fold_idx in range(n_folds):
        train_end = initial_train_size + (fold_idx * val_size)
        val_start = train_end
        val_end = val_start + val_size

        train_range = range(0, train_end)
        val_range = range(val_start, val_end)
        folds.append((train_range, val_range))

    validate_fold_boundaries(folds, n_rows)
    return folds


def validate_fold_boundaries(folds: List[Tuple[range, range]], n_rows: int) -> None:
    """Validate fold ordering, non-overlap, and in-range boundaries."""
    for i, (train_range, val_range) in enumerate(folds, start=1):
        if len(train_range) == 0:
            raise ValueError(f"Fold {i}: train range is empty.")
        if len(val_range) == 0:
            raise ValueError(f"Fold {i}: validation range is empty.")

        train_start, train_end = train_range.start, train_range.stop
        val_start, val_end = val_range.start, val_range.stop

        if not (0 <= train_start < train_end <= n_rows):
            raise ValueError(f"Fold {i}: invalid train range {train_range}.")
        if not (0 <= val_start < val_end <= n_rows):
            raise ValueError(f"Fold {i}: invalid validation range {val_range}.")
        if train_end > val_start:
            raise ValueError(f"Fold {i}: train and validation overlap.")
