from __future__ import annotations

import pandas as pd


def compute_forward_return(close_series: pd.Series, horizon: int = 1) -> pd.Series:
    """
    Compute forward return from t to t+horizon:
        (Close[t+horizon] / Close[t]) - 1
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")

    returns = close_series.shift(-horizon) / close_series - 1.0
    returns.name = "forward_return"
    return returns


def compute_threshold(
    train_returns: pd.Series,
    method: str = "quantile",
    quantile: float = 0.40,
) -> float:
    """
    Compute labeling threshold using train-only returns.

    Supported methods:
    - "quantile": threshold = quantile(|train_returns|)
    """
    valid_returns = train_returns.dropna().abs()

    if len(valid_returns) == 0:
        raise ValueError("train_returns contains no valid values for threshold computation.")

    if method != "quantile":
        raise ValueError(f"Unsupported threshold method: {method}")

    if not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1.")

    threshold = float(valid_returns.quantile(quantile))
    return threshold


def make_labels(
    returns: pd.Series,
    threshold: float,
    bull_label: int = 1,
    bear_label: int = 0,
    neutral_label: int = -1,
) -> pd.Series:
    """
    Create bull/bear/neutral labels from forward returns and threshold.

    Rules:
    - return > +threshold -> bull_label
    - return < -threshold -> bear_label
    - otherwise -> neutral_label
    """
    if threshold < 0:
        raise ValueError("threshold must be non-negative.")

    labels = pd.Series(neutral_label, index=returns.index, dtype="int64")

    labels.loc[returns > threshold] = bull_label
    labels.loc[returns < -threshold] = bear_label

    # NaN forward returns remain neutral by design.
    labels.name = "label"
    return labels


def summarize_label_distribution(labels: pd.Series) -> pd.DataFrame:
    """Return label counts and proportions for diagnostics."""
    counts = labels.value_counts(dropna=False).sort_index()
    proportions = labels.value_counts(dropna=False, normalize=True).sort_index()

    summary = pd.DataFrame({
        "count": counts,
        "proportion": proportions,
    })
    summary.index.name = "label"
    return summary
