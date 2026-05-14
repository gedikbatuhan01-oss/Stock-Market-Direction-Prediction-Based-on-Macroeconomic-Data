import numpy as np
import pandas as pd

from src.data.labeling import compute_forward_return, compute_threshold
from src.run_experiment import (
    build_labeled_sequences_for_endpoints,
    make_labelable_endpoint_indices,
)
from src.training.trainer import optimize_probability_threshold


def test_endpoint_sequences_can_use_past_rows_across_split_boundary():
    df = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=10, freq="D"),
        "feature": np.arange(10, dtype=float),
        "close": np.arange(100, 110, dtype=float),
    })

    X, y, timestamps, label_summary, diagnostics = build_labeled_sequences_for_endpoints(
        context_df=df,
        feature_cols=["feature"],
        close_col="close",
        endpoint_indices=range(6, 10),
        threshold=0.0,
        lookback=3,
        horizon=1,
        bull_label=1,
        bear_label=0,
        neutral_label=-1,
    )

    assert X.shape == (3, 3, 1)
    assert X[0, :, 0].tolist() == [4.0, 5.0, 6.0]
    assert diagnostics["endpoint_indices"][0] == 6
    assert str(timestamps.iloc[0].date()) == "2024-01-07"
    assert y.tolist() == [1, 1, 1]
    assert label_summary.loc[1, "count"] == 3


def test_labelable_endpoint_indices_keep_label_horizon_inside_split():
    endpoints = make_labelable_endpoint_indices(range(10, 15), horizon=1)
    assert list(endpoints) == [10, 11, 12, 13]

    endpoints = make_labelable_endpoint_indices(range(10, 15), horizon=2)
    assert list(endpoints) == [10, 11, 12]


def test_threshold_is_computed_from_train_slice_only():
    train_df = pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0]})
    val_df = pd.DataFrame({"close": [103.0, 200.0, 50.0]})

    train_threshold = compute_threshold(
        train_returns=compute_forward_return(train_df["close"], horizon=1),
        method="quantile",
        quantile=0.5,
    )
    leaked_threshold = compute_threshold(
        train_returns=compute_forward_return(pd.concat([train_df, val_df], ignore_index=True)["close"], horizon=1),
        method="quantile",
        quantile=0.5,
    )

    assert train_threshold < 0.01
    assert leaked_threshold > train_threshold


def test_probability_threshold_search_selects_best_mcc_threshold():
    result = optimize_probability_threshold(
        y_true=np.asarray([0, 0, 1, 1]),
        y_prob=np.asarray([0.2, 0.4, 0.6, 0.8]),
        thresholds=np.asarray([0.3, 0.5, 0.7]),
        metric="mcc",
        default_threshold=0.5,
    )

    assert result["selected_threshold"] == 0.5
    assert result["selected_score"] == 1.0
    assert len(result["curve"]) == 3
