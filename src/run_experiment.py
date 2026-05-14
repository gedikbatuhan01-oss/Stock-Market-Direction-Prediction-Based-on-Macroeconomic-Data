from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.data.labeling import (
    compute_forward_return,
    compute_threshold,
    make_labels,
    summarize_label_distribution,
)
from src.data.load_data import load_market_data
from src.data.preprocess import apply_missing_value_policy, select_columns
from src.data.sequence_builder import (
    build_sequences_for_endpoints,
    flatten_sequences,
)
from src.data.splitters import make_expanding_folds, split_dev_test
from src.evaluation.metrics import summarize_fold_metrics
from src.evaluation.metrics import compute_classification_metrics
from src.evaluation.reports import (
    append_experiment_summary,
    save_experiment_report,
    save_fold_results,
)
from src.models.model_factory import (
    build_model,
    get_model_family,
    requires_sequence_input,
)
from src.training.trainer import (
    build_probability_threshold_grid,
    evaluate_sklearn_model,
    evaluate_torch_model,
    optimize_probability_threshold,
    predict_proba,
    predict_proba_torch_model,
    run_single_torch_fold,
    train_sklearn_model,
    extract_sklearn_training_history,
)
from src.utils.config import load_config
from src.utils.io import ensure_dir
from src.utils.seed import set_global_seed


def get_scaler(scaler_name: str):
    scaler_name = scaler_name.lower().strip()
    if scaler_name == "standard":
        return StandardScaler()
    if scaler_name == "robust":
        return RobustScaler()
    raise ValueError(f"Unsupported scaler: {scaler_name}")


def scale_sequence_data(
    X_train_seq: np.ndarray,
    X_other_seq: np.ndarray,
    scaler_name: str,
) -> Tuple[np.ndarray, np.ndarray, Any]:
    """
    Fit scaler on train sequences only, then transform train and other sequences.
    Scaling is applied feature-wise after stacking all time steps from train sequences.
    """
    if X_train_seq.ndim != 3 or X_other_seq.ndim != 3:
        raise ValueError("Both X_train_seq and X_other_seq must be 3D arrays.")

    scaler = get_scaler(scaler_name)

    n_train, lookback, n_features = X_train_seq.shape
    n_other = X_other_seq.shape[0]

    train_2d = X_train_seq.reshape(-1, n_features)
    other_2d = X_other_seq.reshape(-1, n_features)

    scaler.fit(train_2d)

    train_scaled = scaler.transform(train_2d).reshape(n_train, lookback, n_features).astype(np.float32)
    other_scaled = scaler.transform(other_2d).reshape(n_other, lookback, n_features).astype(np.float32)

    return train_scaled, other_scaled, scaler


def prepare_model_inputs(X_seq_scaled: np.ndarray, model_name: str) -> np.ndarray:
    """
    Route model inputs by model type:
    - logreg  -> flattened 2D
    - gru     -> 3D sequence
    - cnn1d   -> 3D sequence
    """
    if requires_sequence_input(model_name):
        return X_seq_scaled
    return flatten_sequences(X_seq_scaled)


def get_weighted_loss_flag(config: Dict[str, Any], model_name: str) -> bool:
    model_cfg = config.get(model_name, {})
    return bool(model_cfg.get("weighted_loss", True))


def make_labelable_endpoint_indices(index_range: range, horizon: int) -> range:
    """
    Keep label endpoints inside a split while allowing feature lookback to use
    earlier rows. The endpoint t is included only when t+horizon remains in the
    same split range.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")

    stop = index_range.stop - horizon
    if stop <= index_range.start:
        return range(index_range.start, index_range.start)
    return range(index_range.start, stop)


def build_labeled_sequences_for_endpoints(
    context_df: pd.DataFrame,
    feature_cols: List[str],
    close_col: str,
    endpoint_indices: range,
    threshold: float,
    lookback: int,
    horizon: int,
    bull_label: int,
    bear_label: int,
    neutral_label: int,
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.DataFrame, Dict[str, Any]]:
    """
    Compute labels on the chronological context and build endpoint-based windows.

    Feature windows may reach backward before the endpoint split, but label
    endpoints are supplied by the caller and must remain inside the split.
    """
    labelable_endpoints = make_labelable_endpoint_indices(endpoint_indices, horizon=horizon)
    if len(labelable_endpoints) == 0:
        raise ValueError("No labelable endpoints remain after applying horizon.")

    returns = compute_forward_return(context_df[close_col], horizon=horizon)
    labels = make_labels(
        returns=returns,
        threshold=threshold,
        bull_label=bull_label,
        bear_label=bear_label,
        neutral_label=neutral_label,
    )

    X_seq, y, timestamps, endpoints = build_sequences_for_endpoints(
        df=context_df,
        feature_cols=feature_cols,
        labels=labels,
        endpoint_indices=labelable_endpoints,
        lookback=lookback,
    )

    all_label_summary = summarize_label_distribution(pd.Series(y))

    keep_mask = y != neutral_label
    X_seq = X_seq[keep_mask]
    y = y[keep_mask]
    timestamps = timestamps.loc[keep_mask].reset_index(drop=True)
    endpoints = endpoints.loc[keep_mask].reset_index(drop=True)

    kept_label_summary = summarize_label_distribution(pd.Series(y))
    label_diagnostics = {
        "all_endpoint_labels": all_label_summary.reset_index().to_dict(orient="records"),
        "kept_binary_labels": kept_label_summary.reset_index().to_dict(orient="records"),
        "n_neutral_dropped": int((~keep_mask).sum()),
        "n_endpoints_before_neutral_drop": int(len(keep_mask)),
        "endpoint_indices": endpoints.astype(int).tolist(),
    }
    return X_seq, y, timestamps, kept_label_summary, label_diagnostics


def get_threshold_search_grid(config: Dict[str, Any]) -> np.ndarray:
    decision_cfg = config.get("decision", {})
    search_cfg = decision_cfg.get("threshold_search", {})
    if not bool(search_cfg.get("enabled", False)):
        return np.asarray([float(decision_cfg.get("probability_threshold", 0.50))])

    return build_probability_threshold_grid(
        start=float(search_cfg.get("start", 0.05)),
        stop=float(search_cfg.get("stop", 0.95)),
        step=float(search_cfg.get("step", 0.01)),
    )


def choose_probability_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    decision_cfg = config.get("decision", {})
    search_cfg = decision_cfg.get("threshold_search", {})
    default_threshold = float(decision_cfg.get("probability_threshold", 0.50))

    if not bool(search_cfg.get("enabled", False)):
        return {
            "selected_threshold": default_threshold,
            "selected_metric": None,
            "selected_score": None,
            "default_threshold": default_threshold,
            "curve": [],
        }

    return optimize_probability_threshold(
        y_true=y_true,
        y_prob=y_prob,
        thresholds=get_threshold_search_grid(config),
        metric=str(search_cfg.get("metric", "mcc")),
        default_threshold=default_threshold,
    )


def compute_naive_baselines(
    context_df: pd.DataFrame,
    y_true: np.ndarray,
    endpoint_indices: List[int],
    close_col: str,
    seed: int,
) -> Dict[str, Any]:
    """Evaluate non-learning baselines on the same endpoint set as the model."""
    y_true = np.asarray(y_true)
    endpoints = [int(idx) for idx in endpoint_indices]
    rng = np.random.default_rng(seed)

    baseline_specs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    always_bull = np.ones_like(y_true, dtype=np.int64)
    baseline_specs["always_bull"] = (always_bull, always_bull.astype(float))

    always_bear = np.zeros_like(y_true, dtype=np.int64)
    baseline_specs["always_bear"] = (always_bear, always_bear.astype(float))

    random_pred = rng.integers(0, 2, size=len(y_true), dtype=np.int64)
    baseline_specs["random"] = (random_pred, random_pred.astype(float))

    close_values = context_df[close_col].to_numpy(dtype=float)
    previous_sign_pred = []
    for endpoint in endpoints:
        if endpoint <= 0:
            previous_sign_pred.append(1)
            continue
        previous_return = (close_values[endpoint] / close_values[endpoint - 1]) - 1.0
        previous_sign_pred.append(1 if previous_return > 0 else 0)
    previous_sign = np.asarray(previous_sign_pred, dtype=np.int64)
    baseline_specs["previous_day_sign"] = (previous_sign, previous_sign.astype(float))

    results = {}
    for name, (y_pred, y_prob) in baseline_specs.items():
        results[name] = compute_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
        )
    return results


def run_cv_for_model_and_scaler(
    config: Dict[str, Any],
    dev_df: pd.DataFrame,
    model_name: str,
    scaler_name: str,
) -> Dict[str, Any]:
    """
    Run expanding-window CV for one (model, scaler) pair.
    Supports both sklearn and torch models.
    """
    split_cfg = config["split"]
    data_cfg = config["data"]
    labeling_cfg = config["labeling"]
    seq_cfg = config["sequence"]
    decision_cfg = config["decision"]
    training_cfg = config["training"]

    model_family = get_model_family(model_name)

    folds = make_expanding_folds(
        n_rows=len(dev_df),
        n_folds=int(split_cfg["n_folds"]),
        val_ratio_within_dev=float(split_cfg["val_ratio_within_dev"]),
    )

    fold_results: List[Dict[str, Any]] = []

    for fold_idx, (train_range, val_range) in enumerate(folds, start=1):
        train_df = dev_df.iloc[list(train_range)].copy().reset_index(drop=True)

        train_returns = compute_forward_return(
            train_df[data_cfg["close_col"]],
            horizon=int(labeling_cfg["horizon"]),
        )

        threshold = compute_threshold(
            train_returns=train_returns,
            method=labeling_cfg["threshold_method"],
            quantile=float(labeling_cfg["threshold_quantile"]),
        )

        X_train_seq, y_train, ts_train, train_label_summary, train_label_diagnostics = build_labeled_sequences_for_endpoints(
            context_df=dev_df,
            feature_cols=data_cfg["feature_cols"],
            close_col=data_cfg["close_col"],
            endpoint_indices=train_range,
            threshold=threshold,
            lookback=int(seq_cfg["lookback"]),
            horizon=int(labeling_cfg["horizon"]),
            bull_label=int(labeling_cfg["bull_label"]),
            bear_label=int(labeling_cfg["bear_label"]),
            neutral_label=int(labeling_cfg["neutral_label"]),
        )

        X_val_seq, y_val, ts_val, val_label_summary, val_label_diagnostics = build_labeled_sequences_for_endpoints(
            context_df=dev_df,
            feature_cols=data_cfg["feature_cols"],
            close_col=data_cfg["close_col"],
            endpoint_indices=val_range,
            threshold=threshold,
            lookback=int(seq_cfg["lookback"]),
            horizon=int(labeling_cfg["horizon"]),
            bull_label=int(labeling_cfg["bull_label"]),
            bear_label=int(labeling_cfg["bear_label"]),
            neutral_label=int(labeling_cfg["neutral_label"]),
        )

        if len(y_train) == 0:
            raise ValueError(f"Fold {fold_idx}: no train samples after neutral filtering.")
        if len(y_val) == 0:
            raise ValueError(f"Fold {fold_idx}: no validation samples after neutral filtering.")

        X_train_seq_scaled, X_val_seq_scaled, scaler = scale_sequence_data(
            X_train_seq=X_train_seq,
            X_other_seq=X_val_seq,
            scaler_name=scaler_name,
        )

        X_train_model = prepare_model_inputs(X_train_seq_scaled, model_name=model_name)
        X_val_model = prepare_model_inputs(X_val_seq_scaled, model_name=model_name)

        model = build_model(
            model_name=model_name,
            config=config,
            input_shape=tuple(X_train_seq_scaled.shape[1:]),
        )

        if model_family == "sklearn":
            X_train_fit, y_train_fit, X_threshold, y_threshold = split_internal_final_train_val(
                X_dev_model=X_train_model,
                y_dev=y_train,
                val_ratio=float(split_cfg["val_ratio_within_dev"]),
            )
            trained_model = train_sklearn_model(
                model=model,
                X_train=X_train_fit,
                y_train=y_train_fit,
            )
            threshold_prob = predict_proba(trained_model, X_threshold)
            threshold_info = choose_probability_threshold(
                y_true=y_threshold,
                y_prob=threshold_prob,
                config=config,
            )
            selected_probability_threshold = float(threshold_info["selected_threshold"])
            eval_model = build_model(
                model_name=model_name,
                config=config,
                input_shape=tuple(X_train_seq_scaled.shape[1:]),
            )
            trained_model = train_sklearn_model(eval_model, X_train_model, y_train)
            val_results = evaluate_sklearn_model(
                model=trained_model,
                X=X_val_model,
                y_true=y_val,
                probability_threshold=selected_probability_threshold,
            )

            fold_metrics = val_results["metrics"]
            trainer_info = {
                "trainer_type": "sklearn",
                "best_epoch": None,
                "best_score": None,
                "history": extract_sklearn_training_history(trained_model),
                "internal_train_size": int(len(y_train_fit)),
                "threshold_source": "fold_train_internal_validation",
                "threshold_optimization": threshold_info,
            }

        elif model_family == "torch":
            X_train_fit, y_train_fit, X_threshold, y_threshold = split_internal_final_train_val(
                X_dev_model=X_train_model,
                y_dev=y_train,
                val_ratio=float(split_cfg["val_ratio_within_dev"]),
            )
            fold_run = run_single_torch_fold(
                model=model,
                X_train=X_train_fit,
                y_train=y_train_fit,
                X_val=X_threshold,
                y_val=y_threshold,
                training_config=training_cfg,
                probability_threshold=float(decision_cfg["probability_threshold"]),
                use_weighted_loss=get_weighted_loss_flag(config, model_name),
                device=None,
            )

            threshold_prob = predict_proba_torch_model(
                model=fold_run["model"],
                X=X_threshold,
                batch_size=int(training_cfg.get("batch_size", 64)),
                device=fold_run.get("device", "cpu"),
            )
            threshold_info = choose_probability_threshold(
                y_true=y_threshold,
                y_prob=threshold_prob,
                config=config,
            )
            selected_probability_threshold = float(threshold_info["selected_threshold"])
            val_results = evaluate_torch_model(
                model=fold_run["model"],
                X=X_val_model,
                y_true=y_val,
                probability_threshold=selected_probability_threshold,
                batch_size=int(training_cfg.get("batch_size", 64)),
                device=fold_run.get("device", "cpu"),
            )
            fold_metrics = val_results["metrics"]
            trainer_info = {
                "trainer_type": "torch",
                "best_epoch": fold_run.get("best_epoch"),
                "best_score": fold_run.get("best_score"),
                "selected_learning_rate": fold_run.get("selected_learning_rate"),
                "learning_rate_candidates": fold_run.get("learning_rate_candidates"),
                "lr_search_results": fold_run.get("lr_search_results", []),
                "history": fold_run.get("history", []),
                "internal_train_size": int(len(y_train_fit)),
                "threshold_source": "fold_train_internal_validation",
                "threshold_optimization": threshold_info,
            }

        else:
            raise ValueError(f"Unsupported model family: {model_family}")

        fold_results.append({
            "fold_index": fold_idx,
            "model_name": model_name,
            "model_family": model_family,
            "scaler_name": scaler_name,
            "threshold": threshold,
            "label_threshold_quantile": float(labeling_cfg["threshold_quantile"]),
            "lookback": int(seq_cfg["lookback"]),
            "selected_probability_threshold": selected_probability_threshold,
            "n_train_samples": int(len(y_train)),
            "n_val_samples": int(len(y_val)),
            "train_start": str(ts_train.iloc[0]) if len(ts_train) else None,
            "train_end": str(ts_train.iloc[-1]) if len(ts_train) else None,
            "val_start": str(ts_val.iloc[0]) if len(ts_val) else None,
            "val_end": str(ts_val.iloc[-1]) if len(ts_val) else None,
            "train_label_distribution": train_label_summary.reset_index().to_dict(orient="records"),
            "val_label_distribution": val_label_summary.reset_index().to_dict(orient="records"),
            "train_label_diagnostics": train_label_diagnostics,
            "val_label_diagnostics": val_label_diagnostics,
            "metrics": fold_metrics,
            "baselines": compute_naive_baselines(
                context_df=dev_df,
                y_true=y_val,
                endpoint_indices=val_label_diagnostics["endpoint_indices"],
                close_col=data_cfg["close_col"],
                seed=int(config["experiment"]["seed"]) + fold_idx,
            ),
            "trainer_info": trainer_info,
        })

    metric_only = [fr["metrics"] for fr in fold_results]
    cv_summary = summarize_fold_metrics(metric_only)

    return {
        "model_name": model_name,
        "model_family": model_family,
        "scaler_name": scaler_name,
        "label_threshold_quantile": float(labeling_cfg["threshold_quantile"]),
        "lookback": int(seq_cfg["lookback"]),
        "fold_results": fold_results,
        "cv_summary": cv_summary,
    }


def select_best_cv_result(
    cv_results: List[Dict[str, Any]],
    primary_metric: str,
) -> Dict[str, Any]:
    if not cv_results:
        raise ValueError("No CV results to select from.")

    metric_key = f"{primary_metric}_mean"

    ranked = sorted(
        cv_results,
        key=lambda x: (
            float("-inf") if x["cv_summary"].get(metric_key) is None else x["cv_summary"][metric_key]
        ),
        reverse=True,
    )
    return ranked[0]


def split_internal_final_train_val(
    X_dev_model: np.ndarray,
    y_dev: np.ndarray,
    val_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Internal chronological split used only for torch final training,
    so early stopping can still operate without touching final test.
    """
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")

    n = len(y_dev)
    if n < 3:
        raise ValueError("Need at least 3 samples for internal final train/val split.")

    val_size = max(1, int(round(n * val_ratio)))
    val_size = min(val_size, n - 1)

    split_idx = n - val_size

    X_train_final = X_dev_model[:split_idx]
    y_train_final = y_dev[:split_idx]
    X_val_internal = X_dev_model[split_idx:]
    y_val_internal = y_dev[split_idx:]

    return X_train_final, y_train_final, X_val_internal, y_val_internal


def run_final_train_and_test(
    config: Dict[str, Any],
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str,
    scaler_name: str,
) -> Dict[str, Any]:
    """
    Train on full development set and evaluate once on final test.
    For torch models, a small internal tail of dev is reserved for early stopping.
    """
    data_cfg = config["data"]
    labeling_cfg = config["labeling"]
    seq_cfg = config["sequence"]
    decision_cfg = config["decision"]
    split_cfg = config["split"]
    training_cfg = config["training"]

    model_family = get_model_family(model_name)
    full_context_df = pd.concat([dev_df, test_df], axis=0, ignore_index=True)
    dev_range = range(0, len(dev_df))
    test_range = range(len(dev_df), len(full_context_df))

    dev_returns = compute_forward_return(
        dev_df[data_cfg["close_col"]],
        horizon=int(labeling_cfg["horizon"]),
    )

    final_threshold = compute_threshold(
        train_returns=dev_returns,
        method=labeling_cfg["threshold_method"],
        quantile=float(labeling_cfg["threshold_quantile"]),
    )

    X_dev_seq, y_dev, ts_dev, dev_label_summary, dev_label_diagnostics = build_labeled_sequences_for_endpoints(
        context_df=full_context_df,
        feature_cols=data_cfg["feature_cols"],
        close_col=data_cfg["close_col"],
        endpoint_indices=dev_range,
        threshold=final_threshold,
        lookback=int(seq_cfg["lookback"]),
        horizon=int(labeling_cfg["horizon"]),
        bull_label=int(labeling_cfg["bull_label"]),
        bear_label=int(labeling_cfg["bear_label"]),
        neutral_label=int(labeling_cfg["neutral_label"]),
    )

    X_test_seq, y_test, ts_test, test_label_summary, test_label_diagnostics = build_labeled_sequences_for_endpoints(
        context_df=full_context_df,
        feature_cols=data_cfg["feature_cols"],
        close_col=data_cfg["close_col"],
        endpoint_indices=test_range,
        threshold=final_threshold,
        lookback=int(seq_cfg["lookback"]),
        horizon=int(labeling_cfg["horizon"]),
        bull_label=int(labeling_cfg["bull_label"]),
        bear_label=int(labeling_cfg["bear_label"]),
        neutral_label=int(labeling_cfg["neutral_label"]),
    )

    if len(y_dev) == 0:
        raise ValueError("No development samples after neutral filtering.")
    if len(y_test) == 0:
        raise ValueError("No test samples after neutral filtering.")

    X_dev_seq_scaled, X_test_seq_scaled, scaler = scale_sequence_data(
        X_train_seq=X_dev_seq,
        X_other_seq=X_test_seq,
        scaler_name=scaler_name,
    )

    X_dev_model = prepare_model_inputs(X_dev_seq_scaled, model_name=model_name)
    X_test_model = prepare_model_inputs(X_test_seq_scaled, model_name=model_name)

    model = build_model(
        model_name=model_name,
        config=config,
        input_shape=tuple(X_dev_seq_scaled.shape[1:]),
    )

    final_train_info: Dict[str, Any] = {
        "model_family": model_family,
        "internal_validation_used": False,
        "best_epoch": None,
        "best_score": None,
    }

    if model_family == "sklearn":
        X_train_final, y_train_final, X_val_internal, y_val_internal = split_internal_final_train_val(
            X_dev_model=X_dev_model,
            y_dev=y_dev,
            val_ratio=float(split_cfg["val_ratio_within_dev"]),
        )
        threshold_model = train_sklearn_model(model, X_train_final, y_train_final)
        threshold_prob = predict_proba(threshold_model, X_val_internal)
        threshold_info = choose_probability_threshold(
            y_true=y_val_internal,
            y_prob=threshold_prob,
            config=config,
        )
        selected_probability_threshold = float(threshold_info["selected_threshold"])
        final_model = build_model(
            model_name=model_name,
            config=config,
            input_shape=tuple(X_dev_seq_scaled.shape[1:]),
        )
        trained_model = train_sklearn_model(final_model, X_dev_model, y_dev)

        test_results = evaluate_sklearn_model(
            model=trained_model,
            X=X_test_model,
            y_true=y_test,
            probability_threshold=selected_probability_threshold,
        )
        final_train_info["internal_validation_used"] = True
        final_train_info["internal_val_size"] = int(len(y_val_internal))
        final_train_info["threshold_source"] = "development_internal_validation"
        final_train_info["threshold_optimization"] = threshold_info

    elif model_family == "torch":
        X_train_final, y_train_final, X_val_internal, y_val_internal = split_internal_final_train_val(
            X_dev_model=X_dev_model,
            y_dev=y_dev,
            val_ratio=float(split_cfg["val_ratio_within_dev"]),
        )

        final_train_run = run_single_torch_fold(
            model=model,
            X_train=X_train_final,
            y_train=y_train_final,
            X_val=X_val_internal,
            y_val=y_val_internal,
            training_config=training_cfg,
            probability_threshold=float(decision_cfg["probability_threshold"]),
            use_weighted_loss=get_weighted_loss_flag(config, model_name),
            device=None,
        )

        trained_model = final_train_run["model"]
        threshold_prob = predict_proba_torch_model(
            model=trained_model,
            X=X_val_internal,
            batch_size=int(training_cfg.get("batch_size", 64)),
            device=final_train_run.get("device", "cpu"),
        )
        threshold_info = choose_probability_threshold(
            y_true=y_val_internal,
            y_prob=threshold_prob,
            config=config,
        )
        selected_probability_threshold = float(threshold_info["selected_threshold"])
        test_results = evaluate_torch_model(
            model=trained_model,
            X=X_test_model,
            y_true=y_test,
            probability_threshold=selected_probability_threshold,
            batch_size=int(training_cfg.get("batch_size", 64)),
            device=final_train_run.get("device", "cpu"),
        )

        final_train_info = {
            "model_family": model_family,
            "internal_validation_used": True,
            "internal_val_size": int(len(y_val_internal)),
            "best_epoch": final_train_run.get("best_epoch"),
            "best_score": final_train_run.get("best_score"),
            "selected_learning_rate": final_train_run.get("selected_learning_rate"),
            "learning_rate_candidates": final_train_run.get("learning_rate_candidates"),
            "lr_search_results": final_train_run.get("lr_search_results", []),
            "history": final_train_run.get("history", []),
            "threshold_source": "development_internal_validation",
            "threshold_optimization": threshold_info,
        }

    else:
        raise ValueError(f"Unsupported model family: {model_family}")

    predictions_df = pd.DataFrame({
        "timestamp": ts_test.astype(str),
        "endpoint_index": test_label_diagnostics["endpoint_indices"],
        "y_true": y_test,
        "y_prob": test_results["y_prob"],
        "y_pred": test_results["y_pred"],
        "probability_threshold": selected_probability_threshold,
    })

    return {
        "model_name": model_name,
        "model_family": model_family,
        "scaler_name": scaler_name,
        "threshold": final_threshold,
        "label_threshold_quantile": float(labeling_cfg["threshold_quantile"]),
        "lookback": int(seq_cfg["lookback"]),
        "selected_probability_threshold": selected_probability_threshold,
        "metrics": test_results["metrics"],
        "n_dev_samples": int(len(y_dev)),
        "n_test_samples": int(len(y_test)),
        "dev_label_distribution": dev_label_summary.reset_index().to_dict(orient="records"),
        "test_label_distribution": test_label_summary.reset_index().to_dict(orient="records"),
        "dev_label_diagnostics": dev_label_diagnostics,
        "test_label_diagnostics": test_label_diagnostics,
        "baselines": compute_naive_baselines(
            context_df=full_context_df,
            y_true=y_test,
            endpoint_indices=test_label_diagnostics["endpoint_indices"],
            close_col=data_cfg["close_col"],
            seed=int(config["experiment"]["seed"]),
        ),
        "final_train_info": final_train_info,
        "predictions_df": predictions_df,
    }


def _as_candidate_list(config_section: Dict[str, Any], candidate_key: str, fallback_key: str) -> List[Any]:
    candidates = config_section.get(candidate_key)
    if candidates is None:
        candidates = [config_section[fallback_key]]
    if isinstance(candidates, (str, int, float)):
        candidates = [candidates]
    return list(candidates)


def build_experiment_config_variants(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand lightweight labeling/lookback candidate grids into config variants."""
    labeling_candidates = _as_candidate_list(
        config["labeling"],
        "threshold_quantile_candidates",
        "threshold_quantile",
    )
    lookback_candidates = _as_candidate_list(
        config["sequence"],
        "lookback_candidates",
        "lookback",
    )

    variants = []
    for threshold_quantile in labeling_candidates:
        for lookback in lookback_candidates:
            variant = deepcopy(config)
            variant["labeling"]["threshold_quantile"] = float(threshold_quantile)
            variant["sequence"]["lookback"] = int(lookback)
            variants.append(variant)
    return variants


def make_variant_tag(config: Dict[str, Any]) -> str:
    q_tag = int(round(float(config["labeling"]["threshold_quantile"]) * 100))
    lookback = int(config["sequence"]["lookback"])
    return f"q{q_tag:02d}_lb{lookback}"


def summarize_labeling_candidates(
    df: pd.DataFrame,
    close_col: str,
    horizon: int,
    quantiles: List[float],
    bull_label: int,
    bear_label: int,
    neutral_label: int,
) -> List[Dict[str, Any]]:
    returns = compute_forward_return(df[close_col], horizon=horizon)
    summaries = []
    for quantile in quantiles:
        threshold = compute_threshold(
            train_returns=returns,
            method="quantile",
            quantile=float(quantile),
        )
        labels = make_labels(
            returns=returns,
            threshold=threshold,
            bull_label=bull_label,
            bear_label=bear_label,
            neutral_label=neutral_label,
        )
        distribution = summarize_label_distribution(labels).reset_index().to_dict(orient="records")
        counts = labels.value_counts().to_dict()
        summaries.append({
            "threshold_quantile": float(quantile),
            "threshold": float(threshold),
            "distribution": distribution,
            "n_bull": int(counts.get(bull_label, 0)),
            "n_bear": int(counts.get(bear_label, 0)),
            "n_neutral": int(counts.get(neutral_label, 0)),
        })
    return summaries


def main(config_path: str = "configs/base.yaml", config_overrides=None) -> Dict[str, Any]:
    config = load_config(config_path, overrides=config_overrides)
    set_global_seed(int(config["experiment"]["seed"]))

    artifact_root = Path(config["artifacts"]["root_dir"])
    ensure_dir(artifact_root)
    ensure_dir(artifact_root / "metrics")
    ensure_dir(artifact_root / "predictions")
    ensure_dir(artifact_root / "models")
    ensure_dir(artifact_root / "plots")

    experiment_name = config["experiment"]["name"]
    primary_metric = config["evaluation"]["primary_metric"]

    raw_df = load_market_data(
        file_path=config["data"]["file_path"],
        date_col=config["data"]["date_col"],
    )

    df = select_columns(
        df=raw_df,
        feature_cols=config["data"]["feature_cols"],
        close_col=config["data"]["close_col"],
        date_col=config["data"]["date_col"],
    )

    df = apply_missing_value_policy(
        df=df,
        method=config["preprocessing"]["missing_policy"],
    )

    dev_df, test_df = split_dev_test(
        df=df,
        test_ratio=float(config["split"]["final_test_ratio"]),
    )

    cv_results: List[Dict[str, Any]] = []
    skipped_models: List[Dict[str, Any]] = []
    config_variants = build_experiment_config_variants(config)
    labeling_quantiles = sorted({
        float(variant["labeling"]["threshold_quantile"])
        for variant in config_variants
    })
    labeling_diagnostics = summarize_labeling_candidates(
        df=dev_df,
        close_col=config["data"]["close_col"],
        horizon=int(config["labeling"]["horizon"]),
        quantiles=labeling_quantiles,
        bull_label=int(config["labeling"]["bull_label"]),
        bear_label=int(config["labeling"]["bear_label"]),
        neutral_label=int(config["labeling"]["neutral_label"]),
    )

    for variant_config in config_variants:
        variant_tag = make_variant_tag(variant_config)
        for model_name in variant_config["models"]["enabled"]:
            for scaler_name in variant_config["preprocessing"]["scalers"]:
                try:
                    cv_result = run_cv_for_model_and_scaler(
                        config=variant_config,
                        dev_df=dev_df,
                        model_name=model_name,
                        scaler_name=scaler_name,
                    )
                    cv_result["variant_tag"] = variant_tag
                    cv_result["config_snapshot"] = variant_config
                    cv_results.append(cv_result)

                    fold_json_path = artifact_root / "metrics" / f"{experiment_name}_{variant_tag}_{model_name}_{scaler_name}_folds.json"
                    save_fold_results(cv_result["fold_results"], fold_json_path)

                except Exception as exc:
                    skipped_models.append({
                        "variant_tag": variant_tag,
                        "model_name": model_name,
                        "scaler_name": scaler_name,
                        "threshold_quantile": float(variant_config["labeling"]["threshold_quantile"]),
                        "lookback": int(variant_config["sequence"]["lookback"]),
                        "reason": f"{type(exc).__name__}: {exc}",
                    })

    if not cv_results:
        raise RuntimeError(
            f"No successful CV runs. Skipped/error combinations: {skipped_models}"
        )

    best_cv = select_best_cv_result(cv_results=cv_results, primary_metric=primary_metric)
    best_config = best_cv["config_snapshot"]

    final_test = run_final_train_and_test(
        config=best_config,
        dev_df=dev_df,
        test_df=test_df,
        model_name=best_cv["model_name"],
        scaler_name=best_cv["scaler_name"],
    )

    predictions_path = artifact_root / "predictions" / f"{experiment_name}_test_predictions.csv"
    final_test["predictions_df"].to_csv(predictions_path, index=False)

    report = {
        "experiment_name": experiment_name,
        "primary_metric": primary_metric,
        "config_snapshot": config,
        "n_total_rows": int(len(df)),
        "n_dev_rows": int(len(dev_df)),
        "n_test_rows": int(len(test_df)),
        "cv_candidates": [
            {
                "model_name": r["model_name"],
                "model_family": r["model_family"],
                "scaler_name": r["scaler_name"],
                "variant_tag": r["variant_tag"],
                "threshold_quantile": r["label_threshold_quantile"],
                "lookback": r["lookback"],
                "cv_summary": r["cv_summary"],
            }
            for r in cv_results
        ],
        "best_cv_selection": {
            "model_name": best_cv["model_name"],
            "model_family": best_cv["model_family"],
            "scaler_name": best_cv["scaler_name"],
            "variant_tag": best_cv["variant_tag"],
            "threshold_quantile": best_cv["label_threshold_quantile"],
            "lookback": best_cv["lookback"],
            "cv_summary": best_cv["cv_summary"],
        },
        "labeling_diagnostics": labeling_diagnostics,
        "final_test": {
            k: v for k, v in final_test.items() if k != "predictions_df"
        },
        "skipped_models": skipped_models,
    }

    report_path = artifact_root / "metrics" / f"{experiment_name}_report.json"
    save_experiment_report(report, report_path)
    append_experiment_summary(report, artifact_root.parent / "experiments_summary.csv")

    print(f"[OK] Best model: {best_cv['model_name']} | family: {best_cv['model_family']} | scaler: {best_cv['scaler_name']} | variant: {best_cv['variant_tag']}")
    print(f"[OK] Test MCC: {final_test['metrics']['mcc']:.6f}")
    print(f"[OK] Report: {report_path}")
    print(f"[OK] Predictions: {predictions_path}")

    if skipped_models:
        print("[INFO] Skipped / failed combinations:")
        for item in skipped_models:
            print(f"  - {item['model_name']} + {item['scaler_name']}: {item['reason']}")

    return report


if __name__ == "__main__":
    main()
