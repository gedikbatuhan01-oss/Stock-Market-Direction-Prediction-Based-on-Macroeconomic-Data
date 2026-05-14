from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from src.utils.io import append_row_to_csv, ensure_dir, save_json


def save_fold_results(fold_results: List[Dict[str, Any]], output_path: str | Path) -> None:
    """
    Save per-fold detailed results to JSON.
    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    save_json({"fold_results": fold_results}, output_path)


def save_experiment_report(report: Dict[str, Any], output_path: str | Path) -> None:
    """
    Save a full experiment report to JSON.
    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    save_json(report, output_path)


def append_experiment_summary(report: Dict[str, Any], csv_path: str | Path) -> None:
    """
    Append one flattened summary row to a CSV file.
    """
    best_cfg = report.get("best_cv_selection", {})
    cv_summary = best_cfg.get("cv_summary", {})
    test_metrics = report.get("final_test", {}).get("metrics", {})

    row = {
        "experiment_name": report.get("experiment_name"),
        "model_name": best_cfg.get("model_name"),
        "scaler_name": best_cfg.get("scaler_name"),
        "primary_metric": report.get("primary_metric"),
        "cv_mcc_mean": cv_summary.get("mcc_mean"),
        "cv_mcc_std": cv_summary.get("mcc_std"),
        "cv_f1_mean": cv_summary.get("f1_mean"),
        "cv_balanced_accuracy_mean": cv_summary.get("balanced_accuracy_mean"),
        "test_mcc": test_metrics.get("mcc"),
        "test_f1": test_metrics.get("f1"),
        "test_balanced_accuracy": test_metrics.get("balanced_accuracy"),
        "test_accuracy": test_metrics.get("accuracy"),
        "test_roc_auc": test_metrics.get("roc_auc"),
        "lookback": report.get("config_snapshot", {}).get("sequence", {}).get("lookback"),
        "threshold_method": report.get("config_snapshot", {}).get("labeling", {}).get("threshold_method"),
        "threshold_quantile": report.get("config_snapshot", {}).get("labeling", {}).get("threshold_quantile"),
        "probability_threshold": report.get("config_snapshot", {}).get("decision", {}).get("probability_threshold"),
        "seed": report.get("config_snapshot", {}).get("experiment", {}).get("seed"),
    }

    append_row_to_csv(row, csv_path)
