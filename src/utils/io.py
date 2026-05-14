from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it does not exist and return Path object."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], output_path: str | Path) -> None:
    """Save dictionary as JSON file."""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def append_row_to_csv(row: Dict[str, Any], csv_path: str | Path) -> None:
    """Append one summary row to a CSV file, creating file if needed."""
    csv_path = Path(csv_path)
    ensure_dir(csv_path.parent)

    file_exists = csv_path.exists()
    fieldnames = list(row.keys())

    if file_exists:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_fields = reader.fieldnames or []
        fieldnames = list(dict.fromkeys(list(existing_fields) + fieldnames))

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def build_artifact_paths(config: Dict[str, Any], experiment_name: str) -> Dict[str, Path]:
    """Build standard artifact output paths for a given experiment."""
    root = Path(config.get("artifacts", {}).get("root_dir", "artifacts"))

    paths = {
        "root": root,
        "metrics": root / "metrics",
        "predictions": root / "predictions",
        "models": root / "models",
        "plots": root / "plots",
        "report_json": root / "metrics" / f"{experiment_name}_report.json",
        "folds_json": root / "metrics" / f"{experiment_name}_folds.json",
        "test_predictions": root / "predictions" / f"{experiment_name}_test_predictions.csv",
    }

    for key in ["root", "metrics", "predictions", "models", "plots"]:
        ensure_dir(paths[key])

    return paths
