from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_SEARCH_PATHS = [
    Path("/content/drive/MyDrive/V_6 ANN/artifacts/metrics/tuning_results.csv"),
    Path("/content/drive/MyDrive/V_6 ANN/tuning_results.csv"),
    Path("V_6 ANN/artifacts/metrics/tuning_results.csv"),
    Path("V_6 ANN/tuning_results.csv"),
    Path("artifacts/metrics/tuning_results.csv"),
    Path("tuning_results.csv"),
]

METRIC_ALIASES = {
    "test_mcc": ["test_mcc", "mcc", "test_matthews_corrcoef"],
    "cv_mcc_mean": ["cv_mcc_mean", "cv_mcc", "val_mcc", "validation_mcc"],
    "test_balanced_accuracy": [
        "test_balanced_accuracy",
        "test_bal_acc",
        "balanced_accuracy",
        "test_balanced_acc",
    ],
    "test_f1": ["test_f1", "f1", "test_f1_score"],
    "test_roc_auc": ["test_roc_auc", "roc_auc", "test_auc"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select the final model from training/tuning CSV results. "
            "The script ranks every tried model configuration, keeps the best "
            "configuration per model, and writes final comparison outputs."
        )
    )
    parser.add_argument(
        "--csv",
        dest="csv_paths",
        action="append",
        default=[],
        help="Path to a tuning/final-comparison CSV. Can be passed multiple times.",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory containing CSV files. If provided, all CSV files in it are read.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Output directory. Default: next to the input CSV under "
            "'final_comparison', or Drive V_6 ANN metrics when auto-detected."
        ),
    )
    parser.add_argument(
        "--primary-metric",
        default="test_mcc",
        help="Primary metric used for ranking. Default: test_mcc.",
    )
    parser.add_argument(
        "--min-cv-mcc",
        type=float,
        default=None,
        help="Optional minimum cv_mcc_mean filter before selection.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top rows to print and save as shortlist. Default: 10.",
    )
    parser.add_argument(
        "--higher-is-better",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether larger primary metric is better. Default: true.",
    )
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        print(f"[INFO] Ignoring notebook/kernel arguments: {unknown_args}")
    return args


def find_default_csvs() -> list[Path]:
    env_csv = os.environ.get("FINAL_COMPARISON_CSV")
    if env_csv:
        path = Path(env_csv)
        if path.exists():
            return [path]

    found = [path for path in DEFAULT_SEARCH_PATHS if path.exists()]
    if found:
        return found

    drive_metrics = Path("/content/drive/MyDrive/V_6 ANN/artifacts/metrics")
    local_metrics = Path("V_6 ANN/artifacts/metrics")
    content_dir = Path("/content")
    for directory in [drive_metrics, local_metrics, content_dir, Path(".")]:
        if directory.exists():
            csvs = sorted(directory.glob("*.csv"))
            if csvs:
                return csvs
    return []


def upload_csv_in_colab() -> list[Path]:
    try:
        from google.colab import files  # type: ignore
    except Exception:
        return []

    print("[INFO] No CSV was found automatically. Please choose your result CSV file.")
    uploaded = files.upload()
    paths = []
    for name in uploaded.keys():
        path = Path("/content") / name
        if path.exists() and path.suffix.lower() == ".csv":
            paths.append(path)
    return paths


def collect_csv_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []

    for raw_path in args.csv_paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")
        paths.append(path)

    if args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        paths.extend(sorted(input_dir.glob("*.csv")))

    if not paths:
        paths = find_default_csvs()

    if not paths:
        paths = upload_csv_in_colab()

    if not paths:
        searched = "\n".join(f" - {path}" for path in DEFAULT_SEARCH_PATHS)
        raise FileNotFoundError(
            "No CSV file was found. Put tuning_results.csv in one of these paths "
            "or pass --csv explicitly:\n"
            f"{searched}"
        )

    return sorted(set(paths))


def first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    normalized = {col.lower().strip(): col for col in df.columns}
    for candidate in candidates:
        found = normalized.get(candidate.lower().strip())
        if found:
            return found
    return None


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    rename_map: dict[str, str] = {}
    for standard_name, aliases in METRIC_ALIASES.items():
        found = first_existing_column(df, aliases)
        if found and found != standard_name:
            rename_map[found] = standard_name

    for required in ["model_name", "experiment_name"]:
        found = first_existing_column(df, [required, required.replace("_", "")])
        if found and found != required:
            rename_map[found] = required

    df = df.rename(columns=rename_map)

    if "model_name" not in df.columns:
        if "experiment_name" in df.columns:
            df["model_name"] = df["experiment_name"].map(infer_model_from_experiment)
        else:
            raise ValueError("CSV must contain either model_name or experiment_name.")

    if "experiment_name" not in df.columns:
        df["experiment_name"] = df["model_name"].astype(str)

    for col in [
        "test_mcc",
        "cv_mcc_mean",
        "test_balanced_accuracy",
        "test_f1",
        "test_roc_auc",
        "threshold_quantile",
        "decision_threshold",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "scaler_name" not in df.columns:
        df["scaler_name"] = ""
    if "model_family" not in df.columns:
        df["model_family"] = ""

    return df


def infer_model_from_experiment(experiment_name: object) -> str:
    value = str(experiment_name).lower()
    known_models = [
        "transformer_encoder",
        "random_forest",
        "lightgbm",
        "catboost",
        "xgboost",
        "cnn1d",
        "logreg",
        "lstm",
        "gru",
        "tcn",
    ]
    for model in known_models:
        if model in value:
            return model
    return "unknown"


def read_results(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame = standardize_columns(frame)
        frame["source_csv"] = str(path)
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined["model_name"] = combined["model_name"].astype(str).str.lower().str.strip()
    combined["experiment_name"] = combined["experiment_name"].astype(str)
    return combined


def add_ranking_columns(
    df: pd.DataFrame,
    primary_metric: str,
    higher_is_better: bool,
) -> pd.DataFrame:
    df = df.copy()

    if primary_metric not in df.columns:
        available = ", ".join(df.columns)
        raise ValueError(
            f"Primary metric '{primary_metric}' was not found. Available columns: {available}"
        )

    metric_cols = [
        primary_metric,
        "cv_mcc_mean",
        "test_balanced_accuracy",
        "test_f1",
        "test_roc_auc",
    ]
    for col in metric_cols:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "decision_threshold" not in df.columns:
        df["decision_threshold"] = pd.NA
    if "threshold_quantile" not in df.columns:
        df["threshold_quantile"] = pd.NA

    df = df[df[primary_metric].notna()].copy()
    if df.empty:
        raise ValueError(f"No rows have a valid value for '{primary_metric}'.")

    ascending = not higher_is_better
    df = df.sort_values(
        by=[
            primary_metric,
            "cv_mcc_mean",
            "test_balanced_accuracy",
            "test_f1",
            "test_roc_auc",
        ],
        ascending=[ascending, False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    df["overall_rank"] = range(1, len(df) + 1)
    return df


def select_best_per_model(ranked: pd.DataFrame) -> pd.DataFrame:
    best = ranked.sort_values("overall_rank").groupby("model_name", as_index=False).first()
    best = best.sort_values("overall_rank").reset_index(drop=True)
    best["model_rank"] = range(1, len(best) + 1)
    return best


def choose_output_dir(args: argparse.Namespace, csv_paths: list[Path]) -> Path:
    if args.out_dir:
        return Path(args.out_dir)

    env_artifact_root = os.environ.get("V6_ANN_ARTIFACT_ROOT")
    if env_artifact_root:
        return Path(env_artifact_root) / "metrics" / "final_comparison"

    drive_artifact_root = Path("/content/drive/MyDrive/V_6 ANN/artifacts")
    if drive_artifact_root.exists() or Path("/content/drive/MyDrive").exists():
        return drive_artifact_root / "metrics" / "final_comparison"

    first_parent = csv_paths[0].parent
    return first_parent / "final_comparison"


def pick_display_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "overall_rank",
        "model_rank",
        "experiment_name",
        "model_name",
        "model_family",
        "scaler_name",
        "threshold_quantile",
        "decision_threshold",
        "cv_mcc_mean",
        "test_mcc",
        "test_balanced_accuracy",
        "test_f1",
        "test_roc_auc",
        "report_path",
        "prediction_path",
        "source_csv",
    ]
    return [col for col in preferred if col in df.columns]


def write_outputs(
    ranked: pd.DataFrame,
    best_per_model: pd.DataFrame,
    out_dir: Path,
    top_n: int,
    primary_metric: str,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    columns_ranked = pick_display_columns(ranked)
    columns_best = pick_display_columns(best_per_model)

    all_ranked_path = out_dir / "all_model_config_rankings.csv"
    best_per_model_path = out_dir / "best_per_model.csv"
    shortlist_path = out_dir / f"top_{top_n}_shortlist.csv"
    winner_json_path = out_dir / "FINAL_MODEL_SELECTION.json"
    winner_md_path = out_dir / "FINAL_MODEL_SELECTION.md"

    ranked[columns_ranked].to_csv(all_ranked_path, index=False)
    best_per_model[columns_best].to_csv(best_per_model_path, index=False)
    ranked.head(top_n)[columns_ranked].to_csv(shortlist_path, index=False)

    winner = ranked.iloc[0].to_dict()
    best_model_row = best_per_model.iloc[0].to_dict()
    payload = {
        "primary_metric": primary_metric,
        "winner": serialize_row(winner),
        "best_model_family_selection": serialize_row(best_model_row),
        "outputs": {
            "all_ranked": str(all_ranked_path),
            "best_per_model": str(best_per_model_path),
            "shortlist": str(shortlist_path),
            "winner_json": str(winner_json_path),
            "winner_markdown": str(winner_md_path),
        },
    }
    winner_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    md = build_markdown_summary(ranked, best_per_model, primary_metric, top_n)
    winner_md_path.write_text(md, encoding="utf-8")
    return payload


def serialize_row(row: dict) -> dict:
    clean = {}
    for key, value in row.items():
        if pd.isna(value):
            clean[key] = None
        elif hasattr(value, "item"):
            clean[key] = value.item()
        else:
            clean[key] = value
    return clean


def build_markdown_summary(
    ranked: pd.DataFrame,
    best_per_model: pd.DataFrame,
    primary_metric: str,
    top_n: int,
) -> str:
    winner = ranked.iloc[0]
    lines = [
        "# Final Model Selection",
        "",
        f"Primary metric: `{primary_metric}`",
        "",
        "## Winner",
        "",
        f"- Experiment: `{winner.get('experiment_name', '')}`",
        f"- Model: `{winner.get('model_name', '')}`",
        f"- Scaler: `{winner.get('scaler_name', '')}`",
        f"- Threshold quantile: `{winner.get('threshold_quantile', '')}`",
        f"- Decision threshold: `{winner.get('decision_threshold', '')}`",
        f"- Test MCC: `{winner.get('test_mcc', '')}`",
        f"- CV MCC mean: `{winner.get('cv_mcc_mean', '')}`",
        f"- Test balanced accuracy: `{winner.get('test_balanced_accuracy', '')}`",
        f"- Test F1: `{winner.get('test_f1', '')}`",
        "",
        "## Best Per Model",
        "",
    ]

    display_cols = [
        col
        for col in [
            "model_rank",
            "model_name",
            "experiment_name",
            "scaler_name",
            "threshold_quantile",
            "decision_threshold",
            "test_mcc",
            "cv_mcc_mean",
            "test_balanced_accuracy",
            "test_f1",
        ]
        if col in best_per_model.columns
    ]
    lines.append(to_markdown_table(best_per_model[display_cols]))
    lines.extend(["", f"## Top {top_n} Configurations", ""])

    top_cols = [col for col in display_cols if col != "model_rank"]
    top_cols = ["overall_rank"] + top_cols
    top_cols = [col for col in top_cols if col in ranked.columns]
    lines.append(to_markdown_table(ranked.head(top_n)[top_cols]))
    lines.append("")
    return "\n".join(lines)


def to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"

    text_df = df.copy()
    for col in text_df.columns:
        text_df[col] = text_df[col].map(format_markdown_value)

    headers = [str(col) for col in text_df.columns]
    rows = text_df.values.tolist()
    widths = []
    for idx, header in enumerate(headers):
        max_row_width = max((len(str(row[idx])) for row in rows), default=0)
        widths.append(max(len(header), max_row_width))

    header_line = "| " + " | ".join(
        header.ljust(widths[idx]) for idx, header in enumerate(headers)
    ) + " |"
    separator_line = "| " + " | ".join("-" * width for width in widths) + " |"
    row_lines = [
        "| " + " | ".join(str(row[idx]).ljust(widths[idx]) for idx in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator_line, *row_lines])


def format_markdown_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def main() -> None:
    args = parse_args()
    csv_paths = collect_csv_paths(args)
    df = read_results(csv_paths)

    if args.min_cv_mcc is not None and "cv_mcc_mean" in df.columns:
        before = len(df)
        df = df[df["cv_mcc_mean"].fillna(float("-inf")) >= args.min_cv_mcc].copy()
        print(f"[INFO] min_cv_mcc filter: {before} -> {len(df)} rows")

    ranked = add_ranking_columns(
        df=df,
        primary_metric=args.primary_metric,
        higher_is_better=args.higher_is_better,
    )
    best_per_model = select_best_per_model(ranked)
    out_dir = choose_output_dir(args, csv_paths)

    payload = write_outputs(
        ranked=ranked,
        best_per_model=best_per_model,
        out_dir=out_dir,
        top_n=args.top_n,
        primary_metric=args.primary_metric,
    )

    winner = payload["winner"]
    print("\n========== FINAL MODEL SELECTION ==========")
    print(f"Input CSV files: {len(csv_paths)}")
    for path in csv_paths:
        print(f" - {path}")
    print("\nWinner:")
    print(f"  experiment_name       : {winner.get('experiment_name')}")
    print(f"  model_name            : {winner.get('model_name')}")
    print(f"  scaler_name           : {winner.get('scaler_name')}")
    print(f"  threshold_quantile    : {winner.get('threshold_quantile')}")
    print(f"  decision_threshold    : {winner.get('decision_threshold')}")
    print(f"  test_mcc              : {winner.get('test_mcc')}")
    print(f"  cv_mcc_mean           : {winner.get('cv_mcc_mean')}")
    print(f"  test_balanced_accuracy: {winner.get('test_balanced_accuracy')}")
    print(f"  test_f1               : {winner.get('test_f1')}")

    print("\nBest per model:")
    display_cols = pick_display_columns(best_per_model)
    summary_cols = [
        col
        for col in [
            "model_rank",
            "model_name",
            "experiment_name",
            "scaler_name",
            "threshold_quantile",
            "decision_threshold",
            "test_mcc",
            "cv_mcc_mean",
            "test_balanced_accuracy",
            "test_f1",
        ]
        if col in display_cols
    ]
    print(best_per_model[summary_cols].to_string(index=False))

    print("\nOutputs:")
    for key, value in payload["outputs"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()



