"""
ANN Project — Colab Toolkit

Drive'a tek dosya olarak yüklenir.
Notebook'lardan veya hücrelerden tek satırla çağrılır.

Kullanım (Colab):

    from google.colab import drive
    drive.mount('/content/drive')

    exec(open('/content/drive/MyDrive/ANN-Project/toolkit.py').read())
    setup()                          # repo + deps + data
    run(models=['gru', 'lstm'])      # deneyi çalıştır
    show_results()                   # en son raporu göster
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = (
    "https://github.com/gedikbatuhan01-oss/"
    "Stock-Market-Direction-Prediction-Based-on-Macroeconomic-Data.git"
)
REPO_BRANCH = "feature/notebooks-sync"
DRIVE_PROJECT = Path("/content/drive/MyDrive/ANN-Project")
REPO_DIR = Path("/content/repo")


def _mount_drive() -> None:
    if Path("/content/drive/MyDrive").exists():
        return
    from google.colab import drive  # type: ignore
    drive.mount("/content/drive")


def _clone_or_pull() -> None:
    if REPO_DIR.exists():
        print(f">> Repo güncelleniyor (git pull, branch: {REPO_BRANCH})...")
        subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin", REPO_BRANCH], check=True)
        subprocess.run(["git", "-C", str(REPO_DIR), "checkout", REPO_BRANCH], check=True)
        subprocess.run(["git", "-C", str(REPO_DIR), "pull", "origin", REPO_BRANCH], check=True)
    else:
        print(f">> Repo clone ediliyor (branch: {REPO_BRANCH})...")
        subprocess.run(
            ["git", "clone", "--branch", REPO_BRANCH, REPO_URL, str(REPO_DIR)],
            check=True,
        )


def _install_deps() -> None:
    req = REPO_DIR / "requirements.txt"
    if not req.exists():
        return
    print(">> Bağımlılıklar yükleniyor...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(req)],
        check=True,
    )


def _link_data() -> None:
    drive_data = DRIVE_PROJECT / "data" / "raw"
    repo_data = REPO_DIR / "data" / "raw"
    repo_data.mkdir(parents=True, exist_ok=True)
    if not drive_data.exists():
        print(f"[UYARI] Drive'da veri yok: {drive_data}")
        return
    for csv in drive_data.glob("*.csv"):
        dest = repo_data / csv.name
        if not dest.exists():
            shutil.copy(csv, dest)
            print(f">> Veri kopyalandı: {csv.name}")


def setup() -> Path:
    """Drive mount + repo clone/pull + deps + data linking. İdempotent."""
    _mount_drive()
    _clone_or_pull()
    _install_deps()
    _link_data()
    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))
    os.chdir(REPO_DIR)
    print(f">> Setup tamam. Çalışma dizini: {REPO_DIR}")
    return REPO_DIR


def run(models: list[str] | None = None, config: str = "configs/base.yaml") -> dict:
    """
    Deneyi çalıştır.

    Args:
        models: Çalıştırılacak modeller listesi. None ise config'deki tüm modeller.
                Örn: ['gru', 'lstm', 'transformer']
        config: Config dosyası yolu (repo köküne göre).
    """
    setup()
    overrides = {"models": {"enabled": models}} if models else None

    from src.run_experiment import main
    report = main(config, config_overrides=overrides)

    _save_artifacts_to_drive()
    return report


def _save_artifacts_to_drive() -> None:
    src = REPO_DIR / "artifacts"
    dst = DRIVE_PROJECT / "artifacts"
    if not src.exists():
        return
    shutil.copytree(src, dst, dirs_exist_ok=True)
    print(f">> Artifacts Drive'a kaydedildi: {dst}")


def _fmt(val) -> str:
    return f"{val:.4f}" if val is not None else "  N/A"


def show_results() -> None:
    """En son raporu güzel formatla göster."""
    reports = sorted((REPO_DIR / "artifacts" / "metrics").glob("*_report.json"))
    if not reports:
        print("Henüz rapor yok. Önce run() çalıştır.")
        return

    with open(reports[-1], encoding="utf-8") as f:
        report = json.load(f)

    best = report["best_cv_selection"]
    test = report["final_test"]["metrics"]

    # --- Tüm modellerin CV sonuçları ---
    print("=" * 58)
    print("TÜM MODEL SONUÇLARI (CV)")
    print("=" * 58)
    cv_results = report.get("cv_results", [])
    if cv_results:
        print(f"{'Model':<18} {'Scaler':<12} {'MCC':>8} {'F1':>8} {'BalAcc':>8}")
        print("-" * 58)
        for r in sorted(
            cv_results,
            key=lambda x: x.get("cv_summary", {}).get("mcc_mean", 0),
            reverse=True,
        ):
            s = r.get("cv_summary", {})
            print(
                f"{r.get('model_name', '?'):<18} "
                f"{r.get('scaler_name', '?'):<12} "
                f"{_fmt(s.get('mcc_mean')):>8} "
                f"{_fmt(s.get('f1_mean')):>8} "
                f"{_fmt(s.get('balanced_accuracy_mean')):>8}"
            )
    else:
        print("cv_results verisi raporda bulunamadı.")

    # --- En iyi model ---
    print()
    print("=" * 58)
    print("EN İYİ MODEL (CV MCC)")
    print("=" * 58)
    print(f"  Model   : {best.get('model_name', '?')}")
    print(f"  Scaler  : {best.get('scaler_name', '?')}")
    print(f"  Variant : {best.get('variant_tag', '-')}")
    cv_s = best.get("cv_summary", {})
    for label, key in [
        ("MCC",    "mcc_mean"),
        ("F1",     "f1_mean"),
        ("BalAcc", "balanced_accuracy_mean"),
    ]:
        val = cv_s.get(key)
        if val is not None:
            print(f"  CV {label:<8}: {val:.4f}")

    # --- Final test ---
    print()
    print("=" * 58)
    print("FİNAL TEST SONUÇLARI")
    print("=" * 58)
    for label, key in [
        ("MCC",               "mcc"),
        ("F1",                "f1"),
        ("Balanced Accuracy", "balanced_accuracy"),
        ("ROC-AUC",           "roc_auc"),
        ("PR-AUC",            "pr_auc"),
    ]:
        val = test.get(key)
        if val is not None:
            print(f"  {label:<20}: {val:.4f}")


AVAILABLE_MODELS = [
    "logreg",
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
    "gru",
    "lstm",
    "cnn1d",
    "tcn",
    "transformer",
]


def list_models() -> None:
    """Kullanılabilir model isimlerini listele."""
    print("Kullanılabilir modeller:")
    for m in AVAILABLE_MODELS:
        print(f"  - {m}")
