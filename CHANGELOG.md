# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.2.0] - 2026-05-15

### Added
- Endpoint-based sequence construction for validation and test windows.
  - Reason: validation/test samples should be allowed to use already-known historical feature rows before the split boundary, while keeping label endpoints inside their own split.
- MCC-based probability threshold search using internal validation data.
  - Reason: a fixed `0.50` decision threshold can suppress MCC/F1 on weak and imbalanced financial signals; threshold selection must still avoid final-test leakage.
- Labeling and lookback experiment grids in `configs/base.yaml`.
  - Reason: `threshold_quantile` and `lookback` materially change neutral-drop rate and signal horizon, so they should be selected by cross-validation rather than hand-picked.
- Naive baseline reporting for `always_bull`, `always_bear`, `random`, and `previous_day_sign`.
  - Reason: low MCC is only meaningful when compared against simple non-learning strategies on the same endpoints.
- PR-AUC reporting alongside ROC-AUC, MCC, F1, balanced accuracy, and confusion matrix.
  - Reason: PR-AUC gives extra visibility when class balance shifts after neutral filtering.
- Focused tests for endpoint windows, split-safe label horizons, train-only threshold computation, and MCC threshold search.
  - Reason: the most important leakage and metric-selection rules should be locked down with regression tests.

### Changed
- Cross-validation now evaluates candidate combinations of label quantile and lookback before selecting the final model.
  - Reason: this makes model selection more defensible for noisy T+1 market-direction prediction.
- Sklearn final training now uses internal development validation only to choose the probability threshold, then retrains on all development samples before final test evaluation.
  - Reason: threshold tuning should not reduce final training data, but the final test set must remain untouched.
- Torch training can shuffle training batches via config while preserving chronological split boundaries.
  - Reason: batch order is not leakage once each sample is already a past-only sequence, and shuffling can improve neural optimization.
- Experiment summaries now include selected variant, selected probability threshold, PR-AUC, lookback, and threshold quantile.
  - Reason: final reports should explain which data/decision regime produced the selected MCC/F1.
- Notebook walkthroughs were synchronized with the latest endpoint-based pipeline, threshold-search flow, report schema, and model config keys.
  - Reason: the notebooks were added while the core training code was changing, so their examples needed to match the current source modules.

### Dependencies
- Added optional tree ensemble and test dependencies: `xgboost`, `lightgbm`, `catboost`, and `pytest`.
  - Reason: the comparison plan requires tree-based financial baselines and local regression tests.

## [0.1.0] — 2026-05-11

### Added
- Initial project scaffolding
- Directory structure: `configs/`, `src/data/`, `src/models/`, `src/training/`, `src/evaluation/`, `src/utils/`
- `.gitignore` for Python ML projects
- `requirements.txt` with core dependencies
- `PROJECT_MAP.md` — agent context and project map
- `CHANGELOG.md` — this file
- `configs/base.yaml` — default experiment configuration
