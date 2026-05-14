# Bull/Bear Market Classifier

A binary market-direction classifier that predicts whether the Nasdaq Composite will close **higher (Bull)** or **lower (Bear)** the next trading day (t+1). Built for the EHB 420E Artificial Neural Networks course.

---

## Problem Statement

Raw price movement is noisy. Instead of predicting the exact return, this project frames the problem as a **signal-filtering task**:

- Compute the 1-day forward return: `(Close[t+1] / Close[t]) - 1`
- Apply a data-driven threshold to separate meaningful moves from noise
- Classify each day as **Bull (1)**, **Bear (0)**, or **Neutral** — neutral samples are dropped before training

The threshold is derived from the 40th percentile of absolute returns on the **training fold only**, ensuring no information leaks from validation or test sets.

---

## Dataset

| Property | Value |
|---|---|
| Source | Nasdaq Composite — 10 years of daily market data |
| File | `data/raw/market_data_10y_enriched.csv` |
| Features | 7 macro/technical indicators |
| Labels | Binary (Bull = 1, Bear = 0) after neutral removal |

**Features used:**

| Feature | Description |
|---|---|
| `VIX_Term_Structure` | Short-vs-long-term implied volatility spread |
| `Yield_Curve` | 10Y–2Y Treasury spread (recession indicator) |
| `SKEW_Index` | Tail-risk sentiment from options market |
| `Risk_Appetite_Ratio` | Equity vs. safe-haven relative strength |
| `Crude_Oil` | Energy price as macro proxy |
| `VWAP_Deviation` | Price deviation from volume-weighted average |
| `Volume_Momentum` | Trend in trading volume |

---

## Methodology

### Labeling

```
forward_return = (Close[t+1] / Close[t]) - 1

if forward_return >  threshold  →  Bull  (1)
if forward_return < -threshold  →  Bear  (0)
otherwise                       →  Neutral (-1, dropped)

threshold = quantile(|train_returns|, q=0.40)
```

### Train / Validation / Test Split

All splits are **strictly chronological** — no shuffling, no random sampling.

```
Full Dataset (10 years)
│
├─── Development Set (85%) ──────────────────────────────┐
│    │                                                    │
│    ├─ Fold 1: [Train          ] | [Val]                 │
│    ├─ Fold 2: [Train ─────────] | [Val]                 │
│    ├─ Fold 3: [Train ──────────────── ] | [Val]         │
│    └─ Fold 4: [Train ────────────────────────] | [Val]  │
│                                                         │
└─── Final Test Set (15%) ── locked until final eval ─────┘
```

- **Expanding window cross-validation** (4 folds): each fold adds the previous validation block to the training set
- **Final test set** is separated first and never used during model selection or hyperparameter tuning

### Sequence Construction

Each sample is a **lookback window of 21 trading days** (~1 calendar month):

```
X shape: (n_samples, 21, 7)   — 3D tensor for sequence models
y shape: (n_samples,)          — binary label at the end of each window
```

Tabular models (Logistic Regression, tree ensembles) receive the flattened version: `(n_samples, 21 × 7 = 147)`.

### Scaling

`StandardScaler` or `RobustScaler` is fit **only on the training portion of each fold**. Validation and test sets are transformed using the training-fit scaler — never re-fit.

---

## Models

| Family | Models |
|---|---|
| Classical ML | Logistic Regression |
| Tree Ensembles | Random Forest, XGBoost, LightGBM, CatBoost |
| Deep Learning (PyTorch) | GRU, CNN1D, LSTM, TCN, Transformer Encoder |

All models are evaluated under the same data pipeline and cross-validation protocol.

---

## Evaluation

| Metric | Role |
|---|---|
| **MCC** (Matthews Correlation Coefficient) | Primary — robust to class imbalance |
| **F1 Score** | Secondary |
| **Balanced Accuracy** | Secondary |
| ROC-AUC, Precision, Recall | Reported per fold |

Cross-validation reports mean ± std across 4 folds. Final test evaluation uses the best model selected by MCC on the development set.

---

## Project Structure

```
project/
├── configs/
│   └── base.yaml                    # All experiment hyperparameters
│
├── data/
│   ├── raw/
│   │   ├── market_data_10y_enriched.csv
│   │   └── market_data_10y_zscore.csv
│   └── processed/
│       ├── X_tensor.npy
│       └── y_tensor.npy
│
├── src/
│   ├── utils/          # Config loading, seed, I/O helpers
│   ├── data/           # Loading, preprocessing, labeling, splitting, sequence building
│   ├── models/         # All model definitions + factory
│   ├── training/       # Training loops, loss functions, early stopping
│   ├── evaluation/     # Metrics, reports, visualizations, final comparison
│   └── run_experiment.py   # Main orchestration script
│
├── artifacts/
│   ├── models/         # Saved model checkpoints
│   ├── metrics/        # Per-fold and summary JSON reports
│   ├── predictions/    # Test set predictions
│   └── plots/          # Generated figures
│
├── codes/              # Original monolithic Colab scripts (reference only)
├── requirements.txt
└── configs/base.yaml
```

---

## Quickstart

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Run an experiment**

```bash
python -m src.run_experiment --config configs/base.yaml
```

The script will:
1. Load and preprocess the market data
2. Build sequences with the configured lookback window
3. Run expanding-window cross-validation for each enabled model
4. Save fold reports, metrics, and artifacts
5. Evaluate the best model on the final held-out test set

**3. Enable / disable models**

Edit `configs/base.yaml`:

```yaml
models:
  enabled:
    - logreg
    - gru
    - lstm
    - cnn1d
    - tcn
    - transformer
    - rf
    - xgboost
    - lightgbm
    - catboost
```

---

## Key Design Constraints

These rules are enforced throughout the entire pipeline to prevent data leakage:

1. **No shuffling** — chronological order is always preserved
2. **No random splits** — all splits are time-based
3. **Final test set is locked first** — never used for any model selection decision
4. **Threshold computed on train only** — applied as a fixed constant to val/test
5. **Scaler fit on train only** — val/test only call `.transform()`
6. **Neutral samples dropped after labeling** — not before threshold computation
7. **No leakage** — no future information can reach the training set

---

## References

Additional sources and repositories used for the MCC/F1 improvement and notebook synchronization work:

- Boughorbel et al. (2017) - *Optimal classifier for imbalanced data using Matthews Correlation Coefficient metric*
  Source: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0177678
- Chicco & Jurman (2020) - *The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy*
  Source: https://link.springer.com/article/10.1186/s12864-019-6413-7
- Ghosh et al. (2020) - *Forecasting directional movements of stock prices for intraday trading using LSTM and random forests*
  Paper: https://arxiv.org/abs/2004.10178
  Repository: https://github.com/pushpendughosh/Stock-market-forecasting
- Microsoft Qlib - AI-oriented quantitative investment platform and model benchmark workflows
  Repository: https://github.com/microsoft/qlib
- MLFinPy documentation - financial labeling methods, including fixed-horizon and triple-barrier labeling
  Documentation: https://mlfinpy.readthedocs.io/en/stable/Labelling.html

1. Chung et al. (2014) — *Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling*
2. Fischer & Krauss (2018) — *Deep learning with long short-term memory networks for financial market predictions*
3. Kingma & Ba (2015) — *Adam: A Method for Stochastic Optimization*
4. Boughorbel et al. (2017) — *Optimal classifier for imbalanced data using Matthews Correlation Coefficient metric*
5. Chicco & Jurman (2020) — *The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy*
6. Estrella & Mishkin (1996/1998) — *The yield curve as a predictor of US recessions*
7. Johnson (2011) — *VIX Term Structure as a predictor of volatility regimes*

---

## Course

**EHB 420E — Artificial Neural Networks**
Istanbul Technical University, Spring 2026
