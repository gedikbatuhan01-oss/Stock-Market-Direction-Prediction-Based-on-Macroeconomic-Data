from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import compute_classification_metrics
from src.training.callbacks import EarlyStopping
from src.training.losses import build_bce_with_logits_loss


# =========================================================
# Common prediction helpers
# =========================================================
def predict_labels_from_proba(
    y_prob: np.ndarray,
    threshold: float = 0.50,
) -> np.ndarray:
    """
    Convert probabilities to binary labels using a decision threshold.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    y_prob = np.asarray(y_prob)
    if y_prob.ndim != 1:
        raise ValueError("y_prob must be 1D.")

    return (y_prob >= threshold).astype(np.int64)


# =========================================================
# sklearn-style training / evaluation
# =========================================================
def train_sklearn_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
) -> Any:
    """
    Fit a sklearn-style model and return the trained model.
    Uses eval_set for XGBoost/LightGBM/CatBoost when validation data is provided.
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    if X_train.ndim != 2:
        raise ValueError("X_train must be 2D for sklearn models.")
    if y_train.ndim != 1:
        raise ValueError("y_train must be 1D.")
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have the same number of samples.")

    if X_val is not None and y_val is not None:
        X_val = np.asarray(X_val)
        y_val = np.asarray(y_val)
        if X_val.ndim != 2:
            raise ValueError("X_val must be 2D for sklearn models.")
        if y_val.ndim != 1:
            raise ValueError("y_val must be 1D.")
        if len(X_val) != len(y_val):
            raise ValueError("X_val and y_val must have the same number of samples.")

        module_name = model.__class__.__module__.lower()
        class_name = model.__class__.__name__.lower()

        if "xgboost" in module_name or "xgb" in class_name:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=False,
            )
            return model

        if "lightgbm" in module_name or "lgbm" in class_name:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                eval_names=["train", "validation"],
                eval_metric="binary_logloss",
            )
            return model

        if "catboost" in module_name or "catboost" in class_name:
            model.fit(
                X_train,
                y_train,
                eval_set=(X_val, y_val),
                verbose=False,
            )
            return model

    model.fit(X_train, y_train)
    return model


def extract_sklearn_training_history(model) -> list[Dict[str, Any]]:
    """
    Normalize iteration-level eval histories from XGBoost, LightGBM, and CatBoost.
    Returns an empty list for estimators without iterative eval history.
    """
    raw_results = None

    if hasattr(model, "evals_result"):
        try:
            raw_results = model.evals_result()
        except Exception:
            raw_results = None

    if raw_results is None and hasattr(model, "evals_result_"):
        raw_results = getattr(model, "evals_result_", None)

    if raw_results is None and hasattr(model, "get_evals_result"):
        try:
            raw_results = model.get_evals_result()
        except Exception:
            raw_results = None

    if not raw_results:
        return []

    def pick_dataset(names: list[str], fallback_index: int = 0):
        for name in names:
            if name in raw_results:
                return name
        keys = list(raw_results.keys())
        if keys:
            return keys[min(fallback_index, len(keys) - 1)]
        return None

    train_key = pick_dataset(["validation_0", "train", "training", "learn"], fallback_index=0)
    val_key = pick_dataset(["validation_1", "validation", "valid_1", "valid_0"], fallback_index=1)

    def pick_metric(dataset_key):
        if dataset_key is None:
            return None, []
        metrics = raw_results.get(dataset_key, {})
        if not isinstance(metrics, dict) or not metrics:
            return None, []
        metric_keys = list(metrics.keys())
        preferred = ["logloss", "binary_logloss", "Logloss", "MultiClass", "auc", "AUC"]
        metric_key = next((m for m in preferred if m in metrics), metric_keys[0])
        return metric_key, list(metrics.get(metric_key, []))

    train_metric, train_values = pick_metric(train_key)
    val_metric, val_values = pick_metric(val_key)
    metric_name = val_metric or train_metric
    n_steps = max(len(train_values), len(val_values))

    history = []
    for idx in range(n_steps):
        history.append({
            "epoch": idx + 1,
            "iteration": idx + 1,
            "train_loss": float(train_values[idx]) if idx < len(train_values) else None,
            "val_loss": float(val_values[idx]) if idx < len(val_values) else None,
            "eval_metric": metric_name,
            "train_dataset": train_key,
            "validation_dataset": val_key,
        })
    return history


def predict_proba(
    model,
    X: np.ndarray,
    positive_class_index: int = 1,
) -> np.ndarray:
    """
    Return positive-class probabilities for sklearn-style models.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be 2D for sklearn models.")

    if not hasattr(model, "predict_proba"):
        raise AttributeError("Model does not implement predict_proba().")

    proba = model.predict_proba(X)

    if proba.ndim != 2:
        raise ValueError("predict_proba output must be 2D.")
    if positive_class_index >= proba.shape[1]:
        raise ValueError("positive_class_index is out of bounds for model probabilities.")

    return proba[:, positive_class_index]


def evaluate_sklearn_model(
    model,
    X: np.ndarray,
    y_true: np.ndarray,
    probability_threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Generate probabilities, predictions, and metrics for a trained sklearn model.
    """
    y_prob = predict_proba(model, X)
    y_pred = predict_labels_from_proba(y_prob, threshold=probability_threshold)
    metrics = compute_classification_metrics(y_true=y_true, y_pred=y_pred, y_prob=y_prob)

    return {
        "y_prob": y_prob,
        "y_pred": y_pred,
        "metrics": metrics,
    }


def run_single_sklearn_fold(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    probability_threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Train on one fold and evaluate on the validation split.
    """
    trained_model = train_sklearn_model(
        model,
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
    )
    val_results = evaluate_sklearn_model(
        model=trained_model,
        X=X_val,
        y_true=y_val,
        probability_threshold=probability_threshold,
    )

    return {
        "model": trained_model,
        "history": extract_sklearn_training_history(trained_model),
        "validation": val_results,
    }


# =========================================================
# torch helpers
# =========================================================
def _to_torch_float32(x: np.ndarray, device: str) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device)


def _to_torch_target(y: np.ndarray, device: str) -> torch.Tensor:
    return torch.tensor(y, dtype=torch.float32, device=device)


def create_torch_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = False,
) -> DataLoader:
    """
    Create TensorDataset/DataLoader for sequence models.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    if X.ndim != 3:
        raise ValueError("X must be 3D for torch sequence models: (n_samples, lookback, n_features).")
    if y.ndim != 1:
        raise ValueError("y must be 1D.")
    if len(X) != len(y):
        raise ValueError("X and y must have the same sample count.")

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,  # keep False for strict chronology
        drop_last=False,
    )


def predict_proba_torch_model(
    model: torch.nn.Module,
    X: np.ndarray,
    batch_size: int = 256,
    device: str = "cpu",
) -> np.ndarray:
    """
    Predict positive-class probabilities for a torch binary classifier
    that returns raw logits.
    """
    model.eval()

    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError("X must be 3D for torch sequence models.")

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    all_probs = []

    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.detach().cpu().numpy())

    return np.concatenate(all_probs, axis=0).astype(np.float32)


def evaluate_torch_model(
    model: torch.nn.Module,
    X: np.ndarray,
    y_true: np.ndarray,
    probability_threshold: float = 0.50,
    batch_size: int = 256,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Evaluate a torch binary classifier on sequence input.
    """
    y_prob = predict_proba_torch_model(
        model=model,
        X=X,
        batch_size=batch_size,
        device=device,
    )
    y_pred = predict_labels_from_proba(y_prob, threshold=probability_threshold)
    metrics = compute_classification_metrics(y_true=y_true, y_pred=y_pred, y_prob=y_prob)

    return {
        "y_prob": y_prob,
        "y_pred": y_pred,
        "metrics": metrics,
    }


def compute_torch_loss(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    loss_fn: torch.nn.Module,
    batch_size: int = 256,
    device: str = "cpu",
) -> float:
    """
    Compute average BCE loss for a torch model on a dataset.
    """
    model.eval()
    loader = create_torch_dataloader(
        X=X,
        y=y,
        batch_size=batch_size,
        shuffle=False,
    )

    losses = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            losses.append(float(loss.detach().cpu().item()))

    return float(np.mean(losses)) if losses else float("nan")


def _get_learning_rate_candidates(training_config: Dict[str, Any]) -> list[float]:
    candidates = training_config.get("learning_rate_candidates")
    if candidates is None:
        candidates = training_config.get("lr_candidates")

    if candidates is None:
        return [float(training_config.get("learning_rate", 1e-3))]

    if isinstance(candidates, (int, float, str)):
        candidates = [candidates]

    parsed = []
    for value in candidates:
        lr = float(value)
        if lr <= 0:
            raise ValueError("learning rate candidates must be positive.")
        if lr not in parsed:
            parsed.append(lr)

    if not parsed:
        raise ValueError("learning_rate_candidates must contain at least one value.")

    return parsed


def _train_torch_model_once(
    model: torch.nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    training_config: Dict[str, Any],
    probability_threshold: float = 0.50,
    use_weighted_loss: bool = True,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train a torch sequence model with early stopping on validation MCC.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = int(training_config.get("batch_size", 64))
    max_epochs = int(training_config.get("max_epochs", 100))
    learning_rate = float(training_config.get("_active_learning_rate", training_config.get("learning_rate", 1e-3)))
    patience = int(training_config.get("early_stopping_patience", 10))
    min_delta = float(training_config.get("early_stopping_min_delta", 0.0))

    model = model.to(device)

    train_loader = create_torch_dataloader(
        X=X_train,
        y=y_train,
        batch_size=batch_size,
        shuffle=False,
    )

    loss_fn = build_bce_with_logits_loss(
        y_train=y_train,
        use_weighted_loss=use_weighted_loss,
        device=device,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    early_stopper = EarlyStopping(
        patience=patience,
        min_delta=min_delta,
        restore_best_state=True,
    )

    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()

        epoch_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        val_loss = compute_torch_loss(
            model=model,
            X=X_val,
            y=y_val,
            loss_fn=loss_fn,
            batch_size=batch_size,
            device=device,
        )

        val_eval = evaluate_torch_model(
            model=model,
            X=X_val,
            y_true=y_val,
            probability_threshold=probability_threshold,
            batch_size=batch_size,
            device=device,
        )
        val_mcc = float(val_eval["metrics"]["mcc"])

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_metrics": val_eval["metrics"],
        })

        early_stopper.step(score=val_mcc, model=model, epoch=epoch)
        if early_stopper.should_stop:
            break

    model = early_stopper.restore(model)

    final_val_eval = evaluate_torch_model(
        model=model,
        X=X_val,
        y_true=y_val,
        probability_threshold=probability_threshold,
        batch_size=batch_size,
        device=device,
    )

    return {
        "model": model,
        "best_epoch": early_stopper.best_epoch,
        "best_score": early_stopper.best_score,
        "history": history,
        "validation": final_val_eval,
        "device": device,
        "learning_rate": learning_rate,
    }


def train_torch_model(
    model: torch.nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    training_config: Dict[str, Any],
    probability_threshold: float = 0.50,
    use_weighted_loss: bool = True,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train a torch sequence model and pick the best learning rate on validation MCC.
    """
    lr_candidates = _get_learning_rate_candidates(training_config)
    initial_state = deepcopy(model.state_dict())

    best_run: Optional[Dict[str, Any]] = None
    lr_search_results = []

    for learning_rate in lr_candidates:
        candidate_model = deepcopy(model)
        candidate_model.load_state_dict(deepcopy(initial_state))

        candidate_config = dict(training_config)
        candidate_config["_active_learning_rate"] = learning_rate

        run = _train_torch_model_once(
            model=candidate_model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            training_config=candidate_config,
            probability_threshold=probability_threshold,
            use_weighted_loss=use_weighted_loss,
            device=device,
        )

        score = run.get("best_score")
        score_for_sort = float("-inf") if score is None else float(score)
        lr_search_results.append({
            "learning_rate": learning_rate,
            "best_epoch": run.get("best_epoch"),
            "best_score": score,
            "final_val_loss": run["history"][-1].get("val_loss") if run.get("history") else None,
        })

        if best_run is None:
            best_run = run
            continue

        best_score = best_run.get("best_score")
        best_score_for_sort = float("-inf") if best_score is None else float(best_score)
        if score_for_sort > best_score_for_sort:
            best_run = run

    if best_run is None:
        raise RuntimeError("No learning-rate candidate produced a training run.")

    best_run["learning_rate_candidates"] = lr_candidates
    best_run["selected_learning_rate"] = best_run.get("learning_rate")
    best_run["lr_search_results"] = lr_search_results
    return best_run


def run_single_torch_fold(
    model: torch.nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    training_config: Dict[str, Any],
    probability_threshold: float = 0.50,
    use_weighted_loss: bool = True,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train and validate one fold for a torch model.
    """
    return train_torch_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        training_config=training_config,
        probability_threshold=probability_threshold,
        use_weighted_loss=use_weighted_loss,
        device=device,
    )
