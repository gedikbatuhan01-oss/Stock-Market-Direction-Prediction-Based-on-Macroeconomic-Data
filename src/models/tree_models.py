from __future__ import annotations

from importlib import import_module
from typing import Any, Dict

from sklearn.ensemble import RandomForestClassifier


def _optional_import(module_name: str, pip_name: str):
    try:
        return import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Optional dependency '{module_name}' is required for this model. "
            f"Install it with: pip install {pip_name}"
        ) from exc


def build_random_forest(config: Dict[str, Any]) -> RandomForestClassifier:
    rf_cfg = config.get("random_forest", {})
    return RandomForestClassifier(
        n_estimators=int(rf_cfg.get("n_estimators", 500)),
        max_depth=rf_cfg.get("max_depth", None),
        min_samples_leaf=int(rf_cfg.get("min_samples_leaf", 5)),
        max_features=rf_cfg.get("max_features", "sqrt"),
        class_weight=rf_cfg.get("class_weight", "balanced"),
        n_jobs=int(rf_cfg.get("n_jobs", -1)),
        random_state=int(config.get("experiment", {}).get("seed", 42)),
    )


def build_xgboost(config: Dict[str, Any]):
    xgb = _optional_import("xgboost", "xgboost")
    xgb_cfg = config.get("xgboost", {})
    return xgb.XGBClassifier(
        n_estimators=int(xgb_cfg.get("n_estimators", 400)),
        max_depth=int(xgb_cfg.get("max_depth", 3)),
        learning_rate=float(xgb_cfg.get("learning_rate", 0.03)),
        subsample=float(xgb_cfg.get("subsample", 0.8)),
        colsample_bytree=float(xgb_cfg.get("colsample_bytree", 0.8)),
        min_child_weight=float(xgb_cfg.get("min_child_weight", 1.0)),
        reg_lambda=float(xgb_cfg.get("reg_lambda", 1.0)),
        scale_pos_weight=float(xgb_cfg.get("scale_pos_weight", 1.0)),
        objective=xgb_cfg.get("objective", "binary:logistic"),
        eval_metric=xgb_cfg.get("eval_metric", "logloss"),
        tree_method=xgb_cfg.get("tree_method", "hist"),
        n_jobs=int(xgb_cfg.get("n_jobs", -1)),
        random_state=int(config.get("experiment", {}).get("seed", 42)),
        verbosity=int(xgb_cfg.get("verbosity", 0)),
    )


def build_lightgbm(config: Dict[str, Any]):
    lgb = _optional_import("lightgbm", "lightgbm")
    lgb_cfg = config.get("lightgbm", {})
    return lgb.LGBMClassifier(
        n_estimators=int(lgb_cfg.get("n_estimators", 400)),
        max_depth=int(lgb_cfg.get("max_depth", -1)),
        num_leaves=int(lgb_cfg.get("num_leaves", 31)),
        learning_rate=float(lgb_cfg.get("learning_rate", 0.03)),
        subsample=float(lgb_cfg.get("subsample", 0.8)),
        colsample_bytree=float(lgb_cfg.get("colsample_bytree", 0.8)),
        min_child_samples=int(lgb_cfg.get("min_child_samples", 20)),
        reg_lambda=float(lgb_cfg.get("reg_lambda", 1.0)),
        class_weight=lgb_cfg.get("class_weight", "balanced"),
        n_jobs=int(lgb_cfg.get("n_jobs", -1)),
        random_state=int(config.get("experiment", {}).get("seed", 42)),
        verbosity=int(lgb_cfg.get("verbosity", -1)),
    )


def build_catboost(config: Dict[str, Any]):
    cb = _optional_import("catboost", "catboost")
    cat_cfg = config.get("catboost", {})
    return cb.CatBoostClassifier(
        iterations=int(cat_cfg.get("iterations", 400)),
        depth=int(cat_cfg.get("depth", 4)),
        learning_rate=float(cat_cfg.get("learning_rate", 0.03)),
        l2_leaf_reg=float(cat_cfg.get("l2_leaf_reg", 3.0)),
        loss_function=cat_cfg.get("loss_function", "Logloss"),
        eval_metric=cat_cfg.get("eval_metric", "MCC"),
        auto_class_weights=cat_cfg.get("auto_class_weights", "Balanced"),
        random_seed=int(config.get("experiment", {}).get("seed", 42)),
        verbose=bool(cat_cfg.get("verbose", False)),
        allow_writing_files=bool(cat_cfg.get("allow_writing_files", False)),
    )
