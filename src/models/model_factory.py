from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from src.models.cnn1d_model import build_cnn1d_model
from src.models.gru_model import build_gru_model
from src.models.logistic_regression import build_logreg
from src.models.lstm_model import build_lstm_model
from src.models.tcn_model import build_tcn_model
from src.models.transformer_encoder_model import build_transformer_encoder_model
from src.models.tree_models import (
    build_catboost,
    build_lightgbm,
    build_random_forest,
    build_xgboost,
)


MODEL_ALIASES = {
    "rf": "random_forest",
    "randomforest": "random_forest",
    "xgb": "xgboost",
    "lgbm": "lightgbm",
    "cat": "catboost",
    "transformer": "transformer_encoder",
}

SKLEARN_MODEL_NAMES = {
    "logreg",
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
}
TORCH_MODEL_NAMES = {
    "gru",
    "cnn1d",
    "lstm",
    "tcn",
    "transformer_encoder",
}


def normalize_model_name(model_name: str) -> str:
    model_name = model_name.lower().strip()
    return MODEL_ALIASES.get(model_name, model_name)


def get_model_family(model_name: str) -> str:
    """
    Return model family:
    - "sklearn"
    - "torch"
    """
    model_name = normalize_model_name(model_name)

    if model_name in SKLEARN_MODEL_NAMES:
        return "sklearn"
    if model_name in TORCH_MODEL_NAMES:
        return "torch"

    raise ValueError(f"Unsupported model_name: {model_name}")


def requires_sequence_input(model_name: str) -> bool:
    """
    Returns True for sequence-native torch models.
    """
    return get_model_family(model_name) == "torch"


def build_model(
    model_name: str,
    config: Dict[str, Any],
    input_shape: Optional[Tuple[int, ...]] = None,
):
    """
    Build a model by name.
    """
    model_name = normalize_model_name(model_name)

    if model_name == "logreg":
        return build_logreg(config)
    if model_name == "random_forest":
        return build_random_forest(config)
    if model_name == "xgboost":
        return build_xgboost(config)
    if model_name == "lightgbm":
        return build_lightgbm(config)
    if model_name == "catboost":
        return build_catboost(config)
    if model_name == "gru":
        return build_gru_model(config=config, input_shape=input_shape)
    if model_name == "cnn1d":
        return build_cnn1d_model(config=config, input_shape=input_shape)
    if model_name == "lstm":
        return build_lstm_model(config=config, input_shape=input_shape)
    if model_name == "tcn":
        return build_tcn_model(config=config, input_shape=input_shape)
    if model_name == "transformer_encoder":
        return build_transformer_encoder_model(config=config, input_shape=input_shape)

    raise ValueError(f"Unsupported model_name: {model_name}")
