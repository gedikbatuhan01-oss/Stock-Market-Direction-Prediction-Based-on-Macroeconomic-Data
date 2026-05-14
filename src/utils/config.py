from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_config(config_path: str | Path, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load YAML config and apply optional overrides."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    if overrides:
        config = deep_update(config, overrides)

    validate_config(config)
    return config


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update nested dictionary values."""
    result = deepcopy(base)
    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def validate_config(config: Dict[str, Any]) -> None:
    """Validate required config keys and basic value constraints."""
    required_top_keys = [
        "experiment",
        "data",
        "split",
        "labeling",
        "sequence",
        "preprocessing",
        "training",
        "evaluation",
        "models",
        "artifacts",
    ]
    missing = [k for k in required_top_keys if k not in config]
    if missing:
        raise ValueError(f"Missing required top-level config keys: {missing}")

    seed = config.get("experiment", {}).get("seed")
    if seed is not None and not isinstance(seed, int):
        raise ValueError(f"experiment.seed must be an integer, got {type(seed).__name__}")

    test_ratio = config.get("split", {}).get("final_test_ratio")
    if test_ratio is not None and not (0 < test_ratio < 1):
        raise ValueError(f"split.final_test_ratio must be between 0 and 1, got {test_ratio}")
