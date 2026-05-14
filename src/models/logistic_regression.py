from __future__ import annotations

from typing import Any, Dict

from sklearn.linear_model import LogisticRegression


def build_logreg(config: Dict[str, Any]) -> LogisticRegression:
    """
    Build sklearn LogisticRegression model from config.
    Expected config keys:
    - logreg.class_weight
    - logreg.max_iter
    - logreg.solver
    """
    logreg_cfg = config.get("logreg", {})

    model = LogisticRegression(
        class_weight=logreg_cfg.get("class_weight", "balanced"),
        max_iter=int(logreg_cfg.get("max_iter", 2000)),
        solver=logreg_cfg.get("solver", "lbfgs"),
        random_state=int(config.get("experiment", {}).get("seed", 42)),
    )
    return model
