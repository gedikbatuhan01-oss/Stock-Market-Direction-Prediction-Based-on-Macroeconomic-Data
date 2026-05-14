from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional


class EarlyStopping:
    """
    Early stopping utility for maximizing a validation metric (e.g. MCC).

    Behavior:
    - tracks the best score seen so far
    - stops when no meaningful improvement occurs for `patience` epochs
    - optionally stores the best model state_dict
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        restore_best_state: bool = True,
    ) -> None:
        if patience < 1:
            raise ValueError("patience must be at least 1.")
        if min_delta < 0:
            raise ValueError("min_delta must be non-negative.")

        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_state = restore_best_state

        self.best_score: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.best_state_dict: Optional[dict[str, Any]] = None

        self.counter = 0
        self.should_stop = False

    def step(self, score: float, model: Optional[Any] = None, epoch: Optional[int] = None) -> bool:
        """
        Update early stopping state.

        Returns:
            bool: whether current score is a new best.
        """
        if self.best_score is None or score > (self.best_score + self.min_delta):
            self.best_score = float(score)
            self.best_epoch = epoch
            self.counter = 0

            if self.restore_best_state and model is not None:
                self.best_state_dict = deepcopy(model.state_dict())

            return True

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True

        return False

    def restore(self, model: Any) -> Any:
        """
        Restore best weights into the given model if available.
        """
        if self.restore_best_state and self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
        return model
