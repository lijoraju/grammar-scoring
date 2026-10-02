"""Simple reusable regression baselines."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from grammar_scoring.evaluation.metrics import _numeric_vector


class MeanRegressor:
    """Predict the mean of the targets supplied to fit."""

    def __init__(self) -> None:
        """Initialize an unfitted regressor."""
        self.mean_: float | None = None

    def fit(self, y_train: ArrayLike) -> "MeanRegressor":
        """Fit the baseline using only the supplied training targets.

        Args:
            y_train: Non-empty, finite, one-dimensional real targets.

        Returns:
            This fitted regressor.

        Raises:
            ValueError: If targets are invalid.
        """
        targets = _numeric_vector(y_train, "y_train")
        # Scale before averaging to avoid overflow for large finite targets.
        scale = float(np.max(np.abs(targets)))
        self.mean_ = float(scale * np.mean(targets / scale)) if scale else 0.0
        return self

    def predict(self, n_samples: int) -> NDArray[np.float64]:
        """Return one fitted-mean prediction per requested sample.

        Args:
            n_samples: Non-negative integer prediction count.

        Returns:
            One-dimensional array of constant predictions.

        Raises:
            ValueError: If the count is invalid.
            RuntimeError: If the regressor has not been fitted.
        """
        if (
            isinstance(n_samples, bool)
            or not isinstance(n_samples, (int, np.integer))
            or n_samples < 0
        ):
            raise ValueError("n_samples must be a non-negative integer")
        if self.mean_ is None:
            raise RuntimeError("MeanRegressor must be fitted before prediction")
        return np.full(n_samples, self.mean_, dtype=np.float64)
