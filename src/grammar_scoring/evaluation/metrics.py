"""Reusable metrics for held-out regression predictions."""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import pearsonr


def _numeric_vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional numeric array")
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    result = array.astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _paired_vectors(
    y_true: ArrayLike, y_pred: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    actual = _numeric_vector(y_true, "y_true")
    predicted = _numeric_vector(y_pred, "y_pred")
    if actual.size != predicted.size:
        raise ValueError("y_true and y_pred must have equal lengths")
    return actual, predicted


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute root mean squared error.

    Args:
        y_true: Non-empty, finite, one-dimensional real targets.
        y_pred: Predictions with the same shape as y_true.

    Returns:
        Root mean squared error as a Python float.

    Raises:
        ValueError: If inputs are invalid or have different lengths.
    """
    actual, predicted = _paired_vectors(y_true, y_pred)
    errors = actual - predicted
    scale = np.max(np.abs(errors))
    if scale == 0:
        return 0.0
    return float(scale * np.sqrt(np.mean(np.square(errors / scale))))


def pearson_correlation(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute Pearson correlation, returning NaN when it is undefined.

    Args:
        y_true: Non-empty, finite, one-dimensional real targets.
        y_pred: Predictions with the same shape as y_true.

    Returns:
        Pearson coefficient as a Python float. Returns NaN for fewer than two
        samples or when either input is constant, including mean predictions.

    Raises:
        ValueError: If inputs are invalid or have different lengths.
    """
    actual, predicted = _paired_vectors(y_true, y_pred)
    if (
        actual.size < 2
        or np.all(actual == actual[0])
        or np.all(predicted == predicted[0])
    ):
        return float("nan")
    return float(pearsonr(actual, predicted).statistic)


def regression_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Compute the competition regression metrics.

    Args:
        y_true: Non-empty, finite, one-dimensional real targets.
        y_pred: Predictions with the same shape as y_true.

    Returns:
        Scores keyed by rmse and pearson_correlation; undefined Pearson is NaN.

    Raises:
        ValueError: If inputs are invalid or have different lengths.
    """
    return {
        "rmse": rmse(y_true, y_pred),
        "pearson_correlation": pearson_correlation(y_true, y_pred),
    }
