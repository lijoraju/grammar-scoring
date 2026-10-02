"""Deterministic, target-only stratification and fold diagnostics."""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.model_selection import StratifiedKFold

from grammar_scoring.evaluation.metrics import _numeric_vector


def _positive_integer(value: int, name: str, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def make_regression_stratification_bins(
    y: ArrayLike, n_bins: int = 5, *, n_splits: int = 5
) -> NDArray[np.int64]:
    """Create categorical quantile strata without changing regression targets.

    Try pandas.qcut with duplicate edges dropped, reducing the requested number
    of bins until all occupied strata have at least n_splits samples. Empty
    categories are removed and IDs are contiguous. A single stratum is a valid
    fallback, including for constant targets; it supplies size balance only.

    Args:
        y: Non-empty, finite, one-dimensional real targets in training row order.
        n_bins: Maximum requested number of quantile bins.
        n_splits: Minimum samples per occupied stratum; at least two.

    Returns:
        Integer stratum IDs in input row order, with no missing values.

    Raises:
        ValueError: If targets or parameters are invalid, or there are fewer
            samples than n_splits.
    """
    targets = _numeric_vector(y, "y")
    _positive_integer(n_bins, "n_bins")
    _positive_integer(n_splits, "n_splits", 2)
    if targets.size < n_splits:
        raise ValueError("Cannot stratify: sample count must be at least n_splits")
    for count in range(min(n_bins, targets.size // n_splits), 1, -1):
        raw = np.asarray(pd.qcut(targets, q=count, labels=False, duplicates="drop"))
        if not np.isfinite(raw).all():
            continue
        _, bins, counts = np.unique(raw, return_inverse=True, return_counts=True)
        if counts.min() >= n_splits:
            return bins.astype(np.int64)
    return np.zeros(targets.size, dtype=np.int64)


def validate_cv_folds(
    y: ArrayLike, fold_assignments: ArrayLike, n_splits: int = 5
) -> None:
    """Check complete, contiguous and balanced validation fold assignments.

    Args:
        y: Non-empty, finite, one-dimensional real targets.
        fold_assignments: One integer fold ID for every target, in row order.
        n_splits: Expected number of non-empty folds, at least two.

    Raises:
        ValueError: If assignments are malformed, missing folds, or fold sizes
            differ by more than one sample.
    """
    targets = _numeric_vector(y, "y")
    _positive_integer(n_splits, "n_splits", 2)
    folds = np.asarray(fold_assignments)
    if folds.ndim != 1 or folds.size != targets.size or folds.dtype.kind not in "iu":
        raise ValueError("fold_assignments must be one integer ID per target")
    if not np.array_equal(np.unique(folds), np.arange(n_splits)):
        raise ValueError("Fold IDs must include exactly 0..n_splits-1")
    counts = np.unique(folds, return_counts=True)[1]
    if counts.max() - counts.min() > 1:
        raise ValueError("Validation fold sizes must differ by at most one sample")


def make_cv_folds(
    y: ArrayLike, n_splits: int = 5, n_bins: int = 5, random_state: int = 42
) -> NDArray[np.int64]:
    """Assign every training row to a fixed stratified validation fold.

    Reproducibility requires identical target row order and arguments. Persist
    assignments with sample identifiers and reuse them in future experiments.
    No test data, feature values, or model fitting is involved.

    Args:
        y: Non-empty, finite, one-dimensional real training targets.
        n_splits: Number of validation folds, at least two.
        n_bins: Maximum requested number of target quantile bins.
        random_state: Deterministic integer seed in the range [0, 2**32 - 1].

    Returns:
        One integer validation fold ID per row, numbered 0..n_splits-1.

    Raises:
        ValueError: If inputs are invalid or stratification is impossible.
    """
    bins = make_regression_stratification_bins(y, n_bins, n_splits=n_splits)
    _positive_integer(random_state, "random_state", 0)
    if random_state > 2**32 - 1:
        raise ValueError("random_state must be <= 2**32 - 1")
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=int(random_state)
    )
    folds = np.full(bins.size, -1, dtype=np.int64)
    for fold, (_, validation_indices) in enumerate(splitter.split(bins, bins)):
        folds[validation_indices] = fold
    validate_cv_folds(y, folds, n_splits)
    return folds


def summarize_cv_folds(y: ArrayLike, fold_assignments: ArrayLike) -> pd.DataFrame:
    """Summarize target distributions for validated, balanced folds.

    Args:
        y: Non-empty, finite, one-dimensional real targets.
        fold_assignments: Contiguous integer fold IDs starting at zero.

    Returns:
        One row per fold with sample count, mean, population standard deviation
        (ddof=0), median, minimum, and maximum target values.

    Raises:
        ValueError: If targets or fold assignments are invalid.
    """
    targets = _numeric_vector(y, "y")
    folds = np.asarray(fold_assignments)
    validate_cv_folds(targets, folds, n_splits=np.unique(folds).size)
    frame = pd.DataFrame({"fold": folds, "target": targets})
    return (
        frame.groupby("fold", sort=True)["target"]
        .agg(
            n_samples="size",
            target_mean="mean",
            target_std=lambda values: values.std(ddof=0),
            target_median="median",
            target_min="min",
            target_max="max",
        )
        .reset_index()
    )
