"""Shared regression metrics and fixed cross-validation protocol."""

from grammar_scoring.evaluation.metrics import (
    pearson_correlation,
    regression_metrics,
    rmse,
)
from grammar_scoring.evaluation.validation import (
    make_cv_folds,
    make_regression_stratification_bins,
    summarize_cv_folds,
    validate_cv_folds,
)

__all__ = [
    "make_cv_folds",
    "make_regression_stratification_bins",
    "pearson_correlation",
    "regression_metrics",
    "rmse",
    "summarize_cv_folds",
    "validate_cv_folds",
]
