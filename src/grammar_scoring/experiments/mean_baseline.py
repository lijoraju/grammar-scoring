"""E001 cross-validated mean regression workflow."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from grammar_scoring.evaluation.metrics import regression_metrics, rmse
from grammar_scoring.evaluation.validation import make_cv_folds
from grammar_scoring.models.baselines import MeanRegressor


@dataclass
class MeanBaselineResult:
    """OOF rows, fold diagnostics, and separate validation and training scores."""

    oof: pd.DataFrame
    per_fold: pd.DataFrame
    oof_metrics: dict[str, float]
    training_rmse: float


def evaluate_mean_baseline(
    train: pd.DataFrame,
    *,
    n_splits: int = 5,
    n_bins: int = 5,
    random_state: int = 42,
) -> MeanBaselineResult:
    """Evaluate E001 using canonical folds and training-only fold means.

    Args:
        train: Training table with unique filenames and finite numeric labels.
            Row order is preserved regardless of the DataFrame index.
        n_splits: Number of canonical validation folds.
        n_bins: Maximum target quantile bins for stratification.
        random_state: Deterministic fold seed.

    Returns:
        OOF predictions, per-fold metrics, overall OOF metrics, and full-training
        fit RMSE. Undefined Pearson scores remain NaN.

    Raises:
        ValueError: If training data or CV configuration is invalid.
        RuntimeError: If OOF coverage or predictions are invalid.
    """
    if not {"filename", "label"}.issubset(train.columns):
        raise ValueError("Training table must contain filename and label columns")
    filenames = train["filename"]
    if (
        filenames.isna().any()
        or filenames.astype(str).str.strip().eq("").any()
        or filenames.duplicated().any()
    ):
        raise ValueError("Training filenames must be non-empty and unique")
    folds = make_cv_folds(train["label"], n_splits, n_bins, random_state)
    targets = train["label"].to_numpy(dtype=np.float64)
    predictions = np.full(targets.size, np.nan)
    coverage = np.zeros(targets.size, dtype=np.int64)
    diagnostics = []
    for fold in range(n_splits):
        validation = folds == fold
        model = MeanRegressor().fit(targets[~validation])
        fold_predictions = model.predict(int(validation.sum()))
        predictions[validation] = fold_predictions
        coverage[validation] += 1
        scores = regression_metrics(targets[validation], fold_predictions)
        diagnostics.append(
            {
                "fold": fold,
                "training_target_mean": model.mean_,
                "validation_target_mean": float(targets[validation].mean()),
                **scores,
            }
        )
    if not np.all(coverage == 1) or not np.isfinite(predictions).all():
        raise RuntimeError("Every training row must have exactly one finite OOF value")
    oof = train.loc[:, ["filename", "label"]].copy()
    oof["fold"] = folds
    oof["prediction"] = predictions
    full_model = MeanRegressor().fit(targets)
    return MeanBaselineResult(
        oof=oof,
        per_fold=pd.DataFrame(diagnostics),
        oof_metrics=regression_metrics(targets, predictions),
        training_rmse=rmse(targets, full_model.predict(targets.size)),
    )
