"""E014 ensembling and out-of-fold linear calibration of fine-tuned runs.

Every E014 run stores fold-held-out OOF predictions and fold-averaged test
predictions. Runs are averaged with equal weight, and an optional linear
calibration ``y ~ a + b * prediction`` corrects the regression-to-the-mean
shrinkage of the averaged model. Calibration is evaluated honestly with
cross-fitting over the frozen folds: each fold's calibrated OOF predictions
come from a line fitted on the other four folds only. The test calibrator is
then fitted on all OOF predictions, which were themselves held out from the
models that produced them.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from grammar_scoring.evaluation.metrics import regression_metrics


def load_runs(
    run_dirs: Sequence[Path], column: str = "prediction"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and align OOF and test predictions from E014 run directories.

    Args:
        run_dirs: Directories each containing ``oof.csv`` and ``test.csv``.
        column: Prediction column to use (``prediction`` or ``prediction_last``).

    Returns:
        ``(oof, test)``. ``oof`` has filename, label, fold and one column per run;
        ``test`` has filename and one column per run. Run columns are named
        after their directories.

    Raises:
        ValueError: If no runs are given or runs disagree on rows or labels.
    """
    if not run_dirs:
        raise ValueError("At least one run directory is required")
    oof: pd.DataFrame | None = None
    test: pd.DataFrame | None = None
    for directory in run_dirs:
        name = Path(directory).name
        # Align OOF rows by filename so runs may store rows in any order.
        run_oof = pd.read_csv(Path(directory) / "oof.csv")
        run_oof = run_oof.sort_values("filename", ignore_index=True)
        run_test = pd.read_csv(Path(directory) / "test.csv")
        if run_oof[column].isna().any() or run_test[column].isna().any():
            raise ValueError(f"Missing predictions in run {name}")
        if oof is None or test is None:
            oof = run_oof[["filename", "label", "fold"]].copy()
            test = run_test[["filename"]].copy()
        else:
            if not run_oof["filename"].equals(oof["filename"]):
                raise ValueError(f"OOF rows of run {name} are not aligned")
            if not np.allclose(run_oof["label"], oof["label"]) or not run_oof[
                "fold"
            ].equals(oof["fold"]):
                raise ValueError(f"Labels or folds of run {name} disagree")
            if not run_test["filename"].equals(test["filename"]):
                raise ValueError(f"Test rows of run {name} are not aligned")
        oof[name] = run_oof[column].to_numpy()
        test[name] = run_test[column].to_numpy()
    assert oof is not None and test is not None
    return oof, test


def fit_line(predictions: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Fit ordinary least squares ``labels ~ intercept + slope * predictions``.

    Args:
        predictions: One-dimensional model predictions.
        labels: Targets with the same length.

    Returns:
        ``(intercept, slope)``.
    """
    slope, intercept = np.polyfit(predictions, labels, deg=1)
    return float(intercept), float(slope)


def cross_fitted_calibration(
    predictions: np.ndarray, labels: np.ndarray, folds: np.ndarray
) -> np.ndarray:
    """Calibrate each fold with a line fitted only on the other folds.

    Args:
        predictions: OOF predictions.
        labels: Targets aligned with predictions.
        folds: Fold identifier of each row.

    Returns:
        Calibrated OOF predictions whose fold-``k`` values never used fold-``k``
        labels.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    folds = np.asarray(folds)
    calibrated = np.empty_like(predictions)
    for fold in np.unique(folds):
        held_out = folds == fold
        intercept, slope = fit_line(predictions[~held_out], labels[~held_out])
        calibrated[held_out] = intercept + slope * predictions[held_out]
    return calibrated


def ensemble_report(
    oof: pd.DataFrame, run_columns: Sequence[str]
) -> dict[str, dict[str, float]]:
    """Report pooled OOF metrics for each run, their mean and its calibration.

    Args:
        oof: Output of :func:`load_runs`.
        run_columns: Run columns to average.

    Returns:
        Mapping from model description to RMSE and Pearson correlation.
    """
    labels = oof["label"].to_numpy()
    report = {
        name: regression_metrics(labels, oof[name].to_numpy()) for name in run_columns
    }
    mean = oof[list(run_columns)].mean(axis=1).to_numpy()
    report["mean"] = regression_metrics(labels, mean)
    calibrated = cross_fitted_calibration(mean, labels, oof["fold"].to_numpy())
    report["mean_calibrated_cv"] = regression_metrics(labels, calibrated)
    return report


def ensemble_predictions(
    oof: pd.DataFrame,
    test: pd.DataFrame,
    run_columns: Sequence[str],
    *,
    calibrate: bool,
) -> pd.DataFrame:
    """Average test predictions of the runs and optionally apply calibration.

    Args:
        oof: OOF frame from :func:`load_runs` (used to fit the calibrator).
        test: Test frame from :func:`load_runs`.
        run_columns: Run columns to average.
        calibrate: Whether to apply the line fitted on all OOF predictions.

    Returns:
        Submission frame with ``filename`` and ``label`` columns.
    """
    prediction = test[list(run_columns)].mean(axis=1).to_numpy()
    if calibrate:
        intercept, slope = fit_line(
            oof[list(run_columns)].mean(axis=1).to_numpy(), oof["label"].to_numpy()
        )
        prediction = intercept + slope * prediction
    return pd.DataFrame({"filename": test["filename"], "label": prediction})


def main(argv: Sequence[str] | None = None) -> None:
    """Report OOF metrics and write an equal-weight-per-group submission.

    Each ``--group`` is a comma-separated list of run directories. Runs are
    averaged within a group and groups are averaged with equal weight, so a
    group with more seeds does not dominate the blend.
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    parser.add_argument("--group", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    groups = [[Path(p) for p in group.split(",")] for group in args.group]
    oof, test = load_runs([run for group in groups for run in group])
    for index, group in enumerate(groups):
        names = [run.name for run in group]
        oof[f"group_{index}"] = oof[names].mean(axis=1)
        test[f"group_{index}"] = test[names].mean(axis=1)
    columns = [f"group_{index}" for index in range(len(groups))]
    print(json.dumps(ensemble_report(oof, columns), indent=2))
    submission = ensemble_predictions(oof, test, columns, calibrate=False)
    submission.to_csv(args.output, index=False)
    print(f"Wrote {len(submission)} rows to {args.output}")
