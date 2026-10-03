"""E004 fixed, fold-scaled Ridge experiments on cached linguistic features."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from grammar_scoring.config.paths import FEATURES_DIR, OOF_DIR
from grammar_scoring.data.dataset import load_train_dataframe
from grammar_scoring.evaluation.metrics import regression_metrics, rmse
from grammar_scoring.evaluation.validation import validate_cv_folds
from grammar_scoring.experiments.embedding_baseline import compare_oof_representations
from grammar_scoring.experiments.tfidf_baseline import _validate_filenames
from grammar_scoring.features.linguistic import FEATURE_FAMILIES
from grammar_scoring.features.linguistic_artifacts import (
    load_features,
    validate_features,
)

EXCLUDED_FEATURES = frozenset({"duration_seconds", "words_per_minute"})
EXPERIMENTS = {
    "E004a_linguistic_surface_ridge": ("surface",),
    "E004b_linguistic_surface_pos_ridge": ("surface", "pos"),
    "E004c_linguistic_surface_pos_syntax_ridge": ("surface", "pos", "syntax"),
    "E004d_linguistic_all_ridge": ("surface", "pos", "syntax", "spoken"),
}


@dataclass
class LinguisticBaselineResult:
    """Canonical OOF rows, validation metrics, and pooled training diagnostics."""

    oof: pd.DataFrame
    per_fold: pd.DataFrame
    oof_metrics: dict[str, float]
    training_rmse: float


def experiment_features(experiment: str) -> tuple[str, ...]:
    """Return canonical registry order with exactly the two mandated exclusions.

    Args:
        experiment: Full canonical E004 experiment name.

    Returns:
        Ordered feature names for the requested cumulative family set.

    Raises:
        ValueError: If the experiment is not one of E004a through E004d.
    """
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown linguistic experiment: {experiment}")
    return tuple(
        feature
        for family in EXPERIMENTS[experiment]
        for feature in FEATURE_FAMILIES[family]
        if feature not in EXCLUDED_FEATURES
    )


def align_linguistic_inputs(
    train: pd.DataFrame,
    frozen_folds: pd.DataFrame,
    features: pd.DataFrame,
    *,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, NDArray[np.int64]]:
    """Validate train-only features and align all inputs explicitly by filename.

    Args:
        train: Canonical filenames and finite labels in desired OOF order.
        frozen_folds: Frozen filename/fold assignments and optional labels.
        features: Complete cached linguistic feature table in any row order.
        n_splits: Expected number of balanced frozen folds.

    Returns:
        Validated feature rows and folds in canonical training order.

    Raises:
        ValueError: If identities, schema, labels, folds, or values are invalid.
    """
    for frame in (train, frozen_folds, features):
        _validate_filenames(frame)
    if "label" not in train or "fold" not in frozen_folds:
        raise ValueError("Training labels and frozen fold column are required")
    if "split" in train and not train["split"].eq("train").all():
        raise ValueError("Only train data is accepted")
    expected = set(train["filename"])
    for name, frame in (("Feature", features), ("Frozen fold", frozen_folds)):
        if set(frame["filename"]) != expected:
            raise ValueError(f"{name} identities differ from training filenames")
    ordered = features.set_index("filename", drop=False).loc[train["filename"]]
    expected_rows = train[["filename"]].assign(split="train")
    validate_features(ordered, expected_rows, "train")
    aligned = frozen_folds.set_index("filename").reindex(train["filename"])
    if "label" in aligned and not np.array_equal(train["label"], aligned["label"]):
        raise ValueError("Frozen fold labels differ from training labels")
    folds = aligned["fold"].to_numpy()
    validate_cv_folds(train["label"].to_numpy(), folds, n_splits)
    return ordered, folds


def evaluate_linguistic_baseline(
    train: pd.DataFrame,
    frozen_folds: pd.DataFrame,
    artifact: pd.DataFrame,
    experiment: str,
    *,
    n_splits: int = 5,
) -> LinguisticBaselineResult:
    """Evaluate a fresh StandardScaler/Ridge pipeline inside every frozen fold.

    Args:
        train: Canonical training filenames and labels.
        frozen_folds: Existing frozen fold assignments.
        artifact: Complete train-only linguistic feature artifact.
        experiment: Full canonical E004 experiment name.
        n_splits: Expected number of validation folds.

    Returns:
        OOF metrics and RMSE pooled over every fold's training predictions,
        using the same diagnostic definition as E002/E003.

    Raises:
        ValueError: If experiment or input validation fails.
        RuntimeError: If OOF coverage or predictions are invalid.
    """
    columns = experiment_features(experiment)
    aligned, folds = align_linguistic_inputs(
        train, frozen_folds, artifact, n_splits=n_splits
    )
    features = aligned[list(columns)].to_numpy(dtype=np.float64)
    targets = train["label"].to_numpy(dtype=np.float64)
    predictions = np.full(len(train), np.nan)
    coverage = np.zeros(len(train), dtype=np.int64)
    diagnostics = []
    training_targets = []
    training_predictions = []
    for fold in range(n_splits):
        valid = folds == fold
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0, solver="lsqr")),
            ]
        )
        model.fit(features[~valid], targets[~valid])
        predicted = model.predict(features[valid])
        predictions[valid] = predicted
        coverage[valid] += 1
        training_targets.append(targets[~valid])
        training_predictions.append(model.predict(features[~valid]))
        diagnostics.append(
            {
                "fold": fold,
                "n_train": int((~valid).sum()),
                "n_valid": int(valid.sum()),
                **regression_metrics(targets[valid], predicted),
            }
        )
    if not np.all(coverage == 1) or not np.isfinite(predictions).all():
        raise RuntimeError("Every row must receive exactly one finite OOF prediction")
    oof = train.loc[:, ["filename", "label"]].copy()
    oof["fold"] = folds
    oof["prediction"] = predictions
    oof["residual"] = targets - predictions
    oof["abs_error"] = np.abs(oof["residual"])
    return LinguisticBaselineResult(
        oof=oof,
        per_fold=pd.DataFrame(diagnostics),
        oof_metrics=regression_metrics(targets, predictions),
        training_rmse=rmse(
            np.concatenate(training_targets), np.concatenate(training_predictions)
        ),
    )


def run_linguistic_experiments() -> tuple[
    dict[str, tuple[LinguisticBaselineResult, Path]], pd.DataFrame
]:
    """Run canonical E004 variants and save validated 769-row OOF artifacts.

    Returns:
        Named results/output paths and diagnostic pairwise correlations,
        including E004d against E002b and E003b.

    Raises:
        ValueError: If canonical training or comparison artifacts are invalid.
        OSError: If artifacts cannot be read or written.
        RuntimeError: If OOF prediction coverage is invalid.
    """
    train = load_train_dataframe()
    if len(train) != 769:
        raise ValueError("Canonical E004 training data must have exactly 769 rows")
    expected = train[["filename"]].assign(split="train")
    artifact = load_features(FEATURES_DIR / "linguistic_train.csv", expected, "train")
    folds = pd.read_csv(FEATURES_DIR / "train_folds.csv")
    results = {}
    comparisons = {}
    for name in EXPERIMENTS:
        result = evaluate_linguistic_baseline(train, folds, artifact, name)
        results[name] = (result, OOF_DIR / f"{name}.csv")
        comparisons[name.split("_")[0]] = result.oof
    for name, filename in (
        ("E002b", "E002b_tfidf_word_char_ridge.csv"),
        ("E003b", "E003b_deberta_ridge.csv"),
    ):
        comparisons[name] = pd.read_csv(OOF_DIR / filename)
    correlations = compare_oof_representations(comparisons)
    correlations = correlations.loc[
        (
            correlations["first"].str.startswith("E004")
            & correlations["second"].str.startswith("E004")
        )
        | correlations["first"].eq("E004d")
    ].reset_index(drop=True)
    for result, output in results.values():
        output.parent.mkdir(parents=True, exist_ok=True)
        result.oof.to_csv(output, index=False)
    return results, correlations
