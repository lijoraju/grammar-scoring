"""E003 fixed Ridge evaluation of cached, label-free train embeddings."""

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import Ridge

from grammar_scoring.config.paths import EMBEDDINGS_DIR, FEATURES_DIR, OOF_DIR
from grammar_scoring.data.transcripts import load_competition_transcripts
from grammar_scoring.evaluation.metrics import (
    pearson_correlation,
    regression_metrics,
    rmse,
)
from grammar_scoring.evaluation.validation import validate_cv_folds
from grammar_scoring.experiments.tfidf_baseline import _validate_filenames
from grammar_scoring.features.embeddings import EmbeddingResult, load_embeddings

OOF_COLUMNS = ["filename", "label", "fold", "prediction", "residual", "abs_error"]


@dataclass
class EmbeddingBaselineResult:
    """Canonical OOF rows, validation metrics, and pooled training diagnostics."""

    oof: pd.DataFrame
    per_fold: pd.DataFrame
    oof_metrics: dict[str, float]
    training_rmse: float


def align_embedding_inputs(
    train: pd.DataFrame,
    frozen_folds: pd.DataFrame,
    embeddings: EmbeddingResult,
    *,
    n_splits: int = 5,
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Validate identities and align features and frozen folds to training order.

    Args:
        train: Canonical unique filenames and finite labels; optional train split.
        frozen_folds: Unique filenames and integer folds, with optional labels.
        embeddings: Validated train-only embedding artifact in any row order.
        n_splits: Expected number of existing folds.

    Returns:
        Embedding matrix and fold IDs in canonical training order.

    Raises:
        ValueError: If identities, labels, splits, or folds are invalid.
    """
    _validate_filenames(train)
    _validate_filenames(frozen_folds)
    embeddings.__post_init__()
    if "label" not in train or "fold" not in frozen_folds:
        raise ValueError("Training labels and frozen fold column are required")
    if "split" in train and not train["split"].eq("train").all():
        raise ValueError("Only train data is accepted")
    if not np.all(embeddings.split == "train"):
        raise ValueError("All embedding records must have split == train")
    expected = set(train["filename"])
    for name, actual in (
        ("Embedding", set(embeddings.filenames)),
        ("Frozen fold", set(frozen_folds["filename"])),
    ):
        if expected != actual:
            raise ValueError(
                f"{name} identities differ: missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            )
    aligned = frozen_folds.set_index("filename").reindex(train["filename"])
    if "label" in aligned and not np.array_equal(
        train["label"].to_numpy(), aligned["label"].to_numpy()
    ):
        raise ValueError("Frozen fold labels differ from training labels")
    folds = aligned["fold"].to_numpy()
    validate_cv_folds(train["label"].to_numpy(), folds, n_splits)
    positions = pd.Index(embeddings.filenames).get_indexer(train["filename"])
    return embeddings.embeddings[positions], folds


def evaluate_embedding_baseline(
    train: pd.DataFrame,
    frozen_folds: pd.DataFrame,
    embeddings: EmbeddingResult,
    *,
    n_splits: int = 5,
) -> EmbeddingBaselineResult:
    """Evaluate fixed Ridge(alpha=1.0, solver='lsqr') within frozen folds.

    Args:
        train: Canonical training filenames and labels in desired OOF order.
        frozen_folds: Previously frozen filename/fold assignments.
        embeddings: Cached, label-free train embeddings.
        n_splits: Expected number of validation folds.

    Returns:
        OOF metrics and diagnostic RMSE pooled over all fold-training rows.

    Raises:
        ValueError: If inputs fail identity or fold validation.
        RuntimeError: If OOF coverage or predictions are invalid.
    """
    features, folds = align_embedding_inputs(
        train, frozen_folds, embeddings, n_splits=n_splits
    )
    targets = train["label"].to_numpy(dtype=np.float64)
    predictions = np.full(len(train), np.nan)
    coverage = np.zeros(len(train), dtype=np.int64)
    diagnostics = []
    training_targets = []
    training_predictions = []
    for fold in range(n_splits):
        valid = folds == fold
        model = Ridge(alpha=1.0, solver="lsqr")
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
    return EmbeddingBaselineResult(
        oof=oof,
        per_fold=pd.DataFrame(diagnostics),
        oof_metrics=regression_metrics(targets, predictions),
        training_rmse=rmse(
            np.concatenate(training_targets), np.concatenate(training_predictions)
        ),
    )


def compare_oof_representations(artifacts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute pairwise prediction and residual correlations after strict alignment.

    Args:
        artifacts: Named OOF tables with the established six-column schema.
            Row order may differ; identities, labels, and folds must agree.

    Returns:
        One row per pair with prediction and residual Pearson correlations.

    Raises:
        ValueError: If schema, identities, labels, folds, or values are invalid.
    """
    if len(artifacts) < 2:
        raise ValueError("At least two OOF artifacts are required")
    aligned = {}
    reference = next(iter(artifacts.values()))
    _validate_filenames(reference)
    for name, frame in artifacts.items():
        _validate_filenames(frame)
        if list(frame.columns) != OOF_COLUMNS:
            raise ValueError(f"{name}: invalid OOF schema")
        if set(frame["filename"]) != set(reference["filename"]):
            raise ValueError(f"{name}: OOF filename alignment differs")
        ordered = frame.set_index("filename").reindex(reference["filename"])
        validate_cv_folds(
            ordered["label"], ordered["fold"], n_splits=reference["fold"].nunique()
        )
        for column in ("label", "fold"):
            if not np.array_equal(ordered[column], reference[column]):
                raise ValueError(f"{name}: OOF {column} alignment differs")
        numeric = ordered[["label", "prediction", "residual", "abs_error"]]
        if any(dtype.kind not in "iuf" for dtype in numeric.dtypes):
            raise ValueError(f"{name}: OOF values must be numeric")
        if not np.isfinite(numeric.to_numpy()).all():
            raise ValueError(f"{name}: OOF values must be finite")
        residual = ordered["label"] - ordered["prediction"]
        if not np.allclose(ordered["residual"], residual, rtol=1e-10, atol=1e-12):
            raise ValueError(f"{name}: inconsistent residuals")
        if not np.allclose(
            ordered["abs_error"], np.abs(residual), rtol=1e-10, atol=1e-12
        ):
            raise ValueError(f"{name}: inconsistent absolute errors")
        aligned[name] = ordered
    return pd.DataFrame(
        [
            {
                "first": first,
                "second": second,
                "prediction_correlation": pearson_correlation(
                    aligned[first]["prediction"], aligned[second]["prediction"]
                ),
                "residual_correlation": pearson_correlation(
                    aligned[first]["residual"], aligned[second]["residual"]
                ),
            }
            for first, second in combinations(aligned, 2)
        ]
    )


def run_embedding_experiments() -> tuple[
    dict[str, tuple[EmbeddingBaselineResult, Path]], pd.DataFrame
]:
    """Run both E003 variants, validate comparisons, and save canonical OOF CSVs.

    Returns:
        Named results/output paths and three diagnostic correlation pairs.

    Raises:
        ValueError: If canonical artifacts are invalid or not exactly 769 rows.
        OSError: If artifacts cannot be read or written.
        RuntimeError: If predictions fail coverage validation.
    """
    train = load_competition_transcripts("train")
    if len(train) != 769:
        raise ValueError("Canonical E003 training data must have exactly 769 rows")
    folds = pd.read_csv(FEATURES_DIR / "train_folds.csv")
    results = {}
    comparisons = {"E002b": pd.read_csv(OOF_DIR / "E002b_tfidf_word_char_ridge.csv")}
    for name, stem, dimension in (
        ("E003a_minilm_ridge", "minilm", 384),
        ("E003b_deberta_ridge", "deberta_v3_base", 768),
    ):
        embeddings = load_embeddings(EMBEDDINGS_DIR / f"{stem}_train.npz", dimension)
        result = evaluate_embedding_baseline(train, folds, embeddings)
        results[name] = (result, OOF_DIR / f"{name}.csv")
        comparisons[name.split("_")[0]] = result.oof
    correlations = compare_oof_representations(comparisons)
    for result, output in results.values():
        output.parent.mkdir(parents=True, exist_ok=True)
        result.oof.to_csv(output, index=False)
    return results, correlations
