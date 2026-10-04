"""Cached E003b Ridge inference with an OOF parity gate and fixed E005 blend."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import Ridge

from grammar_scoring.evaluation.metrics import regression_metrics
from grammar_scoring.experiments.embedding_baseline import align_embedding_inputs
from grammar_scoring.experiments.tfidf_baseline import _validate_filenames
from grammar_scoring.features.embeddings import EmbeddingResult
from grammar_scoring.inference.cli import validate_predictions

E005_WEIGHT = 0.65
E003B_WEIGHT = 0.35
FOLD_IDS = (0, 1, 2, 3, 4)
EMBEDDING_DIMENSION = 768
OOF_ATOL = 1e-6
OOF_RTOL = 1e-6
EXPECTED_METRICS = {"rmse": 0.909674, "pearson_correlation": 0.705234}


@dataclass
class EmbeddingRidgeResult:
    """Fold diagnostics and canonical predictions after successful OOF parity."""

    fold_ids: tuple[int, ...]
    fold_test_predictions: NDArray[np.float64]
    predictions: pd.DataFrame
    oof_predictions: NDArray[np.float64]
    oof_metrics: dict[str, float]
    parity_max_absolute_error: float


def align_prediction_frame(
    predictions: pd.DataFrame, canonical: pd.DataFrame
) -> pd.DataFrame:
    """Validate exact prediction identities and align to canonical filename order.

    Args:
        predictions: Unmodified filename,label predictions in any order.
        canonical: Desired unique filename order.

    Returns:
        Validated predictions in canonical order.

    Raises:
        ValueError: If schema, identities, or numeric values are invalid.
    """
    if predictions.columns.tolist() != ["filename", "label"]:
        raise ValueError("Prediction columns must be exactly filename,label in order")
    _validate_filenames(canonical)
    _validate_filenames(predictions)
    if set(predictions.filename) != set(canonical.filename):
        raise ValueError("Prediction identities differ from canonical filenames")
    ordered = predictions.set_index("filename").reindex(canonical.filename)
    ordered = ordered.reset_index()
    validate_predictions(ordered, canonical)
    return ordered


def infer_embedding_ridge(
    train: pd.DataFrame,
    frozen_folds: pd.DataFrame,
    train_embeddings: EmbeddingResult,
    test: pd.DataFrame,
    test_embeddings: EmbeddingResult,
    canonical_oof: pd.DataFrame,
) -> EmbeddingRidgeResult:
    """Reconstruct five frozen Ridge fits and require canonical OOF parity.

    Args:
        train: Canonical training identities and labels.
        frozen_folds: Existing five frozen fold assignments.
        train_embeddings: Safely loaded cached training embeddings.
        test: Canonical test identities; placeholder labels are ignored.
        test_embeddings: Safely loaded cached test embeddings.
        canonical_oof: Existing filename,label,fold,prediction OOF artifact.

    Returns:
        Equal fold ensemble, per-fold predictions, and OOF parity diagnostics.

    Raises:
        ValueError: If inputs or OOF parity are invalid.
    """
    _validate_filenames(test)
    if len(train) != 769 or len(test) != 216:
        raise ValueError("Expected exactly 769 training and 216 test examples")
    for artifact in (train_embeddings, test_embeddings):
        artifact.__post_init__()
        if artifact.dimension != EMBEDDING_DIMENSION:
            raise ValueError("E003b embedding dimension must be exactly 768")
    features, folds = align_embedding_inputs(train, frozen_folds, train_embeddings)
    if not np.all(test_embeddings.split == "test"):
        raise ValueError("Test embeddings must have split exactly test")
    if set(test_embeddings.filenames) != set(test.filename):
        raise ValueError("Test embedding identities differ from canonical test")
    positions = pd.Index(test_embeddings.filenames).get_indexer(test.filename)
    test_features = test_embeddings.embeddings[positions]
    _validate_filenames(canonical_oof)
    if not {"label", "fold", "prediction"}.issubset(canonical_oof.columns):
        raise ValueError("Canonical OOF requires label, fold, prediction")
    reference = align_prediction_frame(
        canonical_oof[["filename", "prediction"]].rename(
            columns={"prediction": "label"}
        ),
        train,
    ).label.to_numpy()
    aligned_oof = canonical_oof.set_index("filename").reindex(train.filename)
    for column, expected in (("label", train.label.to_numpy()), ("fold", folds)):
        if not np.array_equal(aligned_oof[column].to_numpy(), expected):
            raise ValueError(f"Canonical OOF {column} differs")
    targets = train.label.to_numpy(dtype=np.float64)
    oof = np.full(len(train), np.nan, dtype=np.float64)
    fold_predictions = []
    for fold in FOLD_IDS:
        valid = folds == fold
        model = Ridge(alpha=1.0, solver="lsqr")
        model.fit(features[~valid], targets[~valid])
        oof[valid] = model.predict(features[valid])
        predicted = np.asarray(model.predict(test_features), dtype=np.float64)
        if predicted.shape != (len(test),) or not np.isfinite(predicted).all():
            raise ValueError("Fold test predictions must be complete and finite")
        fold_predictions.append(predicted)
    if not np.isfinite(oof).all():
        raise ValueError("OOF predictions must be finite")
    if not np.allclose(oof, reference, rtol=OOF_RTOL, atol=OOF_ATOL):
        raise ValueError("E003b OOF parity failed")
    metrics = regression_metrics(targets, oof)
    for name, expected in EXPECTED_METRICS.items():
        if not np.isclose(metrics[name], expected, rtol=0, atol=1e-6):
            raise ValueError(f"E003b OOF {name} does not reproduce canonical metric")
    matrix = np.stack(fold_predictions)
    predictions = pd.DataFrame(
        {"filename": test.filename.to_numpy(), "label": matrix.mean(axis=0)}
    )
    validate_predictions(predictions, test)
    return EmbeddingRidgeResult(
        FOLD_IDS,
        matrix,
        predictions,
        oof,
        metrics,
        float(np.max(np.abs(oof - reference))),
    )


def blend_predictions(e005: pd.DataFrame, e003b: pd.DataFrame) -> pd.DataFrame:
    """Combine aligned raw predictions with the fixed 0.65/0.35 weights.

    Args:
        e005: E005 filename,label predictions in any order.
        e003b: E003b predictions in canonical test order.

    Returns:
        Unclipped, unrounded blend in E003b order.
    """
    validate_predictions(e003b, e003b)
    aligned = align_prediction_frame(e005, e003b)
    result = e003b.copy()
    result["label"] = (
        E005_WEIGHT * aligned.label.to_numpy() + E003B_WEIGHT * e003b.label.to_numpy()
    )
    validate_predictions(result, e003b)
    return result


def write_predictions(
    predictions: pd.DataFrame, canonical: pd.DataFrame, output: Path
) -> None:
    """Write validated filename,label predictions without an index.

    Args:
        predictions: Predictions already in canonical order.
        canonical: Required filename order.
        output: Destination CSV path.
    """
    validate_predictions(predictions, canonical)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index=False)


def prediction_summary(values: NDArray[np.float64]) -> dict[str, float | int]:
    """Return count and population statistics for finite predictions.

    Args:
        values: Finite prediction vector.

    Returns:
        Count, minimum, maximum, mean, and population standard deviation.
    """
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Summary requires a nonempty finite vector")
    return {
        "count": len(values),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }
