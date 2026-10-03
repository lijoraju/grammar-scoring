"""Offline identity, frozen-fold, diagnostic, and comparison tests for E003."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from grammar_scoring.evaluation.metrics import pearson_correlation, rmse
from grammar_scoring.experiments.embedding_baseline import (
    OOF_COLUMNS,
    align_embedding_inputs,
    compare_oof_representations,
    evaluate_embedding_baseline,
)
from grammar_scoring.features.embeddings import EmbeddingResult


@pytest.fixture
def inputs():
    rng = np.random.default_rng(42)
    train = pd.DataFrame(
        {
            "filename": [f"sample_{i}.wav" for i in range(23)],
            "label": rng.normal(size=23),
            "split": "train",
        },
        index=np.arange(23) + 100,
    )
    folds = train[["filename", "label"]].copy()
    folds["fold"] = np.arange(23) % 5
    artifact = EmbeddingResult(
        np.full(23, "train"),
        train["filename"].to_numpy(dtype=str),
        rng.normal(size=(23, 7)).astype(np.float32),
        7,
    )
    return train, folds, artifact


def test_reordered_alignment_and_canonical_oof(inputs):
    train, folds, artifact = inputs
    reordered = EmbeddingResult(
        artifact.split[::-1], artifact.filenames[::-1], artifact.embeddings[::-1], 7
    )
    features, assignments = align_embedding_inputs(train, folds.iloc[::-1], reordered)
    np.testing.assert_array_equal(features, artifact.embeddings)
    np.testing.assert_array_equal(assignments, folds["fold"])
    first = evaluate_embedding_baseline(train, folds, artifact)
    second = evaluate_embedding_baseline(train, folds.iloc[::-1], reordered)
    pd.testing.assert_frame_equal(first.oof, second.oof)
    assert list(first.oof.columns) == OOF_COLUMNS
    assert first.oof.index.equals(train.index)
    assert first.oof.filename.tolist() == train.filename.tolist()
    assert len(first.oof) == 23
    assert first.oof.filename.is_unique
    assert np.isfinite(first.oof.prediction).all()
    np.testing.assert_array_equal(first.oof.fold, folds.fold)
    assert first.per_fold.n_valid.sum() == len(train)


@pytest.mark.parametrize("issue", ["missing", "extra", "duplicate", "split"])
def test_embedding_identity_rejections(inputs, issue):
    train, folds, artifact = inputs
    if issue == "missing":
        artifact.split = artifact.split[:-1]
        artifact.filenames = artifact.filenames[:-1]
        artifact.embeddings = artifact.embeddings[:-1]
    elif issue == "extra":
        artifact.split = np.append(artifact.split, "train")
        artifact.filenames = np.append(artifact.filenames, "unexpected.wav")
        artifact.embeddings = np.vstack([artifact.embeddings, artifact.embeddings[0]])
    elif issue == "duplicate":
        artifact.filenames[0] = artifact.filenames[1]
    else:
        artifact.split[0] = "test"
    with pytest.raises(ValueError):
        align_embedding_inputs(train, folds, artifact)


@pytest.mark.parametrize("issue", ["missing", "extra", "duplicate", "label", "fold"])
def test_frozen_fold_rejections(inputs, issue):
    train, folds, artifact = inputs
    if issue == "missing":
        folds = folds.iloc[:-1]
    elif issue == "extra":
        folds = pd.concat([folds, folds.iloc[[0]].assign(filename="extra.wav")])
    elif issue == "duplicate":
        folds = pd.concat([folds, folds.iloc[[0]]])
    elif issue == "label":
        folds.iloc[0, folds.columns.get_loc("label")] += 1
    else:
        folds["fold"] = 0
    with pytest.raises(ValueError):
        evaluate_embedding_baseline(train, folds, artifact)


@pytest.mark.parametrize("issue", ["duplicate", "split", "label"])
def test_training_rejections(inputs, issue):
    train, folds, artifact = inputs
    if issue == "duplicate":
        train.iloc[0, train.columns.get_loc("filename")] = train.filename.iloc[1]
    elif issue == "split":
        train.iloc[0, train.columns.get_loc("split")] = "test"
    else:
        train.iloc[0, train.columns.get_loc("label")] = np.nan
    with pytest.raises(ValueError):
        evaluate_embedding_baseline(train, folds, artifact)


def test_pooled_training_rmse_and_exact_ridge_predictions(inputs):
    train, folds, artifact = inputs
    result = evaluate_embedding_baseline(train, folds, artifact)
    targets, predicted = [], []
    for fold in range(5):
        valid = folds.fold.to_numpy() == fold
        model = Ridge(alpha=1.0, solver="lsqr")
        model.fit(artifact.embeddings[~valid], train.label.to_numpy()[~valid])
        np.testing.assert_array_equal(
            result.oof.prediction.to_numpy()[valid],
            model.predict(artifact.embeddings[valid]),
        )
        targets.extend(train.label.to_numpy()[~valid])
        predicted.extend(model.predict(artifact.embeddings[~valid]))
    assert result.training_rmse == pytest.approx(rmse(targets, predicted))


def test_prediction_and_residual_correlations(inputs):
    train, folds, artifact = inputs
    first = evaluate_embedding_baseline(train, folds, artifact).oof
    second = first.copy()
    second["prediction"] = np.linspace(-1, 1, len(first))
    second["residual"] = second.label - second.prediction
    second["abs_error"] = second.residual.abs()
    third = first.copy()
    result = compare_oof_representations(
        {"E002b": first, "E003a": second.iloc[::-1], "E003b": third}
    )
    assert len(result) == 3
    assert result.iloc[0].prediction_correlation == pytest.approx(
        pearson_correlation(first.prediction, second.prediction)
    )
    assert result.iloc[0].residual_correlation == pytest.approx(
        pearson_correlation(first.residual, second.residual)
    )
    assert result.iloc[1].prediction_correlation == pytest.approx(1)


@pytest.mark.parametrize(
    "issue",
    [
        "filename",
        "duplicate",
        "label",
        "fold",
        "prediction",
        "residual",
        "abs_error",
        "schema",
    ],
)
def test_comparison_rejects_misalignment(inputs, issue):
    train, folds, artifact = inputs
    first = evaluate_embedding_baseline(train, folds, artifact).oof
    second = first.copy()
    if issue == "filename":
        second.iloc[0, 0] = "unknown.wav"
    elif issue == "duplicate":
        second.iloc[0, 0] = second.filename.iloc[1]
    elif issue == "schema":
        second = second.drop(columns="abs_error")
    elif issue == "prediction":
        second.iloc[0, second.columns.get_loc(issue)] = np.inf
    else:
        second.iloc[0, second.columns.get_loc(issue)] += 1
    with pytest.raises(ValueError):
        compare_oof_representations({"E002b": first, "E003a": second})
