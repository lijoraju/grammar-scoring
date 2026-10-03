import numpy as np
import pandas as pd
import pytest

from grammar_scoring.evaluation.metrics import rmse
from grammar_scoring.experiments.tfidf_baseline import evaluate_tfidf_baseline
from grammar_scoring.models.tfidf import build_tfidf_ridge


def synthetic_data():
    train = pd.DataFrame(
        {
            "filename": [f"audio_{i}.wav" for i in range(8)],
            "label": [1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5],
            "text": ["common fluent speech", "common halting speech"] * 4,
            "split": "train",
        },
        index=[9, 4, 1, 8, 7, 3, 6, 2],
    )
    folds = train[["filename", "label"]].copy()
    folds["fold"] = [1, 0, 1, 0, 0, 1, 0, 1]
    return train, folds.iloc[::-1]


@pytest.mark.parametrize("include_char", [False, True])
def test_frozen_oof_and_training_metric(include_char, monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("Folds must never be regenerated")

    monkeypatch.setattr(
        "grammar_scoring.evaluation.validation.make_cv_folds", forbidden
    )
    train, folds = synthetic_data()
    result = evaluate_tfidf_baseline(
        train, folds, include_char=include_char, n_splits=2
    )
    assert list(result.oof.columns) == [
        "filename",
        "label",
        "fold",
        "prediction",
        "residual",
        "abs_error",
    ]
    assert result.oof.index.equals(train.index)
    assert result.oof.filename.tolist() == train.filename.tolist()
    assert result.oof.filename.is_unique
    assert len(result.oof) == len(train)
    assert np.isfinite(result.oof.prediction).all()
    assert result.oof.fold.tolist() == [1, 0, 1, 0, 0, 1, 0, 1]
    expected_targets, expected_predictions = [], []
    for fold in range(2):
        valid = result.oof.fold.to_numpy() == fold
        model = build_tfidf_ridge(include_char=include_char).fit(
            train.text.to_numpy()[~valid], train.label.to_numpy()[~valid]
        )
        np.testing.assert_allclose(
            result.oof.prediction.to_numpy()[valid],
            model.predict(train.text.to_numpy()[valid]),
        )
        expected_targets.extend(train.label.to_numpy()[~valid])
        expected_predictions.extend(model.predict(train.text.to_numpy()[~valid]))
    assert result.training_rmse == pytest.approx(
        rmse(expected_targets, expected_predictions)
    )
    np.testing.assert_allclose(
        result.oof.residual, result.oof.label - result.oof.prediction
    )
    np.testing.assert_allclose(result.oof.abs_error, abs(result.oof.residual))
    assert result.per_fold.n_train.tolist() == [4, 4]
    assert result.per_fold.n_valid.tolist() == [4, 4]
    path = tmp_path / "oof.csv"
    result.oof.to_csv(path, index=False)
    assert pd.read_csv(path).columns.tolist() == result.oof.columns.tolist()
    repeated = evaluate_tfidf_baseline(
        train, folds, include_char=include_char, n_splits=2
    )
    pd.testing.assert_frame_equal(result.oof, repeated.oof)


@pytest.mark.parametrize(
    "issue", ["duplicate", "missing", "label", "fold", "text", "split", "target"]
)
def test_invalid_inputs(issue):
    train, folds = synthetic_data()
    if issue == "duplicate":
        folds.iloc[0, 0] = folds.iloc[1, 0]
    elif issue == "missing":
        folds = folds.iloc[:-1]
    elif issue == "label":
        folds.iloc[0, 1] = 999
    elif issue == "fold":
        folds["fold"] = 0.5
    elif issue == "text":
        train.loc[9, "text"] = None
    elif issue == "split":
        train.loc[9, "split"] = "test"
    else:
        train.loc[9, "label"] = np.inf
        folds = folds.drop(columns="label")
    with pytest.raises(ValueError):
        evaluate_tfidf_baseline(train, folds, n_splits=2)


def test_evaluation_fits_vocabulary_only_on_training_partition(monkeypatch):
    train, folds = synthetic_data()
    train["text"] = [
        "common validationexclusive" if fold == 1 else "common trainingonly"
        for fold in [1, 0, 1, 0, 0, 1, 0, 1]
    ]
    models = []

    def capture_model(*, include_char=False):
        model = build_tfidf_ridge(include_char=include_char)
        models.append(model)
        return model

    monkeypatch.setattr(
        "grammar_scoring.experiments.tfidf_baseline.build_tfidf_ridge", capture_model
    )
    evaluate_tfidf_baseline(train, folds, n_splits=2)
    vocabulary = models[1].named_steps["tfidf"].vocabulary_
    assert "trainingonly" in vocabulary
    assert "validationexclusive" not in vocabulary
