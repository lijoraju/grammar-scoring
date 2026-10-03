"""Deterministic offline E004 tests; no NLP models or test data required."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from grammar_scoring.evaluation.metrics import regression_metrics, rmse
from grammar_scoring.experiments.embedding_baseline import OOF_COLUMNS
from grammar_scoring.experiments.linguistic_baseline import (
    EXPERIMENTS,
    align_linguistic_inputs,
    evaluate_linguistic_baseline,
    experiment_features,
)
from grammar_scoring.features.linguistic import FEATURE_COLUMNS, FEATURE_FAMILIES


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
    features = train[["split", "filename"]].reset_index(drop=True)
    for column in FEATURE_COLUMNS:
        features[column] = rng.normal(size=23)
    # A held-out extreme makes fitting the scaler on all rows detectable.
    features.loc[0, "word_count"] = 1e6
    return train, folds, features


@pytest.mark.parametrize(
    "name,count", list(zip(EXPERIMENTS, [17, 51, 79, 91], strict=True))
)
def test_exact_feature_sets(name, count):
    families = list(FEATURE_FAMILIES)[: list(EXPERIMENTS).index(name) + 1]
    expected = tuple(
        feature
        for family in families
        for feature in FEATURE_FAMILIES[family]
        if feature not in {"duration_seconds", "words_per_minute"}
    )
    actual = experiment_features(name)
    assert actual == expected == experiment_features(name)
    assert len(actual) == count
    assert len(set(actual)) == count
    assert "duration_seconds" not in actual
    assert "words_per_minute" not in actual


def test_unknown_experiment():
    with pytest.raises(ValueError, match="Unknown"):
        experiment_features("duration_diagnostic")


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_alignment_pipeline_metrics_and_training_semantics(inputs, name, monkeypatch):
    train, folds, features = inputs
    original_fit = StandardScaler.fit
    seen = []

    def recording_fit(self, x, y=None, sample_weight=None):
        seen.append(np.asarray(x).copy())
        return original_fit(self, x, y, sample_weight=sample_weight)

    monkeypatch.setattr(StandardScaler, "fit", recording_fit)
    result = evaluate_linguistic_baseline(
        train, folds.iloc[::-1], features.iloc[::-1], name
    )
    assert len(seen) == 5
    assert list(result.oof.columns) == OOF_COLUMNS
    assert result.oof.index.equals(train.index)
    assert result.oof.filename.tolist() == train.filename.tolist()
    assert result.oof.filename.is_unique
    assert len(result.oof) == len(train)
    np.testing.assert_array_equal(result.oof.label, train.label)
    np.testing.assert_array_equal(result.oof.fold, folds.fold)
    assert np.isfinite(result.oof.prediction).all()
    x = features[list(experiment_features(name))].to_numpy()
    y = train.label.to_numpy()
    pooled_y, pooled_predictions = [], []
    for fold in range(5):
        valid = folds.fold.to_numpy() == fold
        np.testing.assert_array_equal(seen[fold], x[~valid])
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0, solver="lsqr")),
            ]
        )
        model.fit(x[~valid], y[~valid])
        predicted = model.predict(x[valid])
        np.testing.assert_array_equal(
            result.oof.prediction.to_numpy()[valid], predicted
        )
        diagnostic = result.per_fold.iloc[fold]
        assert diagnostic.n_train == (~valid).sum()
        assert diagnostic.n_valid == valid.sum()
        for metric, value in regression_metrics(y[valid], predicted).items():
            assert diagnostic[metric] == pytest.approx(value)
        pooled_y.extend(y[~valid])
        pooled_predictions.extend(model.predict(x[~valid]))
    assert result.training_rmse == pytest.approx(rmse(pooled_y, pooled_predictions))
    assert result.oof_metrics == regression_metrics(y, result.oof.prediction)
    np.testing.assert_array_equal(result.oof.residual, y - result.oof.prediction)
    np.testing.assert_array_equal(result.oof.abs_error, result.oof.residual.abs())


@pytest.mark.parametrize("source", ["train", "folds", "features"])
@pytest.mark.parametrize("issue", ["missing", "extra", "duplicate"])
def test_identity_rejections(inputs, source, issue):
    tables = dict(zip(["train", "folds", "features"], inputs, strict=True))
    frame = tables[source]
    if issue == "missing":
        tables[source] = frame.iloc[:-1]
    elif issue == "extra":
        tables[source] = pd.concat(
            [frame, frame.iloc[[0]].assign(filename="extra.wav")]
        )
    else:
        tables[source] = pd.concat([frame, frame.iloc[[0]]])
    with pytest.raises(ValueError):
        align_linguistic_inputs(tables["train"], tables["folds"], tables["features"])


@pytest.mark.parametrize("issue", ["label", "fold", "nan", "split", "schema"])
def test_invalid_values(inputs, issue):
    train, folds, features = inputs
    if issue == "label":
        folds.iloc[0, folds.columns.get_loc("label")] += 1
    elif issue == "fold":
        folds["fold"] = 0
    elif issue == "nan":
        features.iloc[0, 2] = np.nan
    elif issue == "split":
        features.loc[0, "split"] = "test"
    else:
        features = features.drop(columns=FEATURE_COLUMNS[-1])
    with pytest.raises(ValueError):
        align_linguistic_inputs(train, folds, features)


def test_runner_saves_canonical_artifacts(inputs, tmp_path, monkeypatch):
    from grammar_scoring.experiments import linguistic_baseline as module

    train, _, features = inputs
    positions = np.arange(769) % len(train)
    canonical = train.iloc[positions].reset_index(drop=True)
    canonical["filename"] = [f"canonical_{i}.wav" for i in range(769)]
    canonical["label"] = np.arange(769) % 21 / 4
    artifact = features.iloc[positions].reset_index(drop=True)
    artifact["filename"] = canonical.filename
    frozen = canonical[["filename", "label"]].copy()
    frozen["fold"] = np.arange(769) % 5
    frozen.iloc[::-1].to_csv(tmp_path / "train_folds.csv", index=False)
    artifact.to_csv(tmp_path / "linguistic_train.csv", index=False)
    reference = module.evaluate_linguistic_baseline(
        canonical, frozen, artifact, next(iter(EXPERIMENTS))
    ).oof
    for name in ("E002b_tfidf_word_char_ridge", "E003b_deberta_ridge"):
        reference.iloc[::-1].to_csv(tmp_path / f"{name}.csv", index=False)
    monkeypatch.setattr(module, "FEATURES_DIR", tmp_path)
    monkeypatch.setattr(module, "OOF_DIR", tmp_path)
    monkeypatch.setattr(module, "load_train_dataframe", lambda: canonical)
    results, correlations = module.run_linguistic_experiments()
    assert list(results) == list(EXPERIMENTS)
    assert len(correlations) == 8
    for name, (result, output) in results.items():
        assert output.name == f"{name}.csv"
        saved = pd.read_csv(output)
        assert len(saved) == 769
        assert saved.filename.is_unique
        assert saved.filename.tolist() == canonical.filename.tolist()
        np.testing.assert_array_equal(saved.label, canonical.label)
        np.testing.assert_array_equal(saved.fold, frozen.fold)
        assert np.isfinite(saved.prediction).all()
        pd.testing.assert_frame_equal(saved, result.oof)


def test_runner_rejects_noncanonical_row_count(inputs, monkeypatch):
    from grammar_scoring.experiments import linguistic_baseline as module

    monkeypatch.setattr(module, "load_train_dataframe", lambda: inputs[0])
    with pytest.raises(ValueError, match="769"):
        module.run_linguistic_experiments()
