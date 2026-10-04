import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from grammar_scoring.experiments.acoustic_baseline import evaluate_acoustic_baseline
from grammar_scoring.features.acoustic import FEATURE_COLUMNS
from grammar_scoring.inference import acoustic_ensemble as inference


def inputs():
    rng = np.random.default_rng(42)
    train = pd.DataFrame(
        {"filename": [f"t{i}.wav" for i in range(20)], "label": rng.uniform(0, 5, 20)}
    )
    test = pd.DataFrame({"filename": [f"s{i}.wav" for i in range(7)]})

    def features(table, split):
        frame = pd.DataFrame(
            rng.normal(size=(len(table), len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS
        )
        frame.insert(0, "duration_seconds", 10000.0)
        frame.insert(0, "split", split)
        frame.insert(0, "filename", table.filename)
        return frame

    return (
        train,
        train.assign(fold=np.arange(20) % 5),
        features(train, "train"),
        test,
        features(test, "test"),
    )


def test_feature_reuse_isolation_and_all_test_rows(monkeypatch):
    train, folds, features, test, test_features = inputs()
    assert inference.FEATURE_COLUMNS is FEATURE_COLUMNS
    assert "duration_seconds" not in FEATURE_COLUMNS
    raw = features[list(FEATURE_COLUMNS)].to_numpy()
    calls, models = [], []
    original_scaler, original_ridge = StandardScaler.fit, Ridge.fit

    def scaler_fit(self, x, y=None, **kwargs):
        calls.append((self, x.copy()))
        return original_scaler(self, x, y, **kwargs)

    def ridge_spy(self, x, y, **kwargs):
        models.append(self)
        expected = raw[folds.fold != len(models) - 1]
        assert self.alpha == 1.0 and self.solver == "lsqr"
        np.testing.assert_allclose(x, (expected - expected.mean(0)) / expected.std(0))
        np.testing.assert_array_equal(y, train.label[folds.fold != len(models) - 1])
        return original_ridge(self, x, y, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", scaler_fit)
    monkeypatch.setattr(Ridge, "fit", ridge_spy)
    result = inference.predict_fold_ensemble(
        train, folds, features, test, test_features
    )
    assert result.shape == (7, 6)
    assert len({id(model) for model in models}) == 5
    assert len({id(model) for model, _ in calls}) == 5
    for fold, (_, x) in enumerate(calls):
        np.testing.assert_array_equal(x, raw[folds.fold != fold])
    assert np.isfinite(result.iloc[:, 1:]).all().all()


def test_alignment_determinism_duration_and_exact_blend():
    args = inputs()
    first = inference.predict_fold_ensemble(*args)
    train, folds, features, test, test_features = args
    features.duration_seconds = 1.0
    second = inference.predict_fold_ensemble(
        train, folds.iloc[::-1], features.iloc[::-1], test, test_features.iloc[::-1]
    )
    pd.testing.assert_frame_equal(first, second)
    e005 = test.assign(label=np.linspace(-2, 8, len(test)))
    acoustic, blended = inference.make_candidates(
        test, first.iloc[::-1], e005.iloc[::-1]
    )
    np.testing.assert_array_equal(acoustic.label, first.iloc[:, 1:].to_numpy().mean(1))
    np.testing.assert_array_equal(
        blended.label, 0.55 * e005.label + 0.45 * acoustic.label
    )
    for frame in (acoustic, blended):
        assert list(frame.columns) == ["filename", "label"]
        assert frame.filename.tolist() == test.filename.tolist()


@pytest.mark.parametrize("kind", ["identity", "duplicate", "finite", "schema"])
def test_prediction_rejection(kind):
    test = inputs()[3]
    frame = test.assign(label=1.0)
    if kind == "identity":
        frame.loc[0, "filename"] = "unknown.wav"
    elif kind == "duplicate":
        frame.loc[0, "filename"] = frame.loc[1, "filename"]
    elif kind == "finite":
        frame.loc[0, "label"] = np.inf
    else:
        frame["extra"] = 0
    with pytest.raises(ValueError):
        inference.validate_predictions(frame, test)


@pytest.mark.parametrize("split", ["train", "test"])
@pytest.mark.parametrize("kind", ["identity", "finite", "schema"])
def test_feature_rejection(split, kind):
    args = list(inputs())
    frame = args[2 if split == "train" else 4]
    if kind == "identity":
        frame.loc[0, "filename"] = "unknown.wav"
    elif kind == "finite":
        frame.loc[0, FEATURE_COLUMNS[0]] = np.nan
    else:
        frame["extra"] = 0
    with pytest.raises(ValueError):
        inference.predict_fold_ensemble(*args)


def test_parity_acceptance_and_rejection():
    train, folds, features, _, _ = inputs()
    oof = evaluate_acoustic_baseline(train, folds, features).oof
    assert inference.check_oof_parity(oof, oof.iloc[::-1])["passed"]
    changed = oof.copy()
    changed.loc[0, "prediction"] += inference.PARITY_ATOL / 2
    assert inference.check_oof_parity(oof, changed)["passed"]
    changed.loc[0, "prediction"] += 1e-6
    with pytest.raises(ValueError, match="parity failed"):
        inference.check_oof_parity(oof, changed)


def test_existing_artifact_loader_alignment(tmp_path):
    _, _, _, test, features = inputs()
    path = tmp_path / "test.csv"
    features.iloc[::-1].to_csv(path, index=False)
    pd.testing.assert_frame_equal(
        inference.load_aligned_features(path, test, "test"), features
    )
