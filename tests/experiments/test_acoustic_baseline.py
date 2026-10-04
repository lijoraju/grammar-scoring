import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from grammar_scoring.evaluation.metrics import pearson_correlation, regression_metrics
from grammar_scoring.experiments.acoustic_baseline import (
    align_inputs,
    align_oof,
    blend_grid,
    complementarity,
    evaluate_acoustic_baseline,
)
from grammar_scoring.features.acoustic import ARTIFACT_COLUMNS, FEATURE_COLUMNS


def inputs():
    rng = np.random.default_rng(42)
    train = pd.DataFrame(
        {"filename": [f"{i}.wav" for i in range(20)], "label": rng.uniform(0, 5, 20)}
    )
    folds = train.assign(fold=np.arange(20) % 5)
    frame = pd.DataFrame(
        rng.normal(size=(20, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS
    )
    frame.insert(0, "duration_seconds", 1.0)
    frame.insert(0, "split", "train")
    frame.insert(0, "filename", train.filename)
    assert tuple(frame.columns) == ARTIFACT_COLUMNS
    return train, folds, frame


def test_alignment_is_by_filename():
    train, folds, features = inputs()
    expected = evaluate_acoustic_baseline(train, folds, features)
    actual = evaluate_acoustic_baseline(train, folds.iloc[::-1], features.iloc[::-1])
    pd.testing.assert_frame_equal(actual.oof, expected.oof)
    pd.testing.assert_frame_equal(actual.per_fold, expected.per_fold)
    assert actual.training_rmse == expected.training_rmse
    assert len(actual.oof) == len(train)
    assert actual.oof.filename.is_unique
    assert np.isfinite(actual.oof.prediction).all()
    np.testing.assert_allclose(actual.oof.residual, train.label - actual.oof.prediction)


def test_scaler_and_ridge_fit_exclude_validation(monkeypatch):
    train, folds, features = inputs()
    raw = features[list(FEATURE_COLUMNS)].to_numpy()
    scaler_fit = StandardScaler.fit
    ridge_fit = Ridge.fit
    scaler_calls, ridge_calls = [], []

    def spy_scaler(self, x, y=None, **kwargs):
        scaler_calls.append(np.array(x))
        return scaler_fit(self, x, y, **kwargs)

    def spy_ridge(self, x, y, **kwargs):
        ridge_calls.append((np.array(x), np.array(y)))
        return ridge_fit(self, x, y, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", spy_scaler)
    monkeypatch.setattr(Ridge, "fit", spy_ridge)
    evaluate_acoustic_baseline(train, folds, features)
    assert len(scaler_calls) == len(ridge_calls) == 5
    for fold, (scaler_x, (ridge_x, ridge_y)) in enumerate(
        zip(scaler_calls, ridge_calls, strict=True)
    ):
        valid = folds.fold.to_numpy() == fold
        expected = raw[~valid]
        np.testing.assert_array_equal(scaler_x, expected)
        np.testing.assert_allclose(
            ridge_x, (expected - expected.mean(axis=0)) / expected.std(axis=0)
        )
        np.testing.assert_array_equal(ridge_y, train.label.to_numpy()[~valid])
        assert scaler_x.shape[0] == ridge_x.shape[0] == 16


@pytest.mark.parametrize("kind", ["identity", "label", "fold", "split", "finite"])
def test_input_mismatch_rejected(kind):
    train, folds, features = inputs()
    if kind == "identity":
        folds.loc[0, "filename"] = "other.wav"
    elif kind == "label":
        folds.loc[0, "label"] += 1
    elif kind == "fold":
        folds.loc[0, "fold"] = 99
    elif kind == "split":
        features["split"] = "test"
    else:
        features.loc[0, "rms_mean"] = np.nan
    with pytest.raises(ValueError):
        align_inputs(train, folds, features)


def paired():
    train, folds, features = inputs()
    acoustic = evaluate_acoustic_baseline(train, folds, features).oof
    e005 = acoustic.copy()
    e005["prediction"] = train.label + np.linspace(-0.6, 0.7, len(train))
    # Stored residuals deliberately stale: diagnostics must recompute them.
    return acoustic, e005


@pytest.mark.parametrize("column", ["filename", "label", "fold", "prediction"])
def test_e005_mismatch_rejected(column):
    acoustic, e005 = paired()
    e005.loc[0, column] = (
        "other.wav"
        if column == "filename"
        else np.nan
        if column == "prediction"
        else 99
    )
    with pytest.raises(ValueError):
        align_oof(acoustic, e005)


def test_complementarity_and_alignment():
    acoustic, e005 = paired()
    aligned = align_oof(acoustic, e005.iloc[::-1])
    values = complementarity(aligned)
    assert values["pooled"]["prediction_correlation"] == pytest.approx(
        pearson_correlation(acoustic.prediction, e005.prediction)
    )
    assert values["pooled"]["residual_correlation"] == pytest.approx(
        pearson_correlation(
            acoustic.label - acoustic.prediction, e005.label - e005.prediction
        )
    )
    assert len(values["per_fold"]) == 5
    for row in values["per_fold"]:
        subset = aligned[aligned.fold == row["fold"]]
        assert row["prediction_correlation"] == pytest.approx(
            pearson_correlation(subset.acoustic_prediction, subset.e005_prediction)
        )


def test_blend_equation_and_per_fold_diagnostics():
    acoustic, e005 = paired()
    aligned = align_oof(acoustic, e005)
    grid = blend_grid(aligned)
    assert len(grid) == 11
    pd.testing.assert_frame_equal(grid, blend_grid(aligned))
    for _, row in grid.iterrows():
        weight = row.e005_weight
        predictions = weight * e005.prediction + (1 - weight) * acoustic.prediction
        metrics = regression_metrics(acoustic.label, predictions)
        assert row.rmse == pytest.approx(metrics["rmse"])
        assert row.pearson_correlation == pytest.approx(metrics["pearson_correlation"])
        rmse_deltas, pearson_deltas = [], []
        for fold in range(5):
            mask = acoustic.fold == fold
            baseline = regression_metrics(e005.label[mask], e005.prediction[mask])
            blended = regression_metrics(acoustic.label[mask], predictions[mask])
            assert row[f"fold_{fold}_rmse"] == pytest.approx(blended["rmse"])
            assert row[f"fold_{fold}_pearson_correlation"] == pytest.approx(
                blended["pearson_correlation"]
            )
            rmse_deltas.append(blended["rmse"] - baseline["rmse"])
            pearson_deltas.append(
                blended["pearson_correlation"] - baseline["pearson_correlation"]
            )
        assert row.folds_improving_rmse == sum(delta < 0 for delta in rmse_deltas)
        assert row.folds_improving_pearson == sum(delta > 0 for delta in pearson_deltas)
        assert row.worst_fold_rmse_delta == pytest.approx(max(rmse_deltas))
        assert row.worst_fold_pearson_delta == pytest.approx(min(pearson_deltas))
    assert grid.iloc[-1].folds_improving_rmse == 0
    assert grid.iloc[-1].worst_fold_rmse_delta == 0


def test_constant_correlations_are_undefined():
    acoustic, e005 = paired()
    acoustic["prediction"] = 1.0
    aligned = align_oof(acoustic, e005)
    assert np.isnan(complementarity(aligned)["pooled"]["prediction_correlation"])
