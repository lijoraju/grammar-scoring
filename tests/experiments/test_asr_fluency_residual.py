import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from grammar_scoring.experiments import asr_fluency_residual as experiment
from grammar_scoring.features.asr_fluency import FEATURE_COLUMNS


@pytest.fixture
def inputs():
    rng = np.random.default_rng(42)
    base = pd.DataFrame(
        {
            "filename": [f"sample_{i}.wav" for i in range(732)],
            "label": rng.uniform(0.5, 5, 732),
            "fold": np.arange(732) % 5,
            "prediction": rng.uniform(1, 4, 732),
        }
    )
    acoustic = pd.DataFrame(
        [
            {
                "filename": name,
                "split": "train",
                "language": "en",
                "text": "um go go" if i % 2 else "one two",
                "duration_seconds": float(i + 3),
                "segments": [
                    {"start": 0.0, "end": float(i % 7 + 1), "text": "one two"}
                ],
            }
            for i, name in enumerate(base.filename)
        ]
    )
    return base, acoustic


def test_alignment(inputs):
    base, acoustic = inputs
    expected = experiment.align_inputs(base, acoustic)
    pd.testing.assert_frame_equal(
        experiment.align_inputs(base, acoustic.sample(frac=1, random_state=1)),
        expected,
    )
    reordered = base.sample(frac=1, random_state=2)
    actual = experiment.align_inputs(reordered, acoustic)
    pd.testing.assert_frame_equal(
        actual.set_index("filename").sort_index(),
        expected.set_index("filename").sort_index(),
    )
    extra = acoustic.iloc[[0]].assign(filename="outside.wav")
    pd.testing.assert_frame_equal(
        experiment.align_inputs(base, pd.concat([acoustic, extra])), expected
    )


@pytest.mark.parametrize("source", [0, 1])
def test_duplicate_filenames(inputs, source):
    frames = list(inputs)
    frames[source].loc[1, "filename"] = frames[source].loc[0, "filename"]
    with pytest.raises(ValueError, match="unique"):
        experiment.align_inputs(*frames)


def test_missing_acoustic(inputs):
    base, acoustic = inputs
    with pytest.raises(ValueError, match="Missing transcript"):
        experiment.align_inputs(base, acoustic.iloc[1:])


@pytest.mark.parametrize(
    "column,value",
    [
        ("label", 0),
        ("label", np.nan),
        ("prediction", np.inf),
        ("fold", 0.5),
        ("fold", 5),
    ],
)
def test_invalid_baseline(inputs, column, value):
    base, acoustic = inputs
    base[column] = base[column].astype(float)
    base.loc[0, column] = value
    with pytest.raises(ValueError):
        experiment.align_inputs(base, acoustic)


def test_population_count(inputs):
    with pytest.raises(ValueError, match="732"):
        experiment.align_inputs(inputs[0].iloc[1:], inputs[1])


@pytest.mark.parametrize("stack", [False, True])
def test_fold_local_fits_and_formulas(inputs, monkeypatch, stack):
    base, acoustic = inputs
    # Poison any stored residual to prove targets are recomputed from E008 OOF.
    base["residual"] = 99999.0
    aligned = experiment.align_inputs(base, acoustic)
    x = aligned[list(FEATURE_COLUMNS)].to_numpy()
    y, baseline = base.label.to_numpy(), base.prediction.to_numpy()
    scaler_calls, model_calls = [], []
    scaler_fit, model_fit = StandardScaler.fit, Ridge.fit

    def fit_scaler(self, matrix, *args, **kwargs):
        scaler_calls.append(matrix.copy())
        return scaler_fit(self, matrix, *args, **kwargs)

    def fit_model(self, matrix, target, *args, **kwargs):
        model_calls.append((matrix.copy(), target.copy()))
        assert self.alpha == 1.0
        return model_fit(self, matrix, target, *args, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", fit_scaler)
    monkeypatch.setattr(Ridge, "fit", fit_model)
    result = experiment.evaluate(aligned, stack=stack)
    assert len(scaler_calls) == len(model_calls) == 5
    for k, (matrix, target) in enumerate(model_calls):
        valid = base.fold.to_numpy() == k
        np.testing.assert_array_equal(scaler_calls[k], x[~valid])
        expected_scaled = StandardScaler().fit_transform(x[~valid])
        np.testing.assert_allclose(matrix[:, 1:] if stack else matrix, expected_scaled)
        np.testing.assert_array_equal(
            target, y[~valid] if stack else (y - baseline)[~valid]
        )
        assert matrix.shape[1] == (20 if stack else 19)
        if stack:
            np.testing.assert_array_equal(matrix[:, 0], baseline[~valid])
    # Independent fold reconstruction checks validation transforms and raw stack
    # baseline, in addition to the intercepted training calls above.
    for k in range(5):
        valid = base.fold.to_numpy() == k
        scaler = StandardScaler().fit(x[~valid])
        train_x = scaler.transform(x[~valid])
        valid_x = scaler.transform(x[valid])
        target = (y - baseline)[~valid]
        if stack:
            train_x = np.column_stack((baseline[~valid], train_x))
            valid_x = np.column_stack((baseline[valid], valid_x))
            target = y[~valid]
        predicted = Ridge(alpha=1.0).fit(train_x, target).predict(valid_x)
        expected = predicted if stack else baseline[valid] + predicted
        np.testing.assert_allclose(result.loc[valid, "prediction"], expected)
    assert result.filename.is_unique and len(result) == 732
    assert np.isfinite(result.prediction).all()
    np.testing.assert_allclose(result.residual, y - result.prediction)
    np.testing.assert_allclose(result.abs_error, abs(result.residual))
    if not stack:
        np.testing.assert_allclose(
            result.prediction, baseline + result.predicted_correction
        )


@pytest.mark.parametrize("stack", [False, True])
def test_validation_labels_excluded(inputs, stack):
    aligned = experiment.align_inputs(*inputs)
    original = experiment.evaluate(aligned, stack=stack)
    valid = aligned.fold == 0
    # Labels alone cannot change predictions in their held-out fold.
    labels_only = aligned.copy()
    labels_only.loc[valid, "label"] += 1000
    altered = experiment.evaluate(labels_only, stack=stack)
    np.testing.assert_array_equal(
        original.loc[valid, "prediction"], altered.loc[valid, "prediction"]
    )


@pytest.mark.parametrize("stack", [False, True])
def test_deterministic_and_fold_preservation(inputs, stack):
    aligned = experiment.align_inputs(*inputs)
    first = experiment.evaluate(aligned, stack=stack)
    pd.testing.assert_frame_equal(first, experiment.evaluate(aligned, stack=stack))
    pd.testing.assert_frame_equal(
        first[["filename", "fold"]], inputs[0][["filename", "fold"]]
    )


def test_deterministic_metrics():
    frame = pd.DataFrame(
        {
            "label": [1.0, 2.0, 3.0, 4.0, 5.0],
            "fold": range(5),
            "prediction": [2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    scores = experiment.metrics(frame)
    assert scores["pooled"]["rmse"] == 1
    assert scores["pooled"]["pearson_correlation"] == pytest.approx(1)
    assert len(scores["per_fold"]) == 5
    assert np.isnan(scores["per_fold"][0]["pearson_correlation"])


def test_artifacts(inputs, tmp_path):
    base, acoustic = inputs
    base_path, acoustic_path = tmp_path / "base.csv", tmp_path / "acoustic.csv"
    base.to_csv(base_path, index=False)
    acoustic.to_json(acoustic_path, orient="records", lines=True)
    with pytest.warns(UserWarning, match="reference"):
        scores = experiment.run_experiment(base_path, acoustic_path, tmp_path)
    assert set(scores) == {"E008", "E013-residual", "E013-stack"}
    for name in ("residual", "stack"):
        output = pd.read_csv(tmp_path / "oof" / f"E013_{name}.csv")
        required = {
            "filename",
            "label",
            "fold",
            "e008_prediction",
            "prediction",
            "residual",
            "abs_error",
        }
        if name == "residual":
            required.add("predicted_correction")
        assert set(output) == required
        np.testing.assert_allclose(output.e008_prediction, base.prediction)
        assert output.filename.is_unique and len(output) == 732
    directory = tmp_path / "experiments" / "E013"
    diagnostics = json.loads((directory / "diagnostics.json").read_text())
    config = json.loads((directory / "config.json").read_text())
    assert diagnostics["provenance"] == config
    assert config["feature_columns"] == list(FEATURE_COLUMNS)
    assert len(config["fold_identity"]) == 732
    assert all(len(item["sha256"]) == 64 for item in config["inputs"].values())
    for report in diagnostics["comparisons"].values():
        assert sum(band["n"] for band in report["score_bands"].values()) == 732
        assert len(report["per_fold"]) == 5
        assert report["compelling"] == (
            report["candidate"]["rmse"] <= 0.58
            and report["candidate"]["pearson_correlation"] >= 0.7942308488890498
            and report["folds_improving_rmse"] >= 4
        )
    assert json.loads((directory / "metrics.json").read_text()) == scores
    with pytest.raises(FileExistsError):
        experiment.run_experiment(base_path, acoustic_path, tmp_path)


def test_duplicate_csv_header(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("filename,label,label\na.wav,1,2\n")
    with pytest.raises(ValueError, match="headers"):
        experiment._read_csv(path)


@pytest.mark.parametrize("field", ["text", "duration_seconds", "segments"])
def test_missing_transcript_fields(inputs, field):
    with pytest.raises(ValueError, match="Missing required"):
        experiment.align_inputs(inputs[0], inputs[1].drop(columns=field))


def test_diagnostic_semantics():
    frame = pd.DataFrame(
        {
            "label": [1.0, 2.0, 3.0, 4.0, 5.0],
            "fold": range(5),
            "e008_prediction": [2.0, 3.0, 4.0, 5.0, 6.0],
            "prediction": [1.0, 2.0, 3.0, 4.0, 5.0],
            "residual": [0.0] * 5,
            "abs_error": [0.0] * 5,
        }
    )
    report = experiment.comparison(frame)
    assert report["rmse_delta"] == -1
    assert report["folds_improving_rmse"] == 5
    assert report["samples_lower_absolute_error"] == 5
    assert report["percent_lower_absolute_error"] == 100
    assert [band["n"] for band in report["score_bands"].values()] == [2, 1, 2]
    assert all(band["rmse_delta"] == -1 for band in report["score_bands"].values())
    assert np.isnan(report["residual_correlation"])
