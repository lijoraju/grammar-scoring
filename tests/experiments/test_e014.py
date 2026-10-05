import json

import numpy as np
import pandas as pd
import pytest

from grammar_scoring.experiments.e014_ensemble import (
    cross_fitted_calibration,
    ensemble_predictions,
    ensemble_report,
    fit_line,
    load_groups,
    load_runs,
)
from grammar_scoring.experiments.e014_finetune import (
    E014Config,
    build_frames,
    layer_depth,
    load_transcripts,
)


def _write_jsonl(path, names):
    path.write_text(
        "\n".join(json.dumps({"filename": n, "text": f" text {n} "}) for n in names)
    )


@pytest.fixture
def inputs(tmp_path):
    names = [f"audio_{i}.wav" for i in range(12)]
    labels = [0.0, 0.0] + [1.0 + (i % 9) / 2 for i in range(10)]
    folds = [i % 5 for i in range(12)]
    pd.DataFrame({"filename": names, "label": labels}).to_csv(
        tmp_path / "train.csv", index=False
    )
    pd.DataFrame({"filename": names, "label": labels, "fold": folds}).to_csv(
        tmp_path / "folds.csv", index=False
    )
    test_names = ["audio_t1.wav", "audio_t0.wav"]
    pd.DataFrame({"filename": test_names, "label": [-1, -1]}).to_csv(
        tmp_path / "test.csv", index=False
    )
    _write_jsonl(tmp_path / "train.jsonl", names)
    _write_jsonl(tmp_path / "test.jsonl", test_names)
    return tmp_path


def _frames(root):
    return build_frames(
        root / "train.csv",
        root / "test.csv",
        root / "folds.csv",
        root / "train.jsonl",
        root / "test.jsonl",
    )


def test_build_frames_drops_zero_labels_and_keeps_folds(inputs):
    train, test = _frames(inputs)
    assert len(train) == 10
    assert (train["label"] > 0).all()
    assert train["fold"].tolist() == [i % 5 for i in range(2, 12)]
    assert train["text"].iloc[0] == "text audio_2.wav"
    assert test["filename"].tolist() == ["audio_t1.wav", "audio_t0.wav"]


def test_build_frames_rejects_label_mismatch(inputs):
    folds = pd.read_csv(inputs / "folds.csv")
    folds.loc[5, "label"] += 1
    folds.to_csv(inputs / "folds.csv", index=False)
    with pytest.raises(ValueError, match="disagree"):
        _frames(inputs)


def test_build_frames_rejects_missing_transcript(inputs):
    _write_jsonl(inputs / "test.jsonl", ["audio_t1.wav"])
    with pytest.raises(ValueError, match="lack transcripts"):
        _frames(inputs)


def test_load_transcripts_rejects_duplicates(tmp_path):
    _write_jsonl(tmp_path / "t.jsonl", ["a.wav", "a.wav"])
    with pytest.raises(ValueError, match="Duplicate"):
        load_transcripts(tmp_path / "t.jsonl")


def test_layer_depth():
    assert layer_depth("backbone.embeddings.word_embeddings.weight", 24) == 0
    assert layer_depth("backbone.encoder.layer.0.output.dense.weight", 24) == 1
    assert layer_depth("backbone.encoder.layer.23.output.dense.bias", 24) == 24
    assert layer_depth("backbone.encoder.rel_embeddings.weight", 24) == 24


def test_run_name():
    config = E014Config(model_name="microsoft/deberta-v3-large", seed=7)
    assert config.run_name == "deberta-v3-large_lr2e-05_s7"


def test_fit_line_recovers_exact_relation():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert fit_line(x, 0.5 + 2 * x) == pytest.approx((0.5, 2.0))


def test_cross_fitted_calibration_never_uses_held_out_labels():
    rng = np.random.default_rng(0)
    pred = rng.normal(size=50)
    labels = 1 + 2 * pred
    folds = np.arange(50) % 5
    corrupted = labels.copy()
    corrupted[folds == 0] += 100
    a = cross_fitted_calibration(pred, labels, folds)
    b = cross_fitted_calibration(pred, corrupted, folds)
    assert np.allclose(a[folds == 0], b[folds == 0])
    assert np.allclose(a, labels)


def _write_run(root, name, oof_pred, test_pred, labels, folds):
    directory = root / name
    directory.mkdir()
    names = [f"f{i}" for i in range(len(labels))]
    pd.DataFrame(
        {
            "filename": names,
            "label": labels,
            "fold": folds,
            "prediction": oof_pred,
            "prediction_last": oof_pred,
        }
    ).to_csv(directory / "oof.csv", index=False)
    pd.DataFrame(
        {
            "filename": ["t0", "t1"],
            "prediction": test_pred,
            "prediction_last": test_pred,
        }
    ).to_csv(directory / "test.csv", index=False)
    return directory


def test_load_runs_and_ensemble(tmp_path):
    labels = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 2)
    folds = np.arange(10) % 5
    a = _write_run(tmp_path, "a", 3 + 0.5 * (labels - 3), [2.0, 4.0], labels, folds)
    b = _write_run(tmp_path, "b", 3 + 0.5 * (labels - 3), [3.0, 5.0], labels, folds)
    oof, test = load_runs([a, b])
    assert list(oof.columns) == ["filename", "label", "fold", "a", "b"]
    report = ensemble_report(oof, ["a", "b"])
    assert report["mean_calibrated_cv"]["rmse"] == pytest.approx(0.0, abs=1e-9)
    raw = ensemble_predictions(oof, test, ["a", "b"], calibrate=False)
    assert raw["label"].tolist() == [2.5, 4.5]
    calibrated = ensemble_predictions(oof, test, ["a", "b"], calibrate=True)
    assert calibrated["label"].tolist() == pytest.approx([2.0, 6.0])


def test_load_runs_rejects_misaligned_runs(tmp_path):
    labels = np.array([1.0, 2.0, 3.0])
    folds = np.array([0, 1, 2])
    a = _write_run(tmp_path, "a", labels, [1.0, 2.0], labels, folds)
    b = _write_run(tmp_path, "b", labels, [1.0, 2.0], labels + 1, folds)
    with pytest.raises(ValueError, match="disagree"):
        load_runs([a, b])


def test_load_groups_adds_equal_weight_group_columns(tmp_path):
    labels = np.array([1.0, 2.0, 3.0])
    folds = np.array([0, 1, 2])
    a = _write_run(tmp_path, "a", labels, [1.0, 3.0], labels, folds)
    b = _write_run(tmp_path, "b", labels + 1, [2.0, 4.0], labels, folds)
    c = _write_run(tmp_path, "c", labels + 3, [5.0, 5.0], labels, folds)
    oof, test = load_groups({"g1": [a, b], "g2": [c]})
    assert oof["g1"].tolist() == pytest.approx((labels + 0.5).tolist())
    assert test["g2"].tolist() == [5.0, 5.0]
    with pytest.raises(ValueError, match="unique"):
        load_groups({"g1": [a], "g2": [a]})
    with pytest.raises(ValueError, match="at least one"):
        load_groups({"g1": [a], "g2": []})
