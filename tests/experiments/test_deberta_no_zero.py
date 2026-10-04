import json

import numpy as np
import pandas as pd
import pytest

from grammar_scoring.experiments import deberta_no_zero as e008


@pytest.fixture
def canonical():
    labels = np.resize([1.0, 2.5, 3.0, 3.5, 4.0, 5.0], 769)
    labels[:37] = 0
    return pd.DataFrame(
        {
            "filename": [f"audio_{i}.wav" for i in range(769)],
            "label": labels,
            "text": [" Raw Whisper text! "] * 769,
            "fold": np.repeat(range(5), [154, 154, 154, 154, 153]),
        }
    )


def predictions(frame, offset):
    return frame[["filename", "label", "fold"]].assign(prediction=frame.label + offset)


def test_population_criterion_preserves_folds_and_raw_text(canonical):
    canonical.loc[38, "filename"] = "audio_5037.wav"
    retained = e008.filter_population(canonical)
    assert len(retained) == 732
    assert "audio_5037.wav" in set(retained.filename)
    assert "audio_0.wav" not in set(retained.filename)
    expected = canonical.loc[canonical.label > 0].reset_index(drop=True)
    pd.testing.assert_frame_equal(retained, expected)


@pytest.mark.parametrize("damage", ["size", "zeros", "negative", "folds", "test"])
def test_population_invariants(canonical, damage):
    if damage == "size":
        canonical = canonical.iloc[:-1]
    elif damage == "zeros":
        canonical.loc[37, "label"] = 0
    elif damage == "negative":
        canonical.loc[0, "label"] = -1
    elif damage == "folds":
        canonical.loc[0, "fold"] = 1
    else:
        canonical["split"] = "test"
    with pytest.raises(ValueError):
        e008.filter_population(canonical)


@pytest.mark.parametrize(
    "damage", ["zero", "negative", "duplicate", "missing", "nan", "fold"]
)
def test_oof_rejections(canonical, damage):
    retained = e008.filter_population(canonical)
    oof = predictions(retained, 0.1)
    if damage in {"zero", "negative"}:
        oof.loc[0, "label"] = 0 if damage == "zero" else -1
    elif damage == "duplicate":
        oof.loc[1, "filename"] = oof.loc[0, "filename"]
    elif damage == "missing":
        oof = oof.iloc[:-1]
    elif damage == "nan":
        oof.loc[0, "prediction"] = np.nan
    else:
        oof.loc[0, "fold"] = 1
    with pytest.raises(ValueError):
        e008.validate_oof(oof, retained)


@pytest.mark.parametrize("column", ["label", "fold"])
def test_baseline_alignment_detects_mismatch(canonical, column):
    retained = e008.filter_population(canonical)
    baseline = predictions(canonical, 0.2)
    baseline.loc[38, column] += 1
    with pytest.raises(ValueError, match=column):
        e008.align_baseline(baseline, retained)


def test_paired_diagnostics_identical_population(canonical, tmp_path):
    retained = e008.filter_population(canonical)
    baseline = predictions(canonical, 0.2)
    baseline.loc[:36, "prediction"] = 100
    oof = predictions(retained, 0.1)
    selected = [{"fold": i, "selected_epoch": 2} for i in range(5)]
    report = e008.diagnostics(
        oof.iloc[::-1],
        baseline.sample(frac=1, random_state=42),
        retained,
        selected,
        tmp_path,
    )
    assert report["n"] == 732
    assert report["e005_rmse"] == pytest.approx(0.2)
    assert report["e008_rmse"] == pytest.approx(0.1)
    assert report["rmse_delta"] == pytest.approx(-0.1)
    assert report["relative_rmse_change_percent"] == pytest.approx(-50)
    assert report["folds_improving_rmse"] == 5
    assert all(row["selected_epoch"] == 2 for row in report["per_fold"])
    groups = pd.read_csv(tmp_path / "score_groups.csv")
    assert groups.n.sum() == 732
    assert groups.group.tolist() == ["low", "mid", "high"]
    assert np.allclose(groups.rmse_delta, -0.1)
    assert len(pd.read_csv(tmp_path / "largest_improvements.csv")) == 20
    assert len(pd.read_csv(tmp_path / "largest_deteriorations.csv")) == 20
    assert json.loads((tmp_path / "e005_comparison.json").read_text())["n"] == 732


def test_score_group_boundaries_and_empty_group():
    joined = pd.DataFrame({"label": [1.0, 2.5, 2.9, 3.0, 3.5, 3.9, 4.0, 5.0]})
    joined["prediction"] = joined.label
    joined["e005_prediction"] = joined.label + 1
    groups = e008.score_groups(joined)
    assert groups.n.tolist() == [3, 3, 2]
    assert groups.rmse_delta.tolist() == [-1, -1, -1]
    empty = e008.score_groups(joined.iloc[:3])
    assert empty.n.tolist() == [3, 0, 0]


def test_runner_reuses_frozen_protocol_and_stops_at_oof(
    canonical, tmp_path, monkeypatch
):
    baseline = predictions(canonical, 0.2)
    calls = []

    def run_shared(retained, artifact_dir, **kwargs):
        calls.append(kwargs)
        assert len(retained) == 732 and retained.label.gt(0).all()
        output = artifact_dir / "oof" / f"{e008.OOF_NAME}.csv"
        output.parent.mkdir()
        predictions(retained, 0.1).to_csv(output, index=False)
        return {
            "experiment": "E008",
            "per_fold": [{"fold": i, "selected_epoch": 2} for i in range(5)],
        }

    monkeypatch.setattr(e008.e005, "_run_aligned_experiment", run_shared)
    result = e008.run_experiment(canonical, baseline, tmp_path, resume=True)
    assert calls == [
        {
            "experiment": "E008",
            "oof_name": e008.OOF_NAME,
            "device": "cuda",
            "resume": True,
        }
    ]
    assert result["e005_comparison"]["n"] == 732
    assert not list(tmp_path.rglob("*submission*"))
    config = json.loads(
        (tmp_path / "experiments/E008/resolved_config.json").read_text()
    )
    assert config["training_protocol"]["seed"] == 42
    assert config["training_protocol"]["max_epochs"] == 5
    baseline.loc[38, "fold"] = 2
    with pytest.raises(ValueError):
        e008.run_experiment(canonical, baseline, tmp_path)
    assert len(calls) == 1


def test_e008_smoke_population_and_namespace(canonical, tmp_path, monkeypatch):
    from grammar_scoring.experiments import deberta_smoke

    retained = e008.filter_population(canonical)
    training, validation = deberta_smoke.smoke_subset(retained)
    assert training.label.gt(0).all() and validation.label.gt(0).all()
    assert len(training) == 32 and len(validation) == 16
    assert set(training.filename).isdisjoint(validation.filename)
    assert deberta_smoke.smoke_root(tmp_path / "smoke/E008", "E008").name == "E008"
    with pytest.raises(ValueError):
        deberta_smoke.smoke_root(tmp_path / "models/E008", "E008")
