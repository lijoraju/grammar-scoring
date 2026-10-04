import numpy as np
import pandas as pd
import pytest

from grammar_scoring.inference import cli


def frame(size=216):
    return pd.DataFrame(
        {
            "filename": [f"test_{i}.wav" for i in reversed(range(size))],
            "text": ["raw transcript"] * size,
        }
    )


@pytest.mark.parametrize("experiment", ["E005", "E008"])
def test_defaults_and_order(monkeypatch, tmp_path, experiment):
    canonical = frame()
    calls = []
    monkeypatch.setattr(cli, "ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(cli, "MODELS_DIR", tmp_path / "separate_models")

    def load(test_csv, transcripts):
        assert test_csv == cli.TEST_CSV
        assert transcripts == cli.TRANSCRIPTS_DIR / "test.jsonl"
        return canonical

    def predict(inputs, model_dir, *, device, experiment="E005"):
        calls.append((model_dir, device, experiment))
        return inputs[["filename"]].assign(label=np.linspace(-0.75, 6.25, 216))

    monkeypatch.setattr(cli, "load_test_inputs", load)
    monkeypatch.setattr(cli, "predict_deberta_ensemble", predict)
    monkeypatch.setattr(cli, "predict_e005_ensemble", predict)
    entry_point = cli.main if experiment == "E005" else cli.main_e008
    assert entry_point([]) == 0
    assert calls == [(cli.MODELS_DIR / experiment, "auto", experiment)]
    saved = pd.read_csv(
        cli.ARTIFACT_DIR / "submissions" / f"{experiment}_test_predictions.csv"
    )
    assert len(saved) == 216
    assert saved.columns.tolist() == ["filename", "label"]
    assert saved.filename.tolist() == canonical.filename.tolist()
    np.testing.assert_allclose(saved.label, np.linspace(-0.75, 6.25, 216))


@pytest.mark.parametrize("size", [215, 217])
def test_e008_requires_216_rows(monkeypatch, size):
    monkeypatch.setattr(cli, "load_test_inputs", lambda *args: frame(size))
    with pytest.raises(ValueError, match="exactly 216"):
        cli.main_e008([])


@pytest.mark.parametrize("invalid", ["order", "duplicate", "count", "nonfinite"])
def test_e008_rejects_invalid_predictions(monkeypatch, tmp_path, invalid):
    canonical = frame()
    result = canonical[["filename"]].assign(label=1.0)
    if invalid == "order":
        result = result.iloc[::-1]
    elif invalid == "duplicate":
        result.loc[1, "filename"] = result.loc[0, "filename"]
    elif invalid == "count":
        result = result.iloc[:-1]
    else:
        result.loc[0, "label"] = np.inf
    output = tmp_path / "predictions.csv"
    monkeypatch.setattr(cli, "load_test_inputs", lambda *args: canonical)
    monkeypatch.setattr(cli, "predict_deberta_ensemble", lambda *a, **kw: result)
    with pytest.raises(ValueError):
        cli.main_e008(["--output", str(output)])
    assert not output.exists()
