import json
import sys

import numpy as np
import pandas as pd
import pytest

from grammar_scoring.inference import cli


def record(filename, text=" raw  ASR\ntext! ", split="test"):
    return {
        "split": split,
        "filename": filename,
        "text": text,
        "language": "en",
        "duration_seconds": 1,
        "segments": [],
    }


def inputs(tmp_path, names=None, records=None):
    test = tmp_path / "test.csv"
    transcripts = tmp_path / "test.jsonl"
    pd.DataFrame(
        {"filename": names if names is not None else ["b.wav", "a.wav"]}
    ).to_csv(test, index=False)
    rows = records if records is not None else [record("a.wav"), record("b.wav")]
    transcripts.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return test, transcripts


@pytest.mark.parametrize("device", ["auto", "cpu", "cuda"])
def test_cli_aligns_raw_text_and_writes_unchanged_values(
    tmp_path, monkeypatch, capsys, device
):
    test, transcripts = inputs(
        tmp_path, records=[record("a.wav", " a  "), record("b.wav", " B\n! ")]
    )
    output = tmp_path / "new" / "nested" / "predictions.csv"
    model = tmp_path / "custom_models"
    # The mocked ensemble must work even when optional model packages are absent.
    monkeypatch.setitem(sys.modules, "transformers", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    calls = []

    def predict(frame, model_dir, *, device):
        calls.append((frame.copy(), model_dir, device))
        return pd.DataFrame(
            {"filename": frame.filename.tolist(), "label": [-0.123456789, 6.987654321]}
        )

    monkeypatch.setattr(cli, "predict_e005_ensemble", predict)
    assert (
        cli.main(
            [
                "--test-csv",
                str(test),
                "--transcripts",
                str(transcripts),
                "--model-dir",
                str(model),
                "--output",
                str(output),
                "--device",
                device,
            ]
        )
        == 0
    )
    frame, model_dir, chosen = calls[0]
    assert frame.columns.tolist() == ["filename", "text"]
    assert frame.filename.tolist() == ["b.wav", "a.wav"]
    assert frame.text.tolist() == [" B\n! ", " a  "]
    assert model_dir == model
    assert chosen == device
    saved = pd.read_csv(output)
    assert saved.columns.tolist() == ["filename", "label"]
    assert saved.filename.tolist() == frame.filename.tolist()
    assert saved.label.tolist() == [-0.123456789, 6.987654321]
    summary = capsys.readouterr().out
    assert "2 predictions" in summary and str(output) in summary
    assert all(name in summary for name in ("min=", "max=", "mean="))


@pytest.mark.parametrize(
    ("names", "records", "message"),
    [
        (["b.wav", "a.wav"], [record("a.wav")], "missing transcripts"),
        (["a.wav"], [record("a.wav"), record("x.wav")], "unexpected transcripts"),
        (["a.wav", "a.wav"], [record("a.wav")], "Test filenames must be unique"),
        (["a.wav"], [record("a.wav"), record("a.wav")], "duplicate identity"),
        (
            ["a.wav"],
            [record("a.wav"), record("a.wav", split="train")],
            "Transcript filenames must be unique",
        ),
        (["a.wav"], [record("a.wav", split="train")], "split 'test'"),
        (["a.wav"], [], "missing transcripts"),
        ([], [], "at least one example"),
        ([""], [], "nonempty strings"),
    ],
)
def test_invalid_alignment_fails(tmp_path, names, records, message):
    test, transcripts = inputs(tmp_path, names, records)
    with pytest.raises(ValueError, match=message):
        cli.load_test_inputs(test, transcripts)


@pytest.mark.parametrize("text", ["", "  \n", None, 123])
def test_empty_or_invalid_transcript_fails(tmp_path, text):
    test, transcripts = inputs(tmp_path, ["a.wav"], [record("a.wav", text)])
    with pytest.raises(ValueError, match="text|transcripts"):
        cli.load_test_inputs(test, transcripts)


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (pd.DataFrame({"filename": ["b.wav"], "label": [1.0]}), "count"),
        (pd.DataFrame({"filename": ["a.wav", "b.wav"], "label": [1, 2]}), "order"),
        (pd.DataFrame({"filename": ["b.wav", "x.wav"], "label": [1, 2]}), "order"),
        (pd.DataFrame({"filename": ["b.wav", "b.wav"], "label": [1, 2]}), "unique"),
        (pd.DataFrame({"label": [1, 2], "filename": ["b.wav", "a.wav"]}), "columns"),
        (
            pd.DataFrame(
                {"filename": ["b.wav", "a.wav"], "label": [1, 2], "extra": [0, 0]}
            ),
            "columns",
        ),
        *[
            (
                pd.DataFrame({"filename": ["b.wav", "a.wav"], "label": [value, value]}),
                "finite",
            )
            for value in [np.nan, np.inf, -np.inf]
        ],
        *[
            (
                pd.DataFrame({"filename": ["b.wav", "a.wav"], "label": [value, value]}),
                "numeric",
            )
            for value in ["bad", "1.2", True, 1j]
        ],
    ],
)
def test_invalid_predictions_fail_before_writing(
    tmp_path, monkeypatch, result, message
):
    test, transcripts = inputs(tmp_path)
    output = tmp_path / "absent" / "predictions.csv"
    monkeypatch.setattr(cli, "predict_e005_ensemble", lambda *a, **kw: result)
    with pytest.raises(ValueError, match=message):
        cli.main(
            [
                "--test-csv",
                str(test),
                "--transcripts",
                str(transcripts),
                "--output",
                str(output),
            ]
        )
    assert not output.parent.exists()


def test_defaults_use_configured_paths(monkeypatch, tmp_path):
    test, transcripts = inputs(tmp_path)
    monkeypatch.setattr(cli, "TEST_CSV", test)
    monkeypatch.setattr(cli, "TRANSCRIPTS_DIR", tmp_path)
    monkeypatch.setattr(cli, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cli, "ARTIFACT_DIR", tmp_path / "artifacts")

    def predict(frame, model_dir, *, device):
        assert model_dir == tmp_path / "models" / "E005"
        assert device == "auto"
        return frame[["filename"]].assign(label=2.5)

    monkeypatch.setattr(cli, "predict_e005_ensemble", predict)
    assert cli.main([]) == 0
    assert (tmp_path / "artifacts/submissions/E005_test_predictions.csv").is_file()


def test_invalid_device_fails():
    with pytest.raises(SystemExit):
        cli.main(["--device", "mps"])
