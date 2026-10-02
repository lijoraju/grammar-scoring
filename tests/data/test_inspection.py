import importlib.util
import wave
from pathlib import Path

import pandas as pd
import pytest

from grammar_scoring.data import dataset
from grammar_scoring.data.inspection import (
    format_audio_report,
    inspect_dataset_audio,
)


def write_wav(path, sample_rate=10, frames=450):
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(bytes(frames * 2))


@pytest.fixture
def configured_dataset(tmp_path, monkeypatch):
    for split in ("train", "test"):
        directory = tmp_path / split
        directory.mkdir()
        write_wav(directory / "normal.wav", sample_rate=10, frames=450)
        pd.DataFrame({"filename": ["normal.wav"], "label": [3]}).to_csv(
            tmp_path / f"{split}.csv", index=False
        )
        monkeypatch.setattr(dataset, f"{split.upper()}_CSV", tmp_path / f"{split}.csv")
        monkeypatch.setattr(dataset, f"{split.upper()}_AUDIO_DIR", directory)
    submission = tmp_path / "sample_submission.csv"
    pd.DataFrame({"filename": ["normal.wav"], "label": [0]}).to_csv(
        submission, index=False
    )
    monkeypatch.setattr(dataset, "SAMPLE_SUBMISSION_CSV", submission)
    return tmp_path


@pytest.fixture
def cli():
    path = Path(__file__).resolve().parents[2] / "scripts" / "inspect_audio.py"
    spec = importlib.util.spec_from_file_location("inspect_audio_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inspection_includes_extras_and_continues_after_errors(configured_dataset):
    directory = configured_dataset / "train"
    for name, seconds in (
        ("short.wav", 20),
        ("shorter.wav", 10),
        ("boundary_short.wav", 30),
        ("boundary_long.wav", 90),
        ("long.wav", 100),
    ):
        write_wav(directory / name, sample_rate=10, frames=seconds * 10)
    (directory / "broken.wav").write_bytes(b"invalid")
    write_wav(directory / "zero.wav", frames=0)
    train, test = inspect_dataset_audio()
    assert train.csv_rows == test.csv_rows == 1
    assert len(train.metadata) == 6
    assert [path.name for path, _ in train.errors] == ["broken.wav", "zero.wav"]
    assert train.metadata.filename.tolist() == sorted(train.metadata.filename)
    report = format_audio_report(train)
    assert "Short recordings (<30 seconds): 2" in report
    assert "Long recordings (>90 seconds): 1" in report
    assert report.index("shorter.wav:") < report.index("short.wav:")
    assert "min=10.000, max=100.000, mean=49.167, median=37.500" in report
    assert "total=295.000" in report
    assert "Sample rates (Hz): 10: 6" in report
    assert "Channel counts: 1: 6" in report
    assert "Sample widths (bytes): 2: 6" in report
    # Threshold anomalies have no effect on dataset integrity validation.
    dataset.validate_dataset()


def test_cli_default_and_export(configured_dataset, cli, capsys):
    before = set(configured_dataset.rglob("*"))
    assert cli.main([]) == 0
    assert set(configured_dataset.rglob("*")) == before
    first = capsys.readouterr().out
    assert cli.main([]) == 0
    assert capsys.readouterr().out == first
    output = configured_dataset / "metadata.csv"
    assert cli.main(["--output", str(output)]) == 0
    exported = pd.read_csv(output)
    assert exported.split.tolist() == ["train", "test"]
    assert exported.duration_seconds.tolist() == [45, 45]
    assert set(exported.columns) == {
        "split",
        "filename",
        "path",
        "duration_seconds",
        "sample_rate",
        "num_channels",
        "sample_width_bytes",
        "num_frames",
    }


def test_cli_fatal_validation_error(configured_dataset, cli, capsys):
    dataset.TRAIN_CSV.unlink()
    output = configured_dataset / "metadata.csv"
    assert cli.main(["--output", str(output)]) == 1
    assert "Inspection failed:" in capsys.readouterr().err
    assert not output.exists()


def test_cli_no_readable_files(configured_dataset, cli, capsys):
    (configured_dataset / "train" / "normal.wav").write_bytes(b"invalid")
    assert cli.main([]) == 1
    captured = capsys.readouterr()
    assert "normal.wav" in captured.out
    assert "no readable WAV files" in captured.err


def test_cli_export_failure(configured_dataset, cli, capsys):
    assert cli.main(["--output", str(configured_dataset / "missing" / "out.csv")]) == 1
    assert "Inspection failed:" in capsys.readouterr().err
