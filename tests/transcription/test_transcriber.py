import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from grammar_scoring.transcription.transcriber import (
    TranscriptionResult,
    TranscriptionSegment,
    WhisperTranscriber,
    WhisperTranscriptionConfig,
    read_jsonl,
    transcribe_files,
    write_jsonl,
)


class FakeModel:
    def __init__(self, texts=(" Um,", " I I goed. é")):
        self.texts = texts
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return iter(
            [
                SimpleNamespace(start=i, end=i + 1, text=text)
                for i, text in enumerate(self.texts)
            ]
        ), SimpleNamespace(language="en", duration=3.5)


def audio(tmp_path, name="same.wav"):
    path = tmp_path / name
    path.touch()
    return path


def test_defaults():
    assert asdict(WhisperTranscriptionConfig()) == {
        "model_name": "large-v3",
        "device": "cuda",
        "compute_type": "float16",
        "language": "en",
        "beam_size": 5,
        "condition_on_previous_text": False,
        "vad_filter": False,
        "word_timestamps": False,
    }


def test_conversion_and_model_reuse(tmp_path):
    model = FakeModel()
    transcriber = WhisperTranscriber(model=model)
    path = audio(tmp_path)
    result = transcriber.transcribe(path, split="train")
    assert result.key == ("train", "same.wav")
    assert result.text == " Um, I I goed. é"
    assert result.segments == (
        TranscriptionSegment(0, 1, " Um,"),
        TranscriptionSegment(1, 2, " I I goed. é"),
    )
    assert result.duration_seconds == 3.5
    assert result.language == "en"
    transcriber.transcribe(path, split="test")
    assert len(model.calls) == 2
    assert model.calls[0][1] == {
        "language": "en",
        "beam_size": 5,
        "condition_on_previous_text": False,
        "vad_filter": False,
        "word_timestamps": False,
    }


def test_empty(tmp_path):
    result = WhisperTranscriber(model=FakeModel(())).transcribe(
        audio(tmp_path), split="test"
    )
    assert result.text == ""
    assert result.segments == ()


@pytest.mark.parametrize(
    "start,end", [(-1, 2), (2, 1), (float("nan"), 2), (0, float("inf"))]
)
def test_bad_timestamps(start, end):
    with pytest.raises(ValueError):
        TranscriptionSegment(start, end, "text")


def test_validation(tmp_path):
    transcriber = WhisperTranscriber(model=FakeModel())
    with pytest.raises(ValueError, match="split"):
        transcriber.transcribe(audio(tmp_path), split="validation")
    with pytest.raises(FileNotFoundError):
        transcriber.transcribe(tmp_path / "absent.wav", split="train")


def test_round_trip_and_overlap(tmp_path):
    transcriber = WhisperTranscriber(model=FakeModel())
    path = audio(tmp_path)
    results = [transcriber.transcribe(path, split=s) for s in ("train", "test")]
    output = tmp_path / "transcripts.jsonl"
    write_jsonl(output, results)
    original = output.read_bytes()
    assert read_jsonl(output) == results
    write_jsonl(output, results)
    assert output.read_bytes() == original
    assert "é" in output.read_text()
    assert "label" not in json.loads(output.read_text().splitlines()[0])
    with pytest.raises(ValueError, match="Duplicate"):
        write_jsonl(output, [results[0], results[0]])


def test_resume_exact_identity_and_duplicates(tmp_path):
    model = FakeModel()
    transcriber = WhisperTranscriber(model=model)
    path = audio(tmp_path)
    output = tmp_path / "out.jsonl"
    assert transcribe_files([("train", path)], output, transcriber) == (1, 0)
    assert transcribe_files(
        [("train", path), ("test", path), ("test", path)], output, transcriber
    ) == (1, 2)
    assert len(model.calls) == 2
    assert [r.key for r in read_jsonl(output)] == [
        ("train", "same.wav"),
        ("test", "same.wav"),
    ]
    metadata = json.loads(output.with_suffix(".meta.json").read_text())
    assert metadata["backend"] == "faster-whisper"
    assert set(metadata["versions"]) == {"av", "ctranslate2", "faster-whisper"}


def test_metadata_mismatch_and_overwrite(tmp_path):
    path = audio(tmp_path)
    output = tmp_path / "out.jsonl"
    transcriber = WhisperTranscriber(model=FakeModel())
    transcribe_files([("train", path)], output, transcriber)
    changed = WhisperTranscriber(
        WhisperTranscriptionConfig(beam_size=1), model=FakeModel()
    )
    with pytest.raises(ValueError, match="metadata"):
        transcribe_files([("train", path)], output, changed)
    assert transcribe_files([("train", path)], output, changed, overwrite=True) == (
        1,
        0,
    )
    assert len(read_jsonl(output)) == 1


def test_interruption_keeps_completed_records(tmp_path):
    path = audio(tmp_path)
    output = tmp_path / "out.jsonl"
    transcriber = WhisperTranscriber(model=FakeModel())
    with pytest.raises(FileNotFoundError):
        transcribe_files(
            [("train", path), ("train", tmp_path / "absent.wav")],
            output,
            transcriber,
        )
    assert len(read_jsonl(output)) == 1
    assert transcribe_files([("train", path)], output, transcriber) == (0, 1)


def test_wrong_identity_rejected(tmp_path):
    class WrongTranscriber(WhisperTranscriber):
        def transcribe(self, path, *, split):
            return TranscriptionResult("test", path.name, "", "en", 0, ())

    with pytest.raises(ValueError, match="identity"):
        transcribe_files(
            [("train", audio(tmp_path))], tmp_path / "out.jsonl", WrongTranscriber()
        )
