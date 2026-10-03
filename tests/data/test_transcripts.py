import json

import pandas as pd
import pytest

from grammar_scoring.data import dataset, transcripts


def record(filename="audio_0.wav", split="train", **updates):
    return {
        "split": split,
        "filename": filename,
        "text": "  Raw Whisper text.\nCafé!  ",
        "language": "en",
        "duration_seconds": 1.25,
        "segments": [{"start": 0, "end": 1.25, "text": " Raw Whisper text."}],
        **updates,
    }


def write_records(path, records):
    path.write_text(
        "\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8"
    )
    return path


def test_valid_jsonl(tmp_path):
    records = [record(), record(split="test", confidence=0.9)]
    path = write_records(tmp_path / "transcripts.jsonl", records)
    frame = transcripts.load_transcript_jsonl(path)
    assert frame["split"].tolist() == ["train", "test"]
    assert frame["filename"].tolist() == ["audio_0.wav", "audio_0.wav"]
    assert frame["text"].tolist() == [item["text"] for item in records]
    assert frame["segments"].tolist() == [item["segments"] for item in records]
    assert frame["duration_seconds"].tolist() == [1.25, 1.25]
    assert frame.loc[1, "confidence"] == 0.9


def test_malformed_json(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(record()) + "\n{broken\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2: malformed JSON"):
        transcripts.load_transcript_jsonl(path)


def test_duplicate_identity(tmp_path):
    path = write_records(tmp_path / "duplicate.jsonl", [record(), record()])
    with pytest.raises(ValueError, match="duplicate identity"):
        transcripts.load_transcript_jsonl(path)


@pytest.mark.parametrize("field", transcripts._REQUIRED_FIELDS)
def test_missing_required_field(tmp_path, field):
    item = record()
    del item[field]
    path = write_records(tmp_path / "missing.jsonl", [item])
    with pytest.raises(ValueError, match=f"missing required fields.*{field}"):
        transcripts.load_transcript_jsonl(path)


@pytest.mark.parametrize("field", ["split", "filename"])
@pytest.mark.parametrize("value", [None, "", "  ", 42])
def test_invalid_identity(tmp_path, field, value):
    path = write_records(tmp_path / "invalid.jsonl", [record(**{field: value})])
    with pytest.raises(ValueError, match=f"{field} must be a nonempty string"):
        transcripts.load_transcript_jsonl(path)


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("text", None, "text must be a string"),
        ("text", 123, "text must be a string"),
        ("duration_seconds", -1, "finite and non-negative"),
        ("duration_seconds", float("nan"), "finite and non-negative"),
        ("duration_seconds", float("inf"), "finite and non-negative"),
        ("duration_seconds", float("-inf"), "finite and non-negative"),
        ("duration_seconds", "1.0", "finite and non-negative"),
        ("duration_seconds", None, "finite and non-negative"),
        ("duration_seconds", True, "finite and non-negative"),
        ("duration_seconds", 10**400, "finite and non-negative"),
        ("segments", {}, "segments must be a list"),
        ("segments", None, "segments must be a list"),
    ],
)
def test_invalid_field(tmp_path, field, value, message):
    path = write_records(tmp_path / "invalid.jsonl", [record(**{field: value})])
    with pytest.raises(ValueError, match=message):
        transcripts.load_transcript_jsonl(path)


@pytest.mark.parametrize("item", [[], None, "text", 1])
def test_non_object_record(tmp_path, item):
    path = write_records(tmp_path / "invalid.jsonl", [item])
    with pytest.raises(ValueError, match="expected a JSON object"):
        transcripts.load_transcript_jsonl(path)


def test_empty_and_blank_lines(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n  \n", encoding="utf-8")
    frame = transcripts.load_transcript_jsonl(path)
    assert frame.empty
    assert set(frame.columns) == set(record())
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record(text="", duration_seconds=0, segments=[])))
    assert transcripts.load_transcript_jsonl(path).loc[0, "text"] == ""


def test_missing_artifact(tmp_path):
    with pytest.raises(FileNotFoundError):
        transcripts.load_transcript_jsonl(tmp_path / "missing.jsonl")


def test_invalid_encoding(tmp_path):
    path = tmp_path / "invalid.jsonl"
    path.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="valid UTF-8"):
        transcripts.load_transcript_jsonl(path)


@pytest.fixture
def competition(tmp_path, monkeypatch):
    monkeypatch.setattr(transcripts, "TRANSCRIPTS_DIR", tmp_path)
    for split, labels in [("train", [4, 2]), ("test", [-1, -1])]:
        path = tmp_path / f"{split}.csv"
        pd.DataFrame(
            {"filename": ["audio_1.wav", "audio_0.wav"], "label": labels}
        ).to_csv(path, index=False)
        monkeypatch.setattr(dataset, f"{split.upper()}_CSV", path)
        write_records(
            tmp_path / f"{split}.jsonl",
            [record(split=split), record("audio_1.wav", split, text=" Second. ")],
        )
    return tmp_path


@pytest.mark.parametrize("split", ["train", "test"])
def test_competition_order_labels_and_raw_text(competition, split):
    frame = transcripts.load_competition_transcripts(split)
    assert frame["filename"].tolist() == ["audio_1.wav", "audio_0.wav"]
    assert frame["text"].tolist() == [" Second. ", record()["text"]]
    assert frame["duration_seconds"].tolist() == [1.25, 1.25]
    assert frame["label"].tolist() == ([4, 2] if split == "train" else [-1, -1])
    assert frame["split"].tolist() == [split, split]
    assert frame["language"].tolist() == ["en", "en"]
    assert frame["segments"].tolist() == [record()["segments"]] * 2


def test_wrong_split(competition):
    write_records(competition / "train.jsonl", [record(split="test")])
    with pytest.raises(ValueError, match="requested split 'train'"):
        transcripts.load_competition_transcripts("train")


@pytest.mark.parametrize("records", [[record()], []])
def test_missing_transcript(competition, records):
    write_records(competition / "train.jsonl", records)
    with pytest.raises(ValueError, match="missing transcripts.*audio_1.wav"):
        transcripts.load_competition_transcripts("train")


def test_extra_transcript(competition):
    write_records(
        competition / "train.jsonl",
        [record(), record("audio_1.wav"), record("extra.wav")],
    )
    with pytest.raises(ValueError, match="unexpected extra transcripts.*extra.wav"):
        transcripts.load_competition_transcripts("train")


def test_invalid_requested_split():
    with pytest.raises(ValueError, match="Expected split"):
        transcripts.load_competition_transcripts("validation")


def test_preserves_index_and_does_not_overwrite_labels(competition, monkeypatch):
    frame = pd.DataFrame(
        {"filename": ["audio_1.wav", "audio_0.wav"], "label": [4, 2]}, index=[9, 3]
    )
    monkeypatch.setattr(dataset, "load_train_dataframe", lambda: frame)
    write_records(
        competition / "train.jsonl",
        [record(label=999), record("audio_1.wav", label=999)],
    )
    result = transcripts.load_competition_transcripts("train")
    assert result.index.tolist() == [9, 3]
    assert result["label"].tolist() == [4, 2]
    assert frame.columns.tolist() == ["filename", "label"]
