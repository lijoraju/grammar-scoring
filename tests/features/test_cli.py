import json

import numpy as np
import pytest

from grammar_scoring.features import cli
from grammar_scoring.features.embeddings import MODELS, EmbeddingResult, load_embeddings


def transcripts(directory):
    for split in ("train", "test"):
        record = {
            "split": split,
            "filename": "same.wav",
            "text": " raw ",
            "language": "en",
            "duration_seconds": 1,
            "segments": [],
        }
        (directory / f"{split}.jsonl").write_text(json.dumps(record) + "\n")


class Embedder:
    instances = []

    def __init__(self, variant, device, batch_size):
        self.variant, self.device, self.batch_size = variant, device, batch_size
        self.calls = []
        self.instances.append(self)

    def encode(self, frame):
        assert "label" not in frame
        self.calls.append(frame.split.tolist())
        return EmbeddingResult(
            frame.split.to_numpy(dtype=str),
            frame.filename.to_numpy(dtype=str),
            np.zeros((len(frame), MODELS[self.variant][1]), dtype=np.float32),
            MODELS[self.variant][1],
        )

    def metadata(self):
        return {
            "model_name": MODELS[self.variant][0],
            "device": self.device,
            "batch_size": self.batch_size,
        }


def arguments(directory, output, split="all"):
    return [
        "--model",
        "minilm",
        "--split",
        split,
        "--transcript-dir",
        str(directory),
        "--output-dir",
        str(output),
    ]


def test_all_reuses_model_separate_identities_metadata(tmp_path, monkeypatch):
    transcripts(tmp_path)
    output = tmp_path / "output"
    monkeypatch.setattr(cli, "FrozenEmbedder", Embedder)
    Embedder.instances.clear()
    assert cli.generate_main(arguments(tmp_path, output)) == 0
    assert len(Embedder.instances) == 1
    assert Embedder.instances[0].calls == [["train"], ["test"]]
    for split in ("train", "test"):
        result = load_embeddings(output / f"minilm_{split}.npz", 384)
        assert result.split.tolist() == [split]
        assert result.filenames.tolist() == ["same.wav"]
    metadata = json.loads((output / "minilm.meta.json").read_text())
    assert set(metadata["artifacts"]) == {"train", "test"}
    assert len(metadata["artifacts"]["train"]["transcript_sha256"]) == 64
    with pytest.raises(FileExistsError):
        cli.generate_main(arguments(tmp_path, output))
    assert len(Embedder.instances) == 1  # Fail before loading expensive models.
    assert cli.generate_main(arguments(tmp_path, output) + ["--overwrite"]) == 0


def test_metadata_add_split_and_config_conflict(tmp_path, monkeypatch):
    transcripts(tmp_path)
    output = tmp_path / "output"
    monkeypatch.setattr(cli, "FrozenEmbedder", Embedder)
    cli.generate_main(arguments(tmp_path, output, "train"))
    cli.generate_main(arguments(tmp_path, output, "test"))
    path = output / "minilm.meta.json"
    metadata = json.loads(path.read_text())
    assert set(metadata["artifacts"]) == {"train", "test"}
    with pytest.raises(ValueError, match="Shared metadata"):
        cli.generate_main(
            arguments(tmp_path, output, "train") + ["--batch-size", "16", "--overwrite"]
        )
    assert json.loads(path.read_text()) == metadata
    cli.generate_main(
        arguments(tmp_path, output) + ["--batch-size", "16", "--overwrite"]
    )
    assert json.loads(path.read_text())["batch_size"] == 16


def test_audit_all_models_and_splits(tmp_path, monkeypatch, capsys):
    transcripts(tmp_path)
    loaded = []

    class Tokenizer:
        model_max_length = 999

        def __call__(self, text, **kwargs):
            assert text == " raw "
            assert kwargs["truncation"] is False
            return {"input_ids": [1, 2, 3]}

    def tokenizer(variant):
        loaded.append(variant)
        return Tokenizer()

    monkeypatch.setattr(cli, "load_audit_tokenizer", tokenizer)
    assert (
        cli.audit_main(
            ["--model", "all", "--split", "all", "--transcript-dir", str(tmp_path)]
        )
        == 0
    )
    assert loaded == ["minilm", "deberta"]
    output = capsys.readouterr().out
    assert output.count("configured limit=256") == 2
    assert output.count("configured limit=512") == 2
    assert "Limit discrepancy" in output


def test_wrong_artifact_split(tmp_path, monkeypatch):
    transcripts(tmp_path)
    (tmp_path / "train.jsonl").write_text((tmp_path / "test.jsonl").read_text())
    monkeypatch.setattr(cli, "FrozenEmbedder", Embedder)
    with pytest.raises(ValueError, match="unexpected split"):
        cli.generate_main(arguments(tmp_path, tmp_path / "out"))
