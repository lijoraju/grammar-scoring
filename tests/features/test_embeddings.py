from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from grammar_scoring.features.embeddings import (
    EmbeddingResult,
    FrozenEmbedder,
    load_embeddings,
    masked_mean_pool,
    save_embeddings,
)


def frame():
    return pd.DataFrame(
        {
            "split": ["train", "test"],
            "filename": ["a", "a"],
            "text": ["  raw TEXT ", "other"],
            "label": [1, 9],
        }
    )


class MiniLM:
    def eval(self):
        self.evaluated = True

    def requires_grad_(self, value):
        assert value is False

    def encode(self, texts, **kwargs):
        self.texts, self.kwargs = texts, kwargs
        return np.vstack([np.full(384, i) for i in range(len(texts))])


class Tensor:
    """Small NumPy tensor adapter for offline torch-free wrapper tests."""

    def __init__(self, values):
        self.values = np.asarray(values)
        self.device = "cpu"
        self.dtype = self.values.dtype

    def unsqueeze(self, axis):
        return Tensor(np.expand_dims(self.values, axis))

    def to(self, *args, **kwargs):
        return Tensor(self.values.astype(kwargs.get("dtype", self.dtype)))

    def sum(self, dim):
        return Tensor(self.values.sum(axis=dim))

    def clamp(self, min):
        return Tensor(np.maximum(self.values, min))

    def __mul__(self, other):
        return Tensor(self.values * other.values)

    def __truediv__(self, other):
        return Tensor(self.values / other.values)

    def float(self):
        return Tensor(self.values.astype(np.float32))

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class Deberta(MiniLM):
    def to(self, device):
        self.device = device

    def __call__(self, **kwargs):
        n = len(kwargs["attention_mask"].values)
        hidden = np.tile(np.array([2, 4, 999])[None, :, None], (n, 1, 768))
        return SimpleNamespace(last_hidden_state=Tensor(hidden))


def test_minilm_order_and_configuration():
    model = MiniLM()
    embedder = FrozenEmbedder("minilm", batch_size=2, model=model)
    result = embedder.encode(frame())
    assert result.embeddings.shape == (2, 384)
    assert result.embeddings.dtype == np.float32
    assert result.split.tolist() == ["train", "test"]
    assert result.filenames.tolist() == ["a", "a"]
    assert result.embeddings[:, 0].tolist() == [0, 1]
    assert model.texts == frame().text.tolist()
    assert model.max_seq_length == 256
    assert model.kwargs["normalize_embeddings"] is False
    assert model.evaluated
    assert embedder.encode(frame().iloc[:0]).embeddings.shape == (0, 384)
    assert embedder.metadata()["max_length"] == 256


def test_deberta_mask_and_configuration():
    calls = []

    def tokenizer(texts, **kwargs):
        calls.append((texts, kwargs))
        return {"attention_mask": Tensor([[1, 1, 0]] * len(texts))}

    embedder = FrozenEmbedder(
        "deberta",
        model=Deberta(),
        tokenizer=tokenizer,
        batch_size=1,
        torch_module=SimpleNamespace(inference_mode=nullcontext),
    )
    result = embedder.encode(frame())
    assert result.embeddings.shape == (2, 768)
    np.testing.assert_array_equal(result.embeddings, 3)
    assert [call[0][0] for call in calls] == frame().text.tolist()
    assert calls[0][1] == {
        "padding": True,
        "truncation": True,
        "max_length": 512,
        "add_special_tokens": True,
        "return_tensors": "pt",
    }


def test_pooling_zero_mask():
    result = masked_mean_pool(Tensor([[[2], [999]]]), Tensor([[0, 0]]))
    np.testing.assert_array_equal(result.numpy(), [[0]])


@pytest.mark.parametrize(
    "matrix",
    [
        np.zeros((2, 383), dtype=np.float32),
        np.zeros(384, dtype=np.float32),
        np.zeros((1, 384), dtype=np.float32),
        np.full((2, 384), np.nan, dtype=np.float32),
        np.full((2, 384), np.inf, dtype=np.float32),
    ],
)
def test_bad_encoder_output(matrix):
    model = MiniLM()
    model.encode = lambda *args, **kwargs: matrix
    with pytest.raises(ValueError):
        FrozenEmbedder("minilm", model=model).encode(frame())


def test_duplicate_identity():
    data = frame()
    data["split"] = "train"
    with pytest.raises(ValueError, match="Duplicate"):
        FrozenEmbedder("minilm", model=MiniLM()).encode(data)


def test_npz_round_trip_and_overwrite(tmp_path):
    result = FrozenEmbedder("minilm", model=MiniLM()).encode(frame())
    path = tmp_path / "embeddings.npz"
    save_embeddings(result, path)
    loaded = load_embeddings(path, 384)
    np.testing.assert_array_equal(loaded.embeddings, result.embeddings)
    np.testing.assert_array_equal(loaded.split, result.split)
    np.testing.assert_array_equal(loaded.filenames, result.filenames)
    with np.load(path, allow_pickle=False) as artifact:
        assert set(artifact.files) == {"split", "filenames", "embeddings"}
        assert all(artifact[key].dtype.kind != "O" for key in artifact.files)
    with pytest.raises(FileExistsError):
        save_embeddings(result, path)
    save_embeddings(result, path, overwrite=True)
    with pytest.raises(ValueError):
        load_embeddings(path, 768)


@pytest.mark.parametrize(
    "field,value",
    [
        ("split", np.array(["train", "test"], dtype=object)),
        ("filenames", np.array(["", "a"])),
        ("embeddings", np.zeros((2, 384), dtype=np.float64)),
        ("split", np.array(["train"])),
    ],
)
def test_invalid_npz(tmp_path, field, value):
    fields = {
        "split": np.array(["train", "test"]),
        "filenames": np.array(["a", "a"]),
        "embeddings": np.zeros((2, 384), dtype=np.float32),
    }
    fields[field] = value
    path = tmp_path / "bad.npz"
    np.savez(path, **fields)
    with pytest.raises(ValueError):
        load_embeddings(path, 384)


def test_result_duplicate_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        EmbeddingResult(
            np.array(["train", "train"]),
            np.array(["a", "a"]),
            np.zeros((2, 384), dtype=np.float32),
            384,
        )


def test_deberta_wrong_dimension():
    class BadModel(Deberta):
        def __call__(self, **kwargs):
            return SimpleNamespace(last_hidden_state=Tensor(np.zeros((1, 1, 767))))

    with pytest.raises(ValueError, match="shape/dimension"):
        FrozenEmbedder(
            "deberta",
            model=BadModel(),
            tokenizer=lambda texts, **kwargs: {"attention_mask": Tensor([[1]])},
            torch_module=SimpleNamespace(inference_mode=nullcontext),
        ).encode(frame().iloc[:1])


@pytest.mark.parametrize(
    "column,value", [("text", None), ("split", " "), ("filename", 123)]
)
def test_bad_transcript_values(column, value):
    data = frame()
    data[column] = data[column].astype(object)
    data.loc[0, column] = value
    with pytest.raises(ValueError):
        FrozenEmbedder("minilm", model=MiniLM()).encode(data)


def test_load_explicitly_disables_pickle(tmp_path, monkeypatch):
    path = tmp_path / "embeddings.npz"
    save_embeddings(FrozenEmbedder("minilm", model=MiniLM()).encode(frame()), path)
    original = np.load
    calls = []

    def load(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(np, "load", load)
    load_embeddings(path, 384)
    assert calls == [{"allow_pickle": False}]
