"""Tests for canonical E005 DeBERTa inference."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from grammar_scoring.inference.deberta import (
    create_inference_loader,
    discover_checkpoints,
    load_checkpoint,
    predict_e005_ensemble,
    predict_loader,
    validate_test_frame,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "filename": ["b.wav", "a.wav", "c.wav"],
            "text": ["second text", "first text", "third text"],
        }
    )


class FakeModel(torch.nn.Module):
    """Small deterministic model for offline inference tests."""

    def __init__(self, value: float = 0.0) -> None:
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(value))

    def forward(self, **inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs["input_ids"].shape[0]
        return self.value.expand(batch_size)


class RecordingTokenizer:
    """Record tokenization calls for E005 inference contract tests."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        texts: list[str],
        **kwargs: object,
    ) -> dict[str, list[list[int]]]:
        self.calls.append({"texts": texts, **kwargs})

        if kwargs.get("truncation") is False:
            return {
                "input_ids": [
                    list(range(300)),
                    [1, 2],
                    [1, 2, 3],
                ]
            }

        return {
            "input_ids": [
                list(range(256)),
                [1, 2],
                [1, 2, 3],
            ],
            "attention_mask": [
                [1] * 256,
                [1, 1],
                [1, 1, 1],
            ],
        }


def test_validate_test_frame_preserves_order() -> None:
    frame = _frame()

    result = validate_test_frame(frame)

    assert result["filename"].tolist() == ["b.wav", "a.wav", "c.wav"]
    assert result["text"].tolist() == [
        "second text",
        "first text",
        "third text",
    ]


@pytest.mark.parametrize("column", ["filename", "text"])
def test_validate_test_frame_requires_columns(column: str) -> None:
    frame = _frame().drop(columns=column)

    with pytest.raises(ValueError, match="missing required columns"):
        validate_test_frame(frame)


def test_validate_test_frame_rejects_duplicate_filenames() -> None:
    frame = _frame()
    frame.loc[1, "filename"] = "b.wav"

    with pytest.raises(ValueError, match="unique"):
        validate_test_frame(frame)


@pytest.mark.parametrize("value", ["", "   ", None])
def test_validate_test_frame_rejects_invalid_filenames(value: object) -> None:
    frame = _frame()
    frame.loc[0, "filename"] = value

    with pytest.raises(ValueError, match="nonempty strings"):
        validate_test_frame(frame)


@pytest.mark.parametrize("value", ["", "   ", None])
def test_validate_test_frame_rejects_invalid_text(value: object) -> None:
    frame = _frame()
    frame.loc[0, "text"] = value

    with pytest.raises(ValueError, match="nonempty strings"):
        validate_test_frame(frame)


def test_discover_checkpoints_returns_fold_order(tmp_path: Path) -> None:
    expected = []

    for fold in reversed(range(5)):
        directory = tmp_path / f"fold_{fold}"
        directory.mkdir()
        checkpoint = directory / "best.pt"
        checkpoint.touch()
        expected.append(checkpoint)

    result = discover_checkpoints(tmp_path)

    assert result == [tmp_path / f"fold_{fold}" / "best.pt" for fold in range(5)]


def test_discover_checkpoints_rejects_missing_fold(tmp_path: Path) -> None:
    for fold in range(4):
        directory = tmp_path / f"fold_{fold}"
        directory.mkdir()
        (directory / "best.pt").touch()

    with pytest.raises(FileNotFoundError, match="fold_4"):
        discover_checkpoints(tmp_path)


def test_discover_checkpoints_rejects_unexpected_fold(tmp_path: Path) -> None:
    for fold in range(6):
        directory = tmp_path / f"fold_{fold}"
        directory.mkdir()
        (directory / "best.pt").touch()

    with pytest.raises(ValueError, match="fold_5"):
        discover_checkpoints(tmp_path)


def test_load_checkpoint_restores_state_dict(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pt"
    source = FakeModel(3.25)
    torch.save(source.state_dict(), checkpoint)

    target = FakeModel(-1.0)
    result = load_checkpoint(target, checkpoint, torch.device("cpu"))

    assert result is target
    assert target.value.item() == pytest.approx(3.25)


def test_predict_loader_preserves_raw_values() -> None:
    model = FakeModel(-0.75)
    loader = [
        {
            "input_ids": torch.ones((2, 3), dtype=torch.long),
            "attention_mask": torch.ones((2, 3), dtype=torch.long),
        }
    ]

    result = predict_loader(model, loader, torch.device("cpu"))

    np.testing.assert_allclose(result, [-0.75, -0.75])
    assert result.dtype == np.float64


def test_predict_loader_rejects_nonfinite_predictions() -> None:
    model = FakeModel(float("nan"))
    loader = [
        {
            "input_ids": torch.ones((1, 2), dtype=torch.long),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        }
    ]

    with pytest.raises(RuntimeError, match="Nonfinite"):
        predict_loader(model, loader, torch.device("cpu"))


def test_ensemble_averages_five_raw_fold_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame()

    for fold in range(5):
        directory = tmp_path / f"fold_{fold}"
        directory.mkdir()
        torch.save(
            FakeModel(float(fold)).state_dict(),
            directory / "best.pt",
        )

    fake_loader = [
        {
            "input_ids": torch.ones((3, 2), dtype=torch.long),
            "attention_mask": torch.ones((3, 2), dtype=torch.long),
        }
    ]

    monkeypatch.setattr(
        "grammar_scoring.inference.deberta.create_inference_loader",
        lambda frame, tokenizer: (fake_loader, 0),
    )

    result = predict_e005_ensemble(
        frame,
        tmp_path,
        device="cpu",
        model_factory=FakeModel,
        tokenizer_factory=lambda: SimpleNamespace(),
    )

    assert result.columns.tolist() == ["filename", "label"]
    assert result["filename"].tolist() == frame["filename"].tolist()
    np.testing.assert_allclose(result["label"], [2.0, 2.0, 2.0])


def test_ensemble_does_not_clip_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame()

    for fold in range(5):
        directory = tmp_path / f"fold_{fold}"
        directory.mkdir()
        torch.save(
            FakeModel(6.25).state_dict(),
            directory / "best.pt",
        )

    fake_loader = [
        {
            "input_ids": torch.ones((3, 2), dtype=torch.long),
            "attention_mask": torch.ones((3, 2), dtype=torch.long),
        }
    ]

    monkeypatch.setattr(
        "grammar_scoring.inference.deberta.create_inference_loader",
        lambda frame, tokenizer: (fake_loader, 0),
    )

    result = predict_e005_ensemble(
        frame,
        tmp_path,
        device="cpu",
        model_factory=FakeModel,
        tokenizer_factory=lambda: SimpleNamespace(),
    )

    np.testing.assert_allclose(result["label"], [6.25, 6.25, 6.25])


def test_create_inference_loader_uses_e005_tokenization_contract() -> None:
    tokenizer = RecordingTokenizer()

    def collator_factory(tokenizer: object, *, padding: bool):
        assert padding is True

        def collate(
            records: list[dict[str, list[int]]],
        ) -> dict[str, torch.Tensor]:
            maximum = max(len(record["input_ids"]) for record in records)

            input_ids = []
            attention_mask = []

            for record in records:
                length = len(record["input_ids"])
                pad = maximum - length
                input_ids.append(record["input_ids"] + [0] * pad)
                attention_mask.append(record["attention_mask"] + [0] * pad)

            return {
                "input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor(attention_mask),
            }

        return collate

    frame = _frame()

    loader, exceeded = create_inference_loader(
        frame,
        tokenizer,
        collator_factory=collator_factory,
    )

    assert exceeded == 1

    assert tokenizer.calls[0] == {
        "texts": frame["text"].tolist(),
        "truncation": False,
        "padding": False,
    }

    assert tokenizer.calls[1] == {
        "texts": frame["text"].tolist(),
        "max_length": 256,
        "truncation": True,
        "padding": False,
    }

    batch = next(iter(loader))

    assert batch["input_ids"].shape == (3, 256)
    assert batch["attention_mask"].shape == (3, 256)
