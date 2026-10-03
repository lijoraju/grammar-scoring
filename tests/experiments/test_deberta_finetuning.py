import json
import sys
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from grammar_scoring.experiments import deberta_finetuning as e005


@pytest.fixture
def frame():
    return pd.DataFrame(
        {
            "filename": [f"sample_{i}.wav" for i in range(769)],
            "label": np.linspace(0, 5, 769),
            "text": [f"  Raw text {i}!  " for i in range(769)],
            "fold": np.repeat(range(5), [154, 154, 154, 154, 153]),
        }
    )


def align(frame):
    return e005.align_inputs(
        frame[["filename", "label"]],
        frame[["filename", "text"]].assign(split="train").iloc[::-1],
        frame[["filename", "label", "fold"]].sample(frac=1, random_state=42),
    )


def test_filename_alignment_and_frozen_splits(frame):
    pd.testing.assert_frame_equal(align(frame), frame)
    for fold in range(5):
        training, validation = e005.split_fold(frame, fold)
        assert set(training.filename).isdisjoint(validation.filename)
        assert validation.fold.eq(fold).all()
        assert len(validation) == (153 if fold == 4 else 154)
        assert len(training) == (616 if fold == 4 else 615)
    changed = frame.copy()
    changed.loc[[0, 154], "fold"] = [1, 0]
    assert e005.experiment_identity(changed) != e005.experiment_identity(frame)


@pytest.mark.parametrize(
    "damage",
    ["duplicate", "missing", "empty", "test", "label", "nonfinite", "fold", "size"],
)
def test_invalid_inputs_rejected(frame, damage):
    if damage == "duplicate":
        frame.loc[1, "filename"] = frame.loc[0, "filename"]
    elif damage == "missing":
        frame = frame.iloc[:-1]
    elif damage == "empty":
        frame.loc[0, "text"] = " \t"
    elif damage == "test":
        with pytest.raises(ValueError, match="only train"):
            e005.align_inputs(frame, frame.assign(split="test"), frame)
        return
    elif damage == "label":
        frame.loc[0, "label"] = 6
    elif damage == "nonfinite":
        frame.loc[0, "label"] = np.inf
    elif damage == "fold":
        frame.loc[0, "fold"] = -1
    elif damage == "size":
        frame.loc[0, "fold"] = 4
    with pytest.raises(ValueError):
        align(frame)


def test_missing_transcript_and_mismatched_fold_label(frame):
    transcripts = frame.assign(split="train")
    with pytest.raises(ValueError, match="filenames"):
        e005.align_inputs(frame, transcripts.iloc[:-1], frame)
    folds = frame.copy()
    folds.loc[0, "label"] = 2
    with pytest.raises(ValueError, match="labels differ"):
        e005.align_inputs(frame, transcripts, folds)


def make_result(frame, fold, directory):
    training, validation = e005.split_fold(frame, fold)
    metrics = e005.regression_metrics(validation.label, validation.label + 0.1)
    history = [
        {
            "fold": fold,
            "epoch": epoch,
            "train_loss": 0.5,
            "validation_rmse": metrics["rmse"] + abs(epoch - 2) * 0.1,
            "validation_pearson": metrics["pearson_correlation"],
            "learning_rate": 1e-5,
        }
        for epoch in range(1, 6)
    ]
    directory.mkdir(parents=True)
    checkpoint = directory / "best.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    result = {
        "fold": fold,
        "identity": e005.experiment_identity(frame),
        "config": asdict(e005.CONFIG),
        "selected_epoch": 2,
        "validation_rmse": metrics["rmse"],
        "validation_pearson": metrics["pearson_correlation"],
        "n_train": len(training),
        "n_valid": len(validation),
        "history": history,
        "validation": e005._prediction_records(validation, validation.label + 0.1),
        "training": e005._prediction_records(training, training.label + 0.05),
        "checkpoint_sha256": e005._file_digest(checkpoint),
        "runtime": {"device": "cpu"},
        "model": {"name": e005.CONFIG.model_name},
    }
    e005._write_json(directory / "history.json", history)
    e005._write_json(directory / "result.json", result)
    return result


def test_best_epoch_minimum_and_first_tie():
    history = [
        {"epoch": i, "validation_rmse": loss}
        for i, loss in enumerate([3, 1, 2, 1, 4], 1)
    ]
    assert e005.select_best_epoch(history)["epoch"] == 2
    with pytest.raises(ValueError):
        e005.select_best_epoch([{"validation_rmse": np.nan}])


def test_resume_valid_incomplete_corrupt_and_mismatch(frame, tmp_path):
    directory = tmp_path / "fold_0"
    result = make_result(frame, 0, directory)
    identity = e005.experiment_identity(frame)
    assert e005.validate_completed_fold(directory, frame, 0, identity) == result
    with pytest.raises(ValueError, match="mismatch"):
        e005.validate_completed_fold(directory, frame, 0, "other")
    result["config"]["seed"] = 43
    e005._write_json(directory / "result.json", result)
    with pytest.raises(ValueError, match="mismatch"):
        e005.validate_completed_fold(directory, frame, 0, identity)
    result["config"]["seed"] = 42
    e005._write_json(directory / "result.json", result)
    (directory / "best.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        e005.validate_completed_fold(directory, frame, 0, identity)
    (directory / "result.json").unlink()
    with pytest.raises(ValueError, match="Incomplete"):
        e005.validate_completed_fold(directory, frame, 0, identity)


def test_resume_rejects_inconsistent_predictions_and_history(frame, tmp_path):
    result = make_result(frame, 0, tmp_path / "fold")
    result["validation"][0]["prediction"] += 1
    e005._write_json(tmp_path / "fold/result.json", result)
    with pytest.raises(ValueError, match="RMSE"):
        e005.validate_completed_fold(
            tmp_path / "fold", frame, 0, e005.experiment_identity(frame)
        )
    result["history"].pop()
    e005._write_json(tmp_path / "fold/result.json", result)
    with pytest.raises(ValueError, match="history"):
        e005.validate_completed_fold(
            tmp_path / "fold", frame, 0, e005.experiment_identity(frame)
        )


def test_oof_coverage_order_finite_labels_folds(frame, tmp_path):
    results = [make_result(frame, fold, tmp_path / str(fold)) for fold in range(5)]
    for result in results:
        result["validation"].reverse()
    oof = e005.assemble_oof(frame, results[::-1])
    assert list(oof.columns) == [
        "filename",
        "label",
        "fold",
        "prediction",
        "residual",
        "abs_error",
    ]
    pd.testing.assert_frame_equal(
        oof[["filename", "label", "fold"]], frame[["filename", "label", "fold"]]
    )
    assert len(oof) == 769 and oof.filename.is_unique
    results[0]["validation"][0]["prediction"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        e005.assemble_oof(frame, results)


@pytest.mark.parametrize("column,value", [("label", 99), ("fold", 4)])
def test_oof_preservation(frame, tmp_path, column, value):
    results = [make_result(frame, fold, tmp_path / str(fold)) for fold in range(5)]
    results[0]["validation"][0][column] = value
    with pytest.raises(ValueError, match="differs"):
        e005.assemble_oof(frame, results)
    with pytest.raises(ValueError):
        e005.assemble_oof(frame, results[:-1])


def test_accumulation_steps_weighted_loss_and_partial_gradient():
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, input_ids):
            return input_ids.float() * self.weight

    model = Model()
    # Three mini-batches, with an uneven final group and known pre-update MSE.
    batches = [
        {"input_ids": torch.ones(n), "labels": torch.full((n,), target)}
        for n, target in [(8, 1.0), (8, 2.0), (3, 4.0)]
    ]
    optimizer = torch.optim.SGD(model.parameters(), lr=0)
    original_step = optimizer.step
    gradients = []

    def record_step(*args, **kwargs):
        gradients.append(model.weight.grad.item())
        return original_step(*args, **kwargs)

    optimizer.step = record_step
    scheduler = Mock()
    loss = e005.train_epoch(
        model,
        batches,
        optimizer,
        scheduler,
        e005.create_scaler(torch.device("cpu")),
        torch.device("cpu"),
    )
    assert loss == pytest.approx((8 * 1 + 8 * 4 + 3 * 16) / 19)
    assert len(gradients) == e005.optimizer_step_count(3) == 2
    assert gradients[-1] != 0
    assert scheduler.step.call_count == 2
    assert model.weight.grad is None


def test_partial_accumulation_matches_full_batch_gradient(monkeypatch):
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.zero_()
    # Disable clipping only for this synthetic gradient-equivalence assertion.
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", lambda *args: None)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batches = [
        {"input": torch.ones(n, 1), "labels": torch.full((n,), target)}
        for n, target in [(2, 1.0), (1, 4.0), (1, 3.0)]
    ]

    class Wrapped(torch.nn.Module):
        def forward(self, input):
            return model(input).squeeze(-1)

        def parameters(self):
            return model.parameters()

    scheduler = Mock()
    e005.train_epoch(
        Wrapped(),
        batches,
        optimizer,
        scheduler,
        e005.create_scaler(torch.device("cpu")),
        torch.device("cpu"),
    )
    # First sample-weighted update is 0.4; final single-batch update is 0.52.
    assert model.weight.item() == pytest.approx(0.92)
    assert scheduler.step.call_count == 2


def test_five_epochs_fresh_factories_reload_best_and_resume(
    frame, tmp_path, monkeypatch
):
    torch = pytest.importorskip("torch")
    # Entire HF surface used by this test is offline and fake.
    fake_hf = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=Mock(side_effect=AssertionError)),
        get_linear_schedule_with_warmup=lambda *args, **kwargs: Mock(),
        __version__="offline-fake",
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_hf)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))
            self.backbone = SimpleNamespace(
                config=SimpleNamespace(
                    to_dict=lambda: {"hidden_size": 1}, _commit_hash="fake-original"
                )
            )

    models = []
    tokenizers = []

    def factory():
        model = Model()
        models.append(model)
        return model

    def tokenizer_factory():
        tokenizer = SimpleNamespace(init_kwargs={"model_max_length": 256})
        tokenizers.append(tokenizer)
        return tokenizer

    monkeypatch.setattr(e005, "_loader", lambda data, tok, shuffle: (data, 0))
    epochs = []

    def epoch(model, *args):
        epoch_number = len(epochs) % 5 + 1
        epochs.append(epoch_number)
        model.weight.data.fill_([3, 1, 2, 4, 5][epoch_number - 1])
        return float(epoch_number)

    monkeypatch.setattr(e005, "train_epoch", epoch)
    monkeypatch.setattr(
        e005,
        "predict",
        lambda model, data, device: data.label.to_numpy() + model.weight.item() * 0.1,
    )
    result = e005.run_experiment(
        frame,
        tmp_path,
        device="cpu",
        model_factory=factory,
        tokenizer_factory=tokenizer_factory,
    )
    assert epochs == list(range(1, 6)) * 5
    assert len(models) == len(tokenizers) == 5
    assert len({id(model) for model in models}) == 5
    assert all(model.weight.item() == 1 for model in models)
    assert all(row["selected_epoch"] == 2 for row in result["per_fold"])
    assert result["oof_metrics"]["rmse"] == pytest.approx(0.1)
    assert result["diagnostic_training_rmse"] == pytest.approx(0.1)
    history_path = tmp_path / "experiments/E005/training_history.json"
    assert len(json.loads(history_path.read_text())) == 25
    resumed = e005.run_experiment(
        frame,
        tmp_path,
        device="cpu",
        resume=True,
        model_factory=Mock(side_effect=AssertionError),
        tokenizer_factory=Mock(side_effect=AssertionError),
    )
    assert resumed == result
    summary_path = tmp_path / "experiments/E005/summary.json"
    saved = summary_path.read_bytes()
    partial = e005.run_experiment(frame, tmp_path, device="cpu", resume=True, fold=0)
    assert partial["complete"] is False
    assert summary_path.read_bytes() == saved


def test_raw_tokenization_dynamic_padding_and_length_diagnostic(frame, monkeypatch):
    torch = pytest.importorskip("torch")
    calls = []
    collator_calls = []

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            calls.append((texts, kwargs))
            lengths = [260, 3]
            if kwargs["truncation"]:
                lengths = [min(length, kwargs["max_length"]) for length in lengths]
            return {
                "input_ids": [[1] * length for length in lengths],
                "attention_mask": [[1] * length for length in lengths],
            }

    class Collator:
        def __init__(self, tokenizer, padding):
            collator_calls.append(padding)

        def __call__(self, records):
            maximum = max(len(row["input_ids"]) for row in records)
            return {
                "input_ids": torch.tensor(
                    [
                        row["input_ids"] + [0] * (maximum - len(row["input_ids"]))
                        for row in records
                    ]
                ),
                "labels": torch.tensor([row["labels"] for row in records]),
            }

    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(DataCollatorWithPadding=Collator)
    )
    loader, exceeded = e005._loader(frame.iloc[:2], Tokenizer(), shuffle=False)
    assert exceeded == 1
    assert calls[0] == (
        frame.text.iloc[:2].tolist(),
        {"truncation": False, "padding": False},
    )
    assert calls[1][1] == {"max_length": 256, "truncation": True, "padding": False}
    assert collator_calls == [True]
    batch = next(iter(loader))
    assert batch["input_ids"].shape == (2, 256)
    assert batch["input_ids"][1].count_nonzero() == 3


def test_seed_reapplication():
    torch = pytest.importorskip("torch")
    import random

    e005.seed_everything()
    first = (random.random(), np.random.random(), torch.rand(1).item())
    e005.seed_everything()
    assert first == (random.random(), np.random.random(), torch.rand(1).item())


def test_predict_raw_unclipped_and_nonfinite_rejected():
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def forward(self, input_ids):
            return input_ids.float()

    loader = [
        {"input_ids": torch.tensor([-1.0, 7.0]), "labels": torch.tensor([0.0, 5.0])}
    ]
    np.testing.assert_array_equal(
        e005.predict(Model(), loader, torch.device("cpu")), [-1.0, 7.0]
    )
    loader[0]["input_ids"][0] = float("nan")
    with pytest.raises(RuntimeError, match="Nonfinite"):
        e005.predict(Model(), loader, torch.device("cpu"))


@pytest.mark.parametrize("skipped", [False, True])
def test_amp_update_order_and_dtype_observer(monkeypatch, skipped):
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0)
    events = []
    scale = [8.0]

    class ScaledLoss:
        def __init__(self, loss):
            self.loss = loss

        def backward(self):
            events.append("backward")
            self.loss.backward()

    def unscale(actual_optimizer):
        assert actual_optimizer is optimizer
        assert model.weight.grad is not None
        events.append("unscale")

    def update():
        events.append("update")
        if skipped:
            scale[0] /= 2

    scaler = Mock()
    scaler.scale.side_effect = ScaledLoss
    scaler.unscale_.side_effect = unscale
    scaler.get_scale.side_effect = lambda: scale[0]
    scaler.step.side_effect = lambda opt: events.append("step")
    scaler.update.side_effect = update
    scheduler = Mock()
    scheduler.step.side_effect = lambda: events.append("scheduler")
    monkeypatch.setattr(
        torch.nn.utils, "clip_grad_norm_", lambda *args: events.append("clip")
    )

    def observe(stage, observed_model, loss):
        assert observed_model is model
        assert loss.dtype == torch.float32
        if stage == "before_unscale":
            assert model.weight.grad.dtype == torch.float32
        events.append(stage)

    e005.train_epoch(
        model,
        [{"input": torch.ones(1, 1), "labels": torch.ones(1, 1)}],
        optimizer,
        scheduler,
        scaler,
        torch.device("cpu"),
        dtype_observer=observe,
    )
    expected = [
        "before_backward",
        "backward",
        "before_unscale",
        "unscale",
        "clip",
        "step",
        "update",
    ]
    if not skipped:
        expected.append("scheduler")
    assert events == expected
    assert model.weight.grad is None
