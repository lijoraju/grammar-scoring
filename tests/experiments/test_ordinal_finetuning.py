import json
import sys
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from grammar_scoring.experiments import deberta_ordinal as e007
from grammar_scoring.experiments.deberta_finetuning import CONFIG as E005_CONFIG
from grammar_scoring.experiments.ordinal_diagnostics import (
    group_diagnostics,
    score_diagnostics,
)
from grammar_scoring.models.ordinal import SCORE_VALUES


@pytest.fixture
def frame():
    return pd.DataFrame(
        {
            "filename": [f"sample_{i}.wav" for i in range(769)],
            "label": np.resize(SCORE_VALUES, 769),
            "text": [f" Raw transcript {i}! " for i in range(769)],
            "fold": np.repeat(range(5), [154, 154, 154, 154, 153]),
        }
    )


def test_protocol_alignment_isolation_identity_and_ties(frame):
    old, new = asdict(E005_CONFIG), asdict(e007.CONFIG)
    for key in old.keys() - {"experiment", "head", "loss"}:
        assert old[key] == new[key]
    aligned = e007.align_inputs(
        frame, frame.assign(split="train").iloc[::-1], frame.iloc[::-1]
    )
    pd.testing.assert_frame_equal(aligned, frame)
    for fold in range(5):
        training, validation = e007.split_fold(frame, fold)
        assert set(training.filename).isdisjoint(validation.filename)
        assert validation.fold.eq(fold).all()
    assert e007.experiment_identity(frame) == e007.experiment_identity(frame.copy())
    changed = frame.copy()
    changed.loc[0, "text"] += "!"
    assert e007.experiment_identity(changed) != e007.experiment_identity(frame)
    history = [
        {"epoch": i, "validation_rmse": x} for i, x in enumerate([3, 1, 2, 1, 4], 1)
    ]
    assert e007.select_best_epoch(history)["epoch"] == 2
    changed.loc[0, "label"] = 0.5
    with pytest.raises(ValueError):
        e007.align_inputs(changed, changed.assign(split="train"), changed)


def test_score_and_group_diagnostics():
    frame = pd.DataFrame(
        {
            "label": [0.0, 0.0, 2.0, 3.0, 4.0, 5.0],
            "prediction": [1.0, 3.0, 2.0, 2.0, 4.0, 4.0],
        }
    )
    by_score = score_diagnostics(frame).set_index("label")
    low = by_score.loc[0.0]
    assert low["count"] == 2
    assert low.prediction_mean == 2
    assert low.prediction_std == 1
    assert low.prediction_min == 1 and low.prediction_max == 3
    assert low.mean_signed_error == low.mae == 2
    assert low.rmse == pytest.approx(np.sqrt(5))
    groups = group_diagnostics(frame).set_index("group")
    assert groups.loc["LOW", "n"] == 3
    assert groups.loc["LOW", "rmse"] == pytest.approx(np.sqrt(10 / 3))
    assert groups.loc["MID", "mean_signed_error"] == -0.5
    assert groups.loc["HIGH", "mae"] == 1
    pd.testing.assert_frame_equal(score_diagnostics(frame), score_diagnostics(frame))


def test_all_epochs_checkpoint_oof_resume_and_reports(frame, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(
                from_pretrained=Mock(side_effect=AssertionError)
            ),
            get_linear_schedule_with_warmup=lambda *a, **kw: Mock(),
            __version__="offline",
        ),
    )

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))
            self.backbone = SimpleNamespace(
                config=SimpleNamespace(
                    to_dict=lambda: {"hidden_size": 1}, _commit_hash="fake"
                )
            )

    models, epochs = [], []

    def factory():
        model = Model()
        models.append(model)
        return model

    def epoch(model, *args):
        number = len(epochs) % 5 + 1
        epochs.append(number)
        model.weight.data.fill_([3, 1, 2, 4, 5][number - 1])
        return 0.5

    # A constant valid categorical distribution; second epoch wins, then reloads.
    def cumulative(model, data, device):
        value = 0.8 + model.weight.item() * 0.02
        return np.full((len(data), 9), value)

    monkeypatch.setattr(e007, "_loader", lambda data, tok, shuffle: (data, 0))
    monkeypatch.setattr(e007, "train_epoch", epoch)
    monkeypatch.setattr(e007, "predict_cumulative", cumulative)
    monkeypatch.setattr(
        e007,
        "predict",
        lambda model, data, device: cumulative(model, data, device)[:, 0] * 5,
    )
    baseline_path = tmp_path / "oof/E005_deberta_finetuned.csv"
    baseline_path.parent.mkdir(parents=True)
    frame.assign(prediction=4.1).drop(columns="text").to_csv(baseline_path, index=False)
    summary = e007.run_experiment(
        frame,
        tmp_path,
        device="cpu",
        model_factory=factory,
        tokenizer_factory=lambda: SimpleNamespace(init_kwargs={}),
    )
    assert epochs == list(range(1, 6)) * 5
    assert len(models) == 5
    assert all(row["selected_epoch"] == 2 for row in summary["per_fold"])
    oof = pd.read_csv(tmp_path / "oof/E007_deberta_ordinal.csv")
    assert list(oof.columns) == [
        "filename",
        "label",
        "fold",
        "prediction",
        "residual",
        "abs_error",
    ]
    assert oof.filename.tolist() == frame.filename.tolist()
    pd.testing.assert_frame_equal(oof[["label", "fold"]], frame[["label", "fold"]])
    resumed = e007.run_experiment(
        frame,
        tmp_path,
        device="cpu",
        resume=True,
        model_factory=Mock(side_effect=AssertionError),
    )
    assert resumed == summary
    diagnostics = json.loads(
        (tmp_path / "experiments/E007/ordinal_diagnostics.json").read_text()
    )
    assert diagnostics["monotonic_every_row"]
    path = tmp_path / "models/E007/fold_0/result.json"
    damaged = json.loads(path.read_text())
    damaged["cumulative_validation"][0][0] = 0
    path.write_text(json.dumps(damaged))
    with pytest.raises(ValueError):
        e007.validate_completed_fold(
            path.parent, frame, 0, e007.experiment_identity(frame)
        )


def test_training_bce_accumulation_and_determinism():
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(9))

        def forward(self, input_ids):
            return self.weight.expand(len(input_ids), -1)

    e007.seed_everything()
    first = torch.rand(3)
    e007.seed_everything()
    torch.testing.assert_close(first, torch.rand(3))
    model = Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    scheduler = Mock()
    loss = e007.train_epoch(
        model,
        [
            {"input_ids": torch.ones(n), "labels": torch.full((n,), value)}
            for n, value in [(8, 0.0), (8, 2.0), (3, 5.0)]
        ],
        optimizer,
        scheduler,
        e007.create_scaler(torch.device("cpu")),
        torch.device("cpu"),
    )
    assert loss == pytest.approx(np.log(2))
    assert scheduler.step.call_count == 2
    assert model.weight.grad is None


def test_real_cpu_prediction_helpers_preserve_order():
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def forward(self, input_ids):
            return input_ids.float()[:, None].expand(-1, 9)

    loader = [
        {
            "input_ids": torch.tensor([-100.0, 0.0, 100.0]),
            "labels": torch.tensor([0.0, 3.0, 5.0]),
        }
    ]
    device = torch.device("cpu")
    q = e007.predict_cumulative(Model(), loader, device)
    assert q.shape == (3, 9)
    np.testing.assert_allclose(e007.predict(Model(), loader, device), [0, 2.5, 5])
    loader[0]["input_ids"][0] = float("nan")
    with pytest.raises(ValueError):
        e007.predict(Model(), loader, device)
