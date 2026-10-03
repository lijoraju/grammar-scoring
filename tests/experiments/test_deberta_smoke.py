import sys
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from grammar_scoring.experiments import deberta_smoke as smoke


def test_subset_preserves_canonical_order_and_frozen_partitions():
    frame = pd.DataFrame(
        {
            "filename": [f"file_{i}" for i in range(100)],
            "fold": np.tile([0, 1, 2, 3, 4], 20),
            "label": np.linspace(0, 5, 100),
            "text": [" Raw! "] * 100,
        }
    )
    original = frame.copy(deep=True)
    training, validation = smoke.smoke_subset(frame)
    pd.testing.assert_frame_equal(training, frame.loc[frame.fold != 0].iloc[:32])
    pd.testing.assert_frame_equal(validation, frame.loc[frame.fold == 0].iloc[:16])
    pd.testing.assert_frame_equal(frame, original)
    assert set(training.filename).isdisjoint(validation.filename)
    with pytest.raises(ValueError, match="32 training"):
        smoke.smoke_subset(frame.iloc[:20])


def test_smoke_namespace_and_symlink_safety(tmp_path):
    assert smoke.smoke_root(tmp_path / "smoke/E005") == tmp_path / "smoke/E005"
    for path in (
        "models/E005",
        "experiments/E005",
        "oof",
        "other",
        "models/E005/smoke/E005",
        "experiments/E005/smoke/E005",
    ):
        with pytest.raises(ValueError):
            smoke.smoke_root(tmp_path / path)
    canonical = tmp_path / "models/E005"
    canonical.mkdir(parents=True)
    alias = tmp_path / "smoke/E005"
    alias.parent.mkdir()
    alias.symlink_to(canonical, target_is_directory=True)
    with pytest.raises(ValueError):
        smoke.smoke_root(alias)
    assert not (tmp_path / "other").exists()


@pytest.mark.parametrize("values", [(3, 2, 2), (4, 1, 1), (4, 2, 1), (4, 3, 3)])
def test_actual_counts_reject_missing_updates(values):
    counts = smoke.SmokeCounts(*values)
    with pytest.raises(RuntimeError):
        counts.validate()


def test_observed_scheduler_excludes_constructor_steps():
    actual = Mock()
    counts = smoke.SmokeCounts(minibatches=4, optimizer_steps=2)
    scheduler = smoke.ObservedScheduler(actual, counts)
    assert counts.scheduler_steps == 0
    scheduler.step()
    scheduler.step()
    assert actual.step.call_count == 2
    counts.validate()


def test_validation_metrics_and_round_trip():
    labels = np.linspace(0, 5, 16)
    predicted = labels + 0.25
    metrics = smoke.validation_metrics(labels, predicted)
    assert metrics["validation_loss"] == pytest.approx(0.0625)
    assert metrics["rmse"] == pytest.approx(0.25)
    smoke.check_round_trip(predicted, predicted + 1e-5)
    with pytest.raises(RuntimeError, match="round trip failed"):
        smoke.check_round_trip(predicted, predicted + 0.1)
    for invalid in (predicted[:-1], np.full(16, np.nan), predicted.reshape(16, 1)):
        with pytest.raises(RuntimeError):
            smoke.validation_metrics(labels, invalid)
        with pytest.raises(RuntimeError):
            smoke.check_round_trip(predicted, invalid)


def test_observed_loader_preserves_batches_and_counts_padding():
    torch = pytest.importorskip("torch")
    batches = [
        {
            "input_ids": torch.tensor([[1, 2, 3], [1, 2, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
        }
    ]
    counts = smoke.SmokeCounts()
    loader = smoke.ObservedLoader(batches, counts)
    assert len(loader) == 1
    assert list(loader)[0] is batches[0]
    assert asdict(counts) == {
        "minibatches": 1,
        "optimizer_steps": 0,
        "scheduler_steps": 0,
        "maximum_padded_length": 3,
        "padded_batches": 1,
    }
    batches[0]["attention_mask"] = torch.tensor([[1, 1, 0], [1, 1, 0]])
    with pytest.raises(RuntimeError, match="dynamically"):
        list(loader)


def test_cpu_rejected_before_inputs_download_or_outputs(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    load_inputs = Mock(side_effect=AssertionError("Must not load inputs"))
    monkeypatch.setattr(smoke, "load_inputs", load_inputs)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace())
    with pytest.raises(RuntimeError, match="CUDA is required"):
        smoke.run_smoke(tmp_path, tmp_path / "folds.csv", tmp_path / "smoke/E005")
    load_inputs.assert_not_called()
    assert not (tmp_path / "smoke").exists()


def test_cli_failure_is_explicit_and_nonzero(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["smoke", "--smoke-dir", str(tmp_path)])
    monkeypatch.setattr(
        smoke, "run_smoke", Mock(side_effect=RuntimeError("CUDA out of memory"))
    )
    with pytest.raises(SystemExit) as error:
        smoke.main()
    assert error.value.code == 1
    output = capsys.readouterr().out
    assert smoke.NOTICE in output
    assert "[FAIL] RuntimeError: CUDA out of memory" in output
    assert "batch size and settings are unchanged" in output


def test_gpu_memory_uses_cuda_apis_and_rejects_capacity_overflow(monkeypatch):
    torch = pytest.importorskip("torch")
    synchronize = Mock()
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 100)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 200)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 150)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(total_memory=1000),
    )
    assert smoke.gpu_memory("fake_cuda") == {
        "allocated": 100,
        "reserved": 200,
        "peak_allocated": 150,
        "total_capacity": 1000,
    }
    synchronize.assert_called_once_with("fake_cuda")
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 1001)
    with pytest.raises(RuntimeError, match="capacity"):
        smoke.gpu_memory("fake_cuda")


def test_forward_inspection_failure_removes_hooks(monkeypatch):
    torch = pytest.importorskip("torch")
    from contextlib import nullcontext

    from grammar_scoring.models.deberta_regressor import DebertaRegressor

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=2)

        def forward(self, **inputs):
            ids = inputs["input_ids"]
            return SimpleNamespace(
                last_hidden_state=ids.float().unsqueeze(-1).repeat(1, 1, 2)
            )

    model = DebertaRegressor(Backbone())
    monkeypatch.setattr(torch.amp, "autocast", lambda *args, **kwargs: nullcontext())
    batch = {
        "input_ids": torch.tensor([[1, 2, 0]]),
        "attention_mask": torch.tensor([[1, 1, 0]]),
    }
    with pytest.raises(RuntimeError, match="CUDA FP16"):
        smoke.inspect_forward(model, batch, torch.device("cpu"))
    assert not model.backbone._forward_hooks
    assert not model.head._forward_pre_hooks
