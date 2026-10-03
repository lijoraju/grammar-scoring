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
    counts = smoke.SmokeCounts(
        minibatches=values[0],
        optimizer_steps=values[1],
        scheduler_steps=values[2],
        optimizer_update_attempts=2,
    )
    with pytest.raises(RuntimeError):
        counts.validate()


def test_observed_scheduler_excludes_constructor_steps():
    actual = Mock()
    counts = smoke.SmokeCounts(
        minibatches=4, optimizer_steps=2, optimizer_update_attempts=2
    )
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
        "optimizer_update_attempts": 0,
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


def test_dtype_diagnostics_before_backward_and_before_unscale():
    torch = pytest.importorskip("torch")
    from grammar_scoring.models.deberta_regressor import DebertaRegressor

    backbone = torch.nn.Linear(2, 2)
    backbone.config = SimpleNamespace(hidden_size=2)
    model = DebertaRegressor(backbone)
    loss = model.head(model.backbone(torch.ones(1, 2))).square().mean()
    before = smoke.dtype_diagnostics("before_backward", model, loss)
    assert before["trainable_parameter_dtypes"] == ["torch.float32"]
    for key in (
        "head_weight_dtype",
        "head_bias_dtype",
        "backbone_parameter_dtype",
        "loss_dtype",
    ):
        assert before[key] == "torch.float32"
    assert (
        smoke.dtype_diagnostics("before_unscale", model, loss)["head_gradient_dtype"]
        is None
    )
    loss.backward()
    after = smoke.dtype_diagnostics("before_unscale", model, loss)
    assert after["gradient_dtypes"] == ["torch.float32"]
    assert after["head_gradient_dtype"] == "torch.float32"
    assert after["backbone_gradient_dtype"] == "torch.float32"


def test_gradient_diagnostics_identify_nonfinite_components():
    torch = pytest.importorskip("torch")
    model = torch.nn.Module()
    model.backbone = torch.nn.Linear(2, 1)
    model.head = torch.nn.Linear(1, 1)
    model.backbone.weight.grad = torch.tensor([[float("inf"), -float("inf")]])
    model.head.bias.grad = torch.tensor([float("nan")])
    before = model.backbone.weight.grad.clone()
    report = smoke.gradient_diagnostics(model)
    assert report["all_finite"] is False
    assert report["positive_inf"] == ["backbone.weight"]
    assert report["negative_inf"] == ["backbone.weight"]
    assert report["nan"] == ["head.bias"]
    assert torch.equal(before, model.backbone.weight.grad)


@pytest.mark.parametrize("scale_factor", [0.5, 1.0, 2.0])
def test_update_diagnostics_capture_attempt_without_changing_training(
    scale_factor, capsys
):
    torch = pytest.importorskip("torch")
    from grammar_scoring.experiments import deberta_finetuning as e005

    model = torch.nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(0.5)
        model.bias.fill_(0.5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = Mock()

    class Scaler:
        scale_value = 8.0

        def scale(self, loss):
            return loss * self.scale_value

        def get_scale(self):
            return self.scale_value

        def unscale_(self, opt):
            for group in opt.param_groups:
                for parameter in group["params"]:
                    parameter.grad.div_(self.scale_value)

        def step(self, opt):
            if scale_factor >= 1:
                return opt.step()
            return None

        def update(self):
            self.scale_value *= scale_factor

    observer = smoke.SmokeUpdateDiagnostics()
    e005.train_epoch(
        model,
        [{"input": torch.ones(1, 1), "labels": torch.zeros(1, 1)}],
        optimizer,
        scheduler,
        Scaler(),
        torch.device("cpu"),
        update_observer=observer,
    )
    report = observer.reports[0]
    assert report["update_index"] == 1
    assert report["microbatches"][0]["raw_loss"] == 1.0
    assert report["microbatches"][0]["scale_before_backward"] == 8.0
    assert report["before_unscale"]["maximum_absolute_gradient"] == 16.0
    assert report["after_unscale"]["maximum_absolute_gradient"] == 2.0
    assert report["after_unscale"]["all_finite"] is True
    assert report["unclipped_gradient_norm"] == pytest.approx(8**0.5)
    assert report["gradient_norm_is_finite"] is True
    assert report["after_clip"]["maximum_absolute_gradient"] < 2
    assert report["scale_before_step"] == 8
    assert report["scale_after_update"] == 8 * scale_factor
    assert (
        report["scale_change"]
        == {0.5: "decreased", 1.0: "equal", 2.0: "increased"}[scale_factor]
    )
    assert report["sentinel_parameter_changed"] == (scale_factor >= 1)
    assert scheduler.step.call_count == int(scale_factor >= 1)
    assert "AMP update diagnostics:" in capsys.readouterr().out


@pytest.mark.parametrize("skips", [0, 1, 2, 5, 6, 7, 8, 14, 15, 16])
def test_bounded_smoke_uses_production_loop_with_simulated_skips(skips):
    """CPU doubles verify control flow, not CUDA AMP numerical behavior."""
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(1, 1)
    counts = smoke.SmokeCounts()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler_impl = Mock()
    scheduler = smoke.ObservedScheduler(scheduler_impl, counts)
    seen = []
    clear_calls = []
    original_zero_grad = optimizer.zero_grad

    def clear(*args, **kwargs):
        clear_calls.append(kwargs)
        original_zero_grad(*args, **kwargs)

    optimizer.zero_grad = clear
    batches = [
        {"input": torch.full((8, 1), float(i + 1)), "labels": torch.zeros(8, 1)}
        for i in range(4)
    ]

    class Scaler:
        value = 65536.0
        attempts = 0

        def scale(self, loss):
            # Start of each attempt must have no stale gradients.
            if len(seen) % 2 == 0:
                assert all(p.grad is None for p in model.parameters())
            seen.append(len(seen) % 4)
            return loss * self.value

        def get_scale(self):
            return self.value

        def unscale_(self, opt):
            for parameter in model.parameters():
                parameter.grad.div_(self.value)

        def step(self, opt):
            self.attempts += 1
            if self.attempts > skips:
                opt.step()

        def update(self):
            if self.attempts <= skips:
                self.value *= 0.5

    def stepped(*args):
        counts.optimizer_steps += 1

    handle = optimizer.register_step_post_hook(stepped)
    frozen = smoke.BoundedSmokeLoader(batches, counts)
    # Cache the actual order once; no new shuffle when it cycles.
    assert all(a is b for a, b in zip(frozen.batches, batches, strict=True))

    class CountBatches:
        def __iter__(self):
            for batch in frozen:
                counts.minibatches += 1
                expected = batches[len(consumed) % 4]
                assert batch is expected
                consumed.append(batch)
                yield batch

    consumed = []
    try:
        smoke.train_epoch(
            model, CountBatches(), optimizer, scheduler, Scaler(), torch.device("cpu")
        )
    finally:
        handle.remove()
    assert smoke.MAX_UPDATE_ATTEMPTS == 16
    assert smoke.REQUIRED_SUCCESSFUL_UPDATES == 2
    expected_attempts = min(skips + 2, 16)
    successes = min(2, 16 - skips)
    assert counts.optimizer_update_attempts == expected_attempts
    assert counts.optimizer_steps == successes
    assert counts.scheduler_steps == successes
    assert scheduler_impl.step.call_count == successes
    assert len(consumed) == expected_attempts * 2
    assert len(clear_calls) == expected_attempts + 1
    assert all(call == {"set_to_none": True} for call in clear_calls)
    assert all(p.grad is None for p in model.parameters())
    if successes == 2:
        counts.validate()
    else:
        with pytest.raises(RuntimeError, match="at most 16 attempts"):
            counts.validate()


def test_overflow_diagnostics_are_valid_json():
    import json

    report = {"norm": float("inf"), "nested": [float("nan"), 1.0]}
    encoded = json.dumps(smoke._finite_json(report), allow_nan=False)
    assert json.loads(encoded) == {"norm": "inf", "nested": ["nan", 1.0]}
