import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from grammar_scoring.experiments import ordinal_smoke as smoke


def test_smoke_subset_namespace_roundtrip_and_cuda_guard(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    frame = pd.DataFrame(
        {"filename": [f"row_{i}" for i in range(100)], "fold": np.tile(range(5), 20)}
    )
    train, valid = smoke.smoke_subset(frame)
    assert len(train) == 32 and len(valid) == 16
    assert set(train.filename).isdisjoint(valid.filename)
    assert smoke.smoke_root(tmp_path / "smoke/E007") == tmp_path / "smoke/E007"
    for path in ["models/E007", "experiments/E007", "models/E007/smoke/E007"]:
        with pytest.raises(ValueError):
            smoke.smoke_root(tmp_path / path)
    predictions = np.linspace(0, 5, 16)
    smoke.check_round_trip(predictions, predictions)
    with pytest.raises(RuntimeError):
        smoke.check_round_trip(predictions, predictions + 1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(smoke, "load_inputs", Mock(side_effect=AssertionError))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace())
    with pytest.raises(RuntimeError, match="CUDA is required"):
        smoke.run_smoke(tmp_path, tmp_path / "folds.csv", tmp_path / "smoke/E007")
    assert not (tmp_path / "smoke").exists()


def test_smoke_dtype_reports_support_bias_free_head():
    torch = pytest.importorskip("torch")
    from grammar_scoring.models.deberta_ordinal import DebertaOrdinal

    backbone = torch.nn.Linear(2, 2)
    backbone.config = SimpleNamespace(hidden_size=2)
    model = DebertaOrdinal(backbone)
    logits = model.head(backbone(torch.ones(1, 2))).float() - model.thresholds()
    loss = logits.square().mean()
    report = smoke.dtype_diagnostics("before_backward", model, loss)
    assert report["trainable_parameter_dtypes"] == ["torch.float32"]
    loss.backward()
    assert smoke.dtype_diagnostics("before_unscale", model, loss)[
        "gradient_dtypes"
    ] == ["torch.float32"]


def test_smoke_cli_failure_and_bounded_counts(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["smoke"])
    monkeypatch.setattr(smoke, "run_smoke", Mock(side_effect=RuntimeError("no CUDA")))
    with pytest.raises(SystemExit) as exc:
        smoke.main()
    assert exc.value.code == 1
    assert "[FAIL]" in capsys.readouterr().out
    counts = smoke.SmokeCounts(
        minibatches=4, optimizer_update_attempts=2, optimizer_steps=2, scheduler_steps=2
    )
    counts.validate()
    counts.scheduler_steps = 1
    with pytest.raises(RuntimeError):
        counts.validate()
