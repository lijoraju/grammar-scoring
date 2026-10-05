import numpy as np
import pandas as pd
import pytest

from grammar_scoring.experiments.e017_stack import (
    feature_columns,
    stack_oof,
    stack_test,
)


def _data(n=60):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"ens": rng.normal(size=n), "gec_rate": rng.normal(size=n)})
    labels = 3 + frame["ens"] - 0.5 * frame["gec_rate"]
    return frame, labels.to_numpy(), np.arange(n) % 5


def test_stack_oof_isolates_validation_fold():
    frame, labels, folds = _data()
    corrupted = labels.copy()
    corrupted[folds == 2] += 50
    a = stack_oof(frame, labels, folds)
    b = stack_oof(frame, corrupted, folds)
    assert np.allclose(a[folds == 2], b[folds == 2])
    assert np.sqrt(np.mean((a - labels) ** 2)) < 0.1


def test_stack_test_returns_coefficients():
    frame, labels, _ = _data()
    predictions, coefficients = stack_test(frame, labels, frame.iloc[:3], alpha=1e-6)
    assert predictions == pytest.approx(labels[:3], abs=1e-3)
    assert coefficients["gec_rate"] < 0 < coefficients["ens"]


def test_feature_columns():
    frame = pd.DataFrame(columns=["filename", "gec_rate", "judge_score", "x"])
    assert feature_columns(frame, ["gec_", "judge_"]) == ["gec_rate", "judge_score"]
