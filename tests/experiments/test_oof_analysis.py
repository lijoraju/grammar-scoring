"""Synthetic tests for frozen OOF diagnostics."""

import numpy as np
import pandas as pd
import pytest

from grammar_scoring.experiments.oof_analysis import (
    align_oof,
    classify_candidate,
    crossfit_ols,
    fixed_blend,
)


@pytest.fixture
def frame():
    return pd.DataFrame(
        {
            "filename": [f"a{i}" for i in range(15)],
            "label": np.linspace(0, 5, 15),
            "fold": np.arange(15) % 5,
            "prediction": np.linspace(-1, 6, 15),
        }
    )


def test_alignment(frame):
    aligned = align_oof({"E005": frame, "other": frame.iloc[::-1]}, 15)
    pd.testing.assert_frame_equal(aligned["E005"], aligned["other"])


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("filename", "a1", "duplicate"),
        ("filename", "unknown", "filename-set"),
        ("label", 42, "label mismatch"),
        ("fold", 1, "fold mismatch"),
        ("prediction", np.inf, "non-finite"),
        ("prediction", np.nan, "non-finite"),
        ("filename", "", "empty"),
        ("fold", 0.5, "integers"),
        ("label", np.inf, "non-finite"),
    ],
)
def test_invalid_artifacts(frame, column, value, message):
    other = frame.copy()
    other[column] = other[column].astype(object)
    other.loc[0, column] = value
    with pytest.raises(ValueError, match=message):
        align_oof({"E005": frame, "other": other}, 15)


def test_schema_and_count(frame):
    with pytest.raises(ValueError, match="required fields"):
        align_oof({"E005": frame.drop(columns="prediction")}, 15)
    with pytest.raises(ValueError, match="expected"):
        align_oof({"E005": frame}, 769)


@pytest.mark.parametrize("columns", [1, 2])
def test_crossfit_excludes_held_out_labels(columns):
    rng = np.random.default_rng(42)
    features = rng.normal(size=(50, columns))
    labels = rng.normal(size=50)
    folds = np.arange(50) % 5
    predictions, parameters = crossfit_ols(features, labels, folds)
    for fold in range(5):
        changed = labels.copy()
        changed[folds == fold] += 1000
        altered, fitted = crossfit_ols(features, changed, folds)
        np.testing.assert_array_equal(
            predictions[folds == fold], altered[folds == fold]
        )
        assert fitted[fold] == parameters[fold]
        train = folds != fold
        design = np.column_stack([np.ones(50), features])
        coef = np.linalg.lstsq(design[train], labels[train], rcond=None)[0]
        np.testing.assert_allclose(predictions[~train], design[~train] @ coef)
    repeated, repeated_parameters = crossfit_ols(features, labels, folds)
    np.testing.assert_array_equal(predictions, repeated)
    assert parameters == repeated_parameters


def test_clipping_and_blend():
    raw = np.array([-1.0, 2.0, 6.0])
    np.testing.assert_array_equal(np.clip(raw, 0, 5), [0, 2, 5])
    secondary = np.array([3.0, 4.0, 5.0])
    np.testing.assert_allclose(fixed_blend(raw, secondary, 0.25), [2, 3.5, 5.25])
    np.testing.assert_array_equal(fixed_blend(raw, secondary, 1), raw)
    np.testing.assert_array_equal(fixed_blend(raw, secondary, 0), secondary)


@pytest.mark.parametrize(
    ("rmse", "pearson", "expected"),
    [
        (0.9, 0.6, "dominates_e005"),
        (0.9, 0.4, "tradeoff"),
        (1.1, 0.6, "tradeoff"),
        (1.1, 0.4, "worse"),
        (1, 0.5, "worse"),
        (1 - 1e-12, 0.5 + 1e-12, "worse"),
    ],
)
def test_classification(rmse, pearson, expected):
    assert (
        classify_candidate(
            {"rmse": rmse, "pearson_correlation": pearson},
            {"rmse": 1, "pearson_correlation": 0.5},
        )
        == expected
    )
