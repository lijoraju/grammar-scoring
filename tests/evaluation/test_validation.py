import numpy as np
import pandas as pd
import pytest

from grammar_scoring.evaluation import (
    make_cv_folds,
    make_regression_stratification_bins,
    summarize_cv_folds,
    validate_cv_folds,
)


@pytest.mark.parametrize("n_splits", [2, 5, 7])
def test_folds_are_complete_balanced_and_deterministic(n_splits):
    targets = np.linspace(0, 5, 103)
    original = targets.copy()
    folds = make_cv_folds(targets, n_splits=n_splits)
    np.testing.assert_array_equal(folds, make_cv_folds(targets, n_splits=n_splits))
    assert not np.array_equal(
        folds, make_cv_folds(targets, n_splits=n_splits, random_state=17)
    )
    assert folds.shape == targets.shape
    assert folds.dtype.kind == "i"
    np.testing.assert_array_equal(np.unique(folds), np.arange(n_splits))
    counts = np.bincount(folds)
    assert counts.max() - counts.min() <= 1
    np.testing.assert_array_equal(targets, original)
    validate_cv_folds(targets, folds, n_splits)
    bins = make_regression_stratification_bins(targets, n_splits=n_splits)
    for stratum in np.unique(bins):
        counts = np.bincount(folds[bins == stratum], minlength=n_splits)
        assert counts.max() - counts.min() <= 1


@pytest.mark.parametrize(
    "targets",
    [
        np.repeat([0, 1, 2, 3], [40, 1, 1, 8]),
        np.repeat([0, 1, 2], [90, 5, 5]),
        np.ones(20),
        np.arange(13),
    ],
)
def test_quantized_and_collapsed_bins(targets):
    original = targets.copy()
    bins = make_regression_stratification_bins(targets, n_bins=10)
    assert np.isfinite(bins).all()
    assert np.bincount(bins).min() >= 5
    np.testing.assert_array_equal(
        bins, make_regression_stratification_bins(targets, 10)
    )
    np.testing.assert_array_equal(targets, original)
    validate_cv_folds(targets, make_cv_folds(targets, n_bins=10))


def test_quantile_strategy_and_sparse_bin_fallback():
    targets = np.arange(100)
    np.testing.assert_array_equal(
        make_regression_stratification_bins(targets),
        pd.qcut(targets, q=5, labels=False, duplicates="drop"),
    )
    sparse = np.repeat([0, 1, 2], [20, 2, 3])
    bins = make_regression_stratification_bins(sparse)
    assert np.bincount(bins).min() >= 5


@pytest.mark.parametrize("targets", [[], [1, 2, 3, 4], [np.nan] * 10, [[1] * 10]])
def test_impossible_or_invalid_stratification(targets):
    with pytest.raises(ValueError):
        make_regression_stratification_bins(targets)
    with pytest.raises(ValueError):
        make_cv_folds(targets)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_splits": 1},
        {"n_splits": 2.5},
        {"n_splits": True},
        {"n_bins": 0},
        {"n_bins": "5"},
        {"random_state": None},
        {"random_state": -1},
        {"random_state": 2**32},
    ],
)
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        make_cv_folds(np.arange(20), **kwargs)


@pytest.mark.parametrize(
    "folds",
    [[0, 1], [0, 0, 0, 0], [0, 0, 0, 1], [1, 1, 2, 2], [0.0, 0, 1, 1], [[0, 1]]],
)
def test_invalid_fold_assignments(folds):
    with pytest.raises(ValueError):
        validate_cv_folds([1, 2, 3, 4], folds, n_splits=2)
    with pytest.raises(ValueError):
        summarize_cv_folds([1, 2, 3, 4], folds)


def test_fold_summary():
    summary = summarize_cv_folds([0, 2, 4, 6], [0, 0, 1, 1])
    expected = pd.DataFrame(
        {
            "fold": [0, 1],
            "n_samples": [2, 2],
            "target_mean": [1.0, 5.0],
            "target_std": [1.0, 1.0],
            "target_median": [1.0, 5.0],
            "target_min": [0.0, 4.0],
            "target_max": [2.0, 6.0],
        }
    )
    pd.testing.assert_frame_equal(summary, expected)
