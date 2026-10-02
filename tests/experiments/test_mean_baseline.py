import numpy as np
import pandas as pd
import pytest

from grammar_scoring.evaluation.metrics import regression_metrics, rmse
from grammar_scoring.evaluation.validation import make_cv_folds
from grammar_scoring.experiments.mean_baseline import evaluate_mean_baseline


def test_oof_uses_training_subset_and_preserves_row_order():
    train = pd.DataFrame(
        {"filename": [f"{i}.wav" for i in range(6)], "label": [0, 0, 0, 0, 0, 6]},
        index=[9, 2, 8, 1, 7, 3],
    )
    result = evaluate_mean_baseline(train, n_splits=3, n_bins=1)
    targets = train.label.to_numpy()
    folds = make_cv_folds(targets, n_splits=3, n_bins=1)
    np.testing.assert_array_equal(result.oof.fold, folds)
    pd.testing.assert_frame_equal(result.oof[["filename", "label"]], train)
    assert list(result.oof.columns) == ["filename", "label", "fold", "prediction"]
    assert len(result.oof) == len(train)
    assert np.isfinite(result.oof.prediction).all()
    for fold in range(3):
        validation = folds == fold
        expected = targets[~validation].mean()
        np.testing.assert_array_equal(result.oof.prediction[validation], expected)
        assert expected != targets.mean()
        row = result.per_fold.iloc[fold]
        assert row.training_target_mean == pytest.approx(expected)
        assert row.validation_target_mean == targets[validation].mean()
        assert row.rmse == rmse(targets[validation], np.full(2, expected))
        assert np.isnan(row.pearson_correlation)
    assert result.oof_metrics == regression_metrics(targets, result.oof.prediction)
    assert result.training_rmse == rmse(targets, np.full(6, targets.mean()))
    assert result.oof_metrics["rmse"] != result.training_rmse
    repeat = evaluate_mean_baseline(train, n_splits=3, n_bins=1)
    pd.testing.assert_frame_equal(result.oof, repeat.oof)


def test_constant_targets_keep_undefined_pearson():
    train = pd.DataFrame({"filename": list("abcdef"), "label": np.ones(6)})
    result = evaluate_mean_baseline(train, n_splits=3)
    assert np.isnan(result.oof_metrics["pearson_correlation"])
    assert result.per_fold.pearson_correlation.isna().all()
    assert result.training_rmse == result.oof_metrics["rmse"] == 0


@pytest.mark.parametrize(
    "train",
    [
        pd.DataFrame({"label": [1, 2]}),
        pd.DataFrame({"filename": ["a", "a"], "label": [1, 2]}),
        pd.DataFrame({"filename": ["a", None], "label": [1, 2]}),
        pd.DataFrame({"filename": ["a", " "], "label": [1, 2]}),
        pd.DataFrame({"filename": ["a", "b"], "label": [1, np.nan]}),
    ],
)
def test_invalid_training_table(train):
    with pytest.raises(ValueError):
        evaluate_mean_baseline(train, n_splits=2)
