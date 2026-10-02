import numpy as np
import pytest

from grammar_scoring.evaluation import (
    pearson_correlation,
    regression_metrics,
    rmse,
)


def test_rmse():
    assert rmse([1, 2, 3], [1, 2, 3]) == 0.0
    assert rmse([1, 2, 3], [2, 4, 5]) == pytest.approx(np.sqrt(3))
    assert isinstance(rmse([1], [2]), float)
    assert rmse([0, 0], [1e200, 1e200]) == pytest.approx(1e200)


@pytest.mark.parametrize("metric", [rmse, pearson_correlation, regression_metrics])
@pytest.mark.parametrize(
    ("actual", "predicted"),
    [
        ([1, 2], [1]),
        ([], []),
        ([np.nan], [1]),
        ([1], [np.inf]),
        ([-np.inf], [1]),
        ([[1, 2]], [[1, 2]]),
        (["1", "2"], [1, 2]),
        ([1j], [1]),
        ([True], [1]),
        (1, 1),
    ],
)
def test_invalid_metric_inputs(metric, actual, predicted):
    with pytest.raises(ValueError):
        metric(actual, predicted)


def test_pearson():
    assert pearson_correlation([1, 2, 3], [2, 4, 6]) == pytest.approx(1)
    assert pearson_correlation([1, 2, 3], [6, 4, 2]) == pytest.approx(-1)
    assert isinstance(pearson_correlation([1, 2], [3, 4]), float)


@pytest.mark.parametrize(
    ("actual", "predicted"),
    [([1, 2, 3], [2, 2, 2]), ([1, 1, 1], [1, 2, 3]), ([1], [2])],
)
def test_undefined_pearson(actual, predicted):
    assert np.isnan(pearson_correlation(actual, predicted))


def test_regression_metrics():
    actual, predicted = [1, 3, 2], [2, 4, 1]
    scores = regression_metrics(actual, predicted)
    assert scores == {
        "rmse": rmse(actual, predicted),
        "pearson_correlation": pearson_correlation(actual, predicted),
    }
    assert np.isnan(regression_metrics(actual, [2, 2, 2])["pearson_correlation"])
