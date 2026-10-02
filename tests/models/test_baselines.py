import numpy as np
import pytest

from grammar_scoring.models.baselines import MeanRegressor


def test_mean_and_predictions():
    model = MeanRegressor().fit([1, 2, 6])
    assert model.mean_ == pytest.approx(3)
    predictions = model.predict(4)
    assert predictions.shape == (4,)
    np.testing.assert_array_equal(predictions, [3, 3, 3, 3])
    assert model.predict(0).shape == (0,)


@pytest.mark.parametrize("targets", [[], [np.nan], [np.inf], [-np.inf], [[1]], ["a"]])
def test_invalid_targets(targets):
    with pytest.raises(ValueError):
        MeanRegressor().fit(targets)


@pytest.mark.parametrize("count", [-1, 1.5, True, "2", None])
def test_invalid_prediction_count(count):
    with pytest.raises(ValueError):
        MeanRegressor().fit([1]).predict(count)


def test_unfitted_prediction():
    with pytest.raises(RuntimeError, match="fitted"):
        MeanRegressor().predict(1)


def test_large_finite_targets():
    model = MeanRegressor().fit([1e308, 1e308])
    assert model.mean_ == 1e308
