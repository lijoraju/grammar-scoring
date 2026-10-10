import numpy as np
import pytest

from grammar_scoring.features.speaker_voxtral import unit_mean, windows


def test_windows_full_and_short_clips():
    assert windows(25, 10) == [(0, 10), (10, 20)]
    assert windows(30, 10) == [(0, 10), (10, 20), (20, 30)]
    assert windows(7, 10) == [(0, 7)]
    with pytest.raises(ValueError):
        windows(0, 10)


def test_unit_mean_normalizes_rows_before_averaging():
    out = unit_mean(np.array([[10.0, 0.0], [0.0, 0.1]]))
    assert out == pytest.approx([np.sqrt(0.5), np.sqrt(0.5)])
    assert np.linalg.norm(out) == pytest.approx(1.0)


def test_unit_mean_rejects_empty_or_cancelling_input():
    with pytest.raises(ValueError, match="non-empty"):
        unit_mean(np.zeros((0, 3)))
    with pytest.raises(ValueError, match="zero length"):
        unit_mean(np.array([[1.0, 0.0], [-1.0, 0.0]]))
