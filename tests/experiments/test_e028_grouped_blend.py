import numpy as np
import pytest

from grammar_scoring.experiments.e028_grouped_blend import (
    cell_weights,
    fold_ridge_oof,
    has_partner,
    nested_blend,
    ridge_channel,
)


def test_has_partner_finds_same_speaker_clips():
    rng = np.random.default_rng(0)
    voices = rng.normal(size=(3, 16))
    clips = np.vstack([voices[0], voices[0] + 0.01, voices[1], voices[2]])
    assert has_partner(clips, clips, threshold=0.9).tolist() == [
        True,
        True,
        False,
        False,
    ]


def test_cell_weights_match_target_shares():
    cells = np.array([0, 0, 0, 1, 1, 2])
    target = np.array([0, 1, 1, 1, 2, 2, 2, 2])
    weights = cell_weights(cells, target)
    for cell in (0, 1, 2):
        share = weights[cells == cell].sum() / weights.sum()
        assert share == pytest.approx(np.mean(target == cell))
    with pytest.raises(ValueError, match="without training rows"):
        cell_weights(np.array([0, 0]), np.array([0, 3]))


def test_fold_ridge_oof_is_fold_isolated():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(60, 4))
    labels = x[:, 0] + rng.normal(scale=0.1, size=60)
    folds = np.arange(60) % 3
    corrupted = labels.copy()
    corrupted[folds == 2] = 50.0
    a = fold_ridge_oof(x, labels, folds, alpha=1.0)
    b = fold_ridge_oof(x, corrupted, folds, alpha=1.0)
    assert np.allclose(a[folds == 2], b[folds == 2])
    assert np.corrcoef(a, labels)[0, 1] > 0.9


def test_ridge_channel_averages_views():
    rng = np.random.default_rng(2)
    labels = rng.normal(size=40)
    folds = np.arange(40) % 4
    data = {}
    for name in ("a", "b"):
        data[f"{name}_train"] = labels[:, None] + rng.normal(scale=0.05, size=(40, 3))
        data[f"{name}_test"] = np.array([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
    oof, test = ridge_channel(
        data, ("a", "b"), np.arange(40), np.array([1, 0]), labels, folds, 1.0
    )
    assert np.corrcoef(oof, labels)[0, 1] > 0.95
    assert test[0] < 0 < test[1]


def test_nested_blend_never_uses_held_out_labels():
    rng = np.random.default_rng(3)
    labels = rng.uniform(1, 5, size=100)
    channels = np.column_stack(
        [labels + rng.normal(scale=0.4, size=100) for _ in range(2)]
    )
    folds = np.arange(100) % 5
    base, chosen = nested_blend(channels, labels, folds, np.ones(100))
    corrupted = labels.copy()
    corrupted[folds == 1] += 100
    other, _ = nested_blend(channels, corrupted, folds, np.ones(100))
    assert np.allclose(base[folds == 1], other[folds == 1])
    assert len(chosen) == 5 and base.min() >= 1 and base.max() <= 5
