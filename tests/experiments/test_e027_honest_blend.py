import numpy as np
import pytest

from grammar_scoring.experiments.e027_honest_blend import (
    apply_calibration,
    best_weights,
    fit_calibration,
    grouped_ridge_oof,
    importance_weights,
    nested_honest_blend,
    ridge_channel,
    ridge_fit_predict,
    simplex_grid,
    speaker_groups,
    unseen_speaker_mask,
    weighted_rmse,
)


def test_speaker_groups_links_similar_voices():
    rng = np.random.default_rng(0)
    voices = rng.normal(size=(3, 16))
    clips = np.vstack([voices[0], voices[0] + 0.01, voices[1], voices[2], voices[2]])
    groups = speaker_groups(clips, clips, threshold=0.9)
    assert groups[0] == groups[1] and groups[3] == groups[4]
    assert len({groups[0], groups[2], groups[3]}) == 3


def test_unseen_speaker_mask():
    groups = np.array([0, 0, 1, 1, 2])
    folds = np.array([0, 1, 2, 2, 3])
    assert unseen_speaker_mask(groups, folds).tolist() == [
        False,
        False,
        True,
        True,
        True,
    ]


def test_importance_weights_match_target_share():
    short = np.array([True, False, False, False, True, False])
    mask = np.array([True, True, True, True, False, False])
    weights = importance_weights(short, mask, 0.7)
    share = weights[mask & short].sum() / weights[mask].sum()
    assert share == pytest.approx(0.7)
    with pytest.raises(ValueError, match="short and long"):
        importance_weights(
            short, np.array([False, True, True, True, False, False]), 0.7
        )


def test_weighted_rmse():
    assert weighted_rmse(
        np.array([0.0, 0.0]), np.array([1.0, 3.0]), np.array([3.0, 1.0])
    ) == (pytest.approx(np.sqrt(3.0)))


def test_grouped_ridge_oof_never_trains_on_the_same_speaker():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(20), 3)
    speaker_effect = rng.normal(size=20)[groups]
    # Each speaker has its own voice vector; labels depend on the speaker only,
    # so the features can only help by recognizing a speaker seen in training.
    features = rng.normal(size=(20, 30))[groups] + rng.normal(scale=0.01, size=(60, 30))
    leaky = ridge_fit_predict(features, speaker_effect, features, alpha=0.01)
    grouped = grouped_ridge_oof(features, speaker_effect, groups, alpha=0.01)
    error = lambda p: np.sqrt(np.mean((p - speaker_effect) ** 2))  # noqa: E731
    assert error(leaky) < 0.2 * speaker_effect.std()
    assert error(grouped) > 0.8 * speaker_effect.std()


def test_simplex_grid_and_best_weights():
    grid = simplex_grid(3, step=0.5)
    assert len(grid) == 6 and np.allclose(grid.sum(axis=1), 1.0) and (grid >= 0).all()
    labels = np.array([1.0, 2.0, 3.0, 4.0])
    channels = np.column_stack([labels, labels[::-1], np.zeros(4)])
    assert best_weights(channels, labels, np.ones(4)).tolist() == [1.0, 0.0, 0.0]


def test_calibration_fits_line_and_clips():
    blended = np.array([2.0, 3.0, 4.0])
    line = fit_calibration(blended, np.array([1.0, 3.0, 5.0]), np.ones(3))
    assert line == pytest.approx((-3.0, 2.0))
    out = apply_calibration(np.array([1.0, 3.5, 9.0]), line)
    assert out.tolist() == pytest.approx([1.0, 4.0, 5.0])


def test_nested_honest_blend_ignores_held_out_and_seen_rows():
    rng = np.random.default_rng(2)
    labels = rng.uniform(1, 5, size=120)
    channels = np.column_stack(
        [labels + rng.normal(scale=0.4, size=120) for _ in range(2)]
    )
    folds = np.arange(120) % 5
    mask = np.arange(120) % 3 != 0
    weights = np.ones(120)
    base, chosen = nested_honest_blend(channels, labels, folds, mask, weights)
    corrupted = labels.copy()
    corrupted[(folds == 1) | ~mask] += 100
    other, _ = nested_honest_blend(channels, corrupted, folds, mask, weights)
    apply = mask & (folds == 1)
    assert np.allclose(base[apply], other[apply])
    assert np.isnan(base[~mask]).all() and len(chosen) == 5


def test_ridge_channel_averages_feature_sets_and_aligns_rows():
    rng = np.random.default_rng(4)
    labels = rng.normal(size=40)
    groups = np.arange(40) % 10
    data = {}
    for name in ("a", "b"):
        data[f"{name}_train"] = np.vstack(
            [np.zeros((1, 3)), labels[:, None] + rng.normal(scale=0.05, size=(40, 3))]
        )
        data[f"{name}_test"] = np.array([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
    rows = np.arange(1, 41)  # skip the padding row
    oof, test = ridge_channel(
        data, ("a", "b"), rows, np.array([1, 0]), labels, groups, 1.0
    )
    assert np.corrcoef(oof, labels)[0, 1] > 0.95
    assert test[0] < 0 < test[1]
