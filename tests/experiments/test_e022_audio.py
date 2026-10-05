import numpy as np
import pandas as pd
import pytest

from grammar_scoring.experiments.e022_audio import (
    best_blend_weight,
    load_audio,
    nested_audio_oof,
    nested_blend,
)


def test_best_blend_weight_recovers_known_mix():
    rng = np.random.default_rng(0)
    labels = rng.normal(size=400)
    text = labels + rng.normal(scale=0.6, size=400)
    audio = labels + rng.normal(scale=0.6, size=400)
    # Independent equal-noise views: the optimal weight is 0.5.
    assert best_blend_weight(text, audio, labels) == pytest.approx(0.5, abs=0.1)
    assert best_blend_weight(labels, audio, labels) == 0.0


def test_nested_blend_never_uses_held_out_labels():
    rng = np.random.default_rng(1)
    labels = rng.normal(size=100)
    text = labels + rng.normal(scale=0.5, size=100)
    audio = labels + rng.normal(scale=0.5, size=100)
    folds = np.arange(100) % 5
    corrupted = labels.copy()
    corrupted[folds == 3] += 100
    a, _ = nested_blend(text, audio, labels, folds)
    b, _ = nested_blend(text, audio, corrupted, folds)
    assert np.allclose(a[folds == 3], b[folds == 3])


def test_nested_audio_oof_is_fold_isolated():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(60, 4))
    labels = x[:, 0] + rng.normal(scale=0.1, size=60)
    folds = np.arange(60) % 3
    corrupted = labels.copy()
    corrupted[folds == 1] = 50.0
    a, params = nested_audio_oof(x, labels, folds)
    b, _ = nested_audio_oof(x, corrupted, folds)
    assert np.allclose(a[folds == 1], b[folds == 1])
    assert len(params) == 3 and set(params[0]) == {"svr__C", "svr__epsilon"}
    assert np.corrcoef(a, labels)[0, 1] > 0.8


def test_load_audio_aligns_and_validates(tmp_path):
    np.save(tmp_path / "e.npy", np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]))
    pd.DataFrame({"filename": ["a", "b", "c"]}).to_csv(tmp_path / "m.csv", index=False)
    out = load_audio(tmp_path / "e.npy", tmp_path / "m.csv", ["c", "a"])
    assert out.tolist() == [[3.0, 3.0], [1.0, 1.0]]
    with pytest.raises(ValueError, match="lack audio"):
        load_audio(tmp_path / "e.npy", tmp_path / "m.csv", ["z"])
    np.save(tmp_path / "bad.npy", np.array([[np.nan, 1.0], [2.0, 2.0], [3.0, 3.0]]))
    with pytest.raises(ValueError, match="finite"):
        load_audio(tmp_path / "bad.npy", tmp_path / "m.csv", ["a"])
