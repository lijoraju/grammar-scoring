import numpy as np
import pytest

from grammar_scoring.features.grammar_signals import (
    edit_features,
    expected_score,
    split_sentences,
    word_edit_distance,
    words,
)


def test_split_sentences_handles_punctuation_and_run_ons():
    assert split_sentences("I go. He went!  Ok") == ["I go.", "He went!", "Ok"]
    run_on = " ".join(["w"] * 85)
    assert [len(s.split()) for s in split_sentences(run_on, max_words=40)] == [
        40,
        40,
        5,
    ]
    assert split_sentences("   ") == []


def test_words_strips_punctuation_and_case():
    assert words("He don't, GO!") == ["he", "don't", "go"]


def test_word_edit_distance():
    assert word_edit_distance(["a", "b", "c"], ["a", "b", "c"]) == 0
    assert word_edit_distance(["he", "go"], ["he", "goes"]) == 1
    assert word_edit_distance([], ["x", "y"]) == 2
    assert word_edit_distance(["a", "b"], []) == 2


def test_edit_features():
    result = edit_features(
        ["He go to school.", "It is fine."], ["He goes to school.", "It is fine."]
    )
    assert result == {
        "gec_edit_rate": pytest.approx(1 / 7),
        "gec_changed_fraction": 0.5,
        "gec_n_words": 7,
    }
    assert edit_features([], [])["gec_edit_rate"] == 0


def test_expected_score():
    mean, std = expected_score(np.array([0.0, 0.0, 0.0, 0.0, 0.0]))
    assert mean == pytest.approx(3.0)
    assert std == pytest.approx(np.sqrt(2.0))
    mean, std = expected_score(np.array([-50.0, -50.0, -50.0, -50.0, 50.0]))
    assert mean == pytest.approx(5.0)
    assert std == pytest.approx(0.0, abs=1e-6)
