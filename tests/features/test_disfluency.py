import pytest

from grammar_scoring.features.disfluency import (
    content_words,
    repair_rate,
    segment_features,
    text_features,
)


def test_content_words_drop_fillers_and_punctuation():
    assert content_words("Um, I- I went, uh, home. Hmm") == ["i", "i", "went", "home"]


def test_repair_rate():
    assert repair_rate("I liked the playground.", "I liked, um, the playground.") == 0
    assert repair_rate("a lot of kids", "a lot kids") == 0
    assert repair_rate("I liked the park", "I liked to park") == pytest.approx(0.25)
    assert repair_rate("anything", "") == 0


def test_segment_features():
    segments = [
        {"start": 0.0, "end": 4.0, "text": "one two three four"},
        {"start": 5.0, "end": 9.0, "text": "five six"},
        {"start": 9.2, "end": 10.0, "text": "seven"},
    ]
    result = segment_features(segments, duration=12.0)
    assert result == {
        "speech_seconds": 10.0,
        "words_per_second": pytest.approx(0.7),
        "long_pauses": 1.0,
        "pause_seconds": pytest.approx(1.0),
    }
    assert segment_features([], 5.0)["words_per_second"] == 0.0


def test_text_features():
    result = text_features("I go. He went home.", "Um, I- I go, go. He went home.")
    assert result["disf_n_words"] == 7
    assert result["disf_filler_rate"] == pytest.approx(1 / 8)
    assert result["disf_false_start_rate"] == pytest.approx(1 / 8)
    assert result["disf_repeat_rate"] == pytest.approx(1 / 8)
    assert result["disf_mean_sentence_words"] == pytest.approx(2.5)
