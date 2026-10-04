import numpy as np
import pytest

from grammar_scoring.features.asr_fluency import FEATURE_COLUMNS, extract_features


def record(segments=None, text="Um uh erm hmm go go go", duration=12):
    return {
        "text": text,
        "duration_seconds": duration,
        "segments": segments
        if segments is not None
        else [
            {"start": 0, "end": 2, "text": "one two"},
            {"start": 2.5, "end": 3.5, "text": "three"},
            {"start": 4.5, "end": 6.5, "text": "four five six"},
            {"start": 8.5, "end": 10.5, "text": "seven"},
        ],
    }


def test_exact_schema_and_calculations():
    features = extract_features(record())
    expected = {
        "segment_count": 4,
        "words_per_segment_mean": 1.75,
        "words_per_segment_std": np.std([2, 1, 3, 1]),
        "segment_duration_mean": 1.75,
        "segment_duration_std": np.std([2, 1, 2, 2]),
        "short_segment_ratio": 0.25,
        "recognized_speech_seconds": 7,
        "speech_coverage_ratio": 7 / 12,
        "words_per_recognized_speech_second": 1,
        "gap_count": 3,
        "gap_mean": 3.5 / 3,
        "gap_std": np.std([0.5, 1, 2]),
        "gap_max": 2,
        "total_gap_seconds": 3.5,
        "long_gap_0_5_count": 3,
        "long_gap_1_0_count": 2,
        "long_gap_2_0_count": 1,
        "filler_rate": 4 / 7,
        "immediate_repetition_rate": 2 / 6,
    }
    assert tuple(features) == tuple(expected) == FEATURE_COLUMNS
    assert len(features) == 19
    assert features == pytest.approx(expected)
    assert all(type(value) is float for value in features.values())


def test_empty():
    assert set(extract_features(record([], "", 0)).values()) == {0.0}


def test_single_segment():
    features = extract_features(record([{"start": 0, "end": 1, "text": "um"}], "um"))
    assert features["words_per_segment_std"] == 0
    assert features["segment_duration_std"] == 0
    assert features["immediate_repetition_rate"] == 0
    assert all(value == 0 for key, value in features.items() if "gap" in key)


def test_overlap_negative_length_and_zero_denominator():
    features = extract_features(
        record(
            [
                {"start": 0, "end": 3, "text": "a"},
                {"start": 1, "end": 0, "text": "b"},
            ],
            duration=0,
        )
    )
    assert features["recognized_speech_seconds"] == 3
    assert features["gap_count"] == 1
    assert features["gap_max"] == 0
    assert features["speech_coverage_ratio"] == 0
    assert np.isfinite(list(features.values())).all()


@pytest.mark.parametrize(
    "field,value",
    [
        ("duration_seconds", np.nan),
        ("duration_seconds", np.inf),
        ("duration_seconds", -1),
        ("text", None),
        ("segments", None),
        ("segments", [{}]),
        ("segments", [{"start": np.inf, "end": 1, "text": ""}]),
        ("segments", [{"start": 0, "end": 1, "text": None}]),
    ],
)
def test_invalid(field, value):
    item = record()
    item[field] = value
    with pytest.raises(ValueError):
        extract_features(item)


@pytest.mark.parametrize("field", ["text", "segments", "duration_seconds"])
def test_missing(field):
    item = record()
    del item[field]
    with pytest.raises(ValueError, match="Missing"):
        extract_features(item)


def test_tokenization_and_zero_speech():
    features = extract_features(record([], "UM, um! uh... er hmm 123", 10))
    assert features["filler_rate"] == 4 / 5
    assert features["immediate_repetition_rate"] == 1 / 4
    assert features["words_per_recognized_speech_second"] == 0
    silent = extract_features(record(text=""))
    assert silent["filler_rate"] == silent["immediate_repetition_rate"] == 0


def test_nonfinite_derived_features_rejected():
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="finite"):
            extract_features(record([{"start": -1e308, "end": 1e308, "text": "word"}]))
