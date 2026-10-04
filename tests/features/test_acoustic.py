import wave

import numpy as np
import pandas as pd
import pytest

from grammar_scoring.features.acoustic import (
    ARTIFACT_COLUMNS,
    FEATURE_COLUMNS,
    activity_features,
    extract_features,
    load_waveform,
    validate_waveform,
)
from grammar_scoring.features.acoustic_artifacts import (
    load_features,
    shift_audit,
    validate_features,
)


@pytest.mark.parametrize(
    "signal",
    [
        np.zeros(16000),
        np.ones(16000) * 0.1,
        np.ones(16000) * 1e-14,
        np.sin(2 * np.pi * 220 * np.arange(16000) / 16000) * 0.2,
        np.array([0.0]),
        np.arange(399) / 1000,
    ],
)
def test_finite_deterministic_order(signal):
    first = extract_features(signal)
    assert tuple(first) == FEATURE_COLUMNS
    assert len(FEATURE_COLUMNS) == 51
    assert "duration_seconds" not in FEATURE_COLUMNS
    assert not any("sample_count" in name for name in FEATURE_COLUMNS)
    assert np.isfinite(list(first.values())).all()
    assert first == extract_features(signal)


@pytest.mark.parametrize(
    "signal,rate",
    [
        (np.array([]), 16000),
        (np.array([np.nan]), 16000),
        (np.array([np.inf]), 16000),
        (np.zeros((10, 2)), 16000),
        (np.ones(10), 8000),
    ],
)
def test_invalid_waveform(signal, rate):
    with pytest.raises(ValueError):
        validate_waveform(signal, rate)


def test_wav_validation(tmp_path):
    path = tmp_path / "audio.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(np.array([0, 32767, -32768], dtype="<i2").tobytes())
    np.testing.assert_array_equal(load_waveform(path), [0, 32767 / 32768, -1])
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(np.zeros(20, dtype="<i2").tobytes())
    with pytest.raises(ValueError, match="mono"):
        load_waveform(path)
    path.write_bytes(b"broken")
    with pytest.raises(ValueError):
        load_waveform(path)


def test_truncated_wav(tmp_path):
    path = tmp_path / "short.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(np.ones(100, dtype="<i2").tobytes())
    path.write_bytes(path.read_bytes()[:-2])
    with pytest.raises(ValueError, match="Truncated"):
        load_waveform(path)


def test_activity_pause_minimum_and_edge_silence():
    rms = np.r_[
        np.zeros(30),
        np.ones(40),
        np.zeros(19),
        np.ones(20),
        np.zeros(20),
        np.ones(40),
        np.zeros(30),
    ]
    values = activity_features(rms)
    assert values["pause_count"] == 1
    assert values["mean_pause_duration"] == pytest.approx(0.2)
    assert values["pause_rate_per_active_second"] == pytest.approx(1.0)
    assert values["active_ratio"] + values["silence_ratio"] == pytest.approx(1)
    silence = activity_features(np.zeros(100))
    assert silence["active_ratio"] == 0
    assert silence["silence_ratio"] == 1
    assert silence["pause_count"] == 0
    assert silence["mean_active_segment_duration"] == 0
    assert activity_features(np.ones(100))["active_ratio"] == 1


def test_waveform_pause_detection():
    tone = np.sin(2 * np.pi * 220 * np.arange(16000) / 16000) * 0.2
    values = extract_features(np.r_[tone, np.zeros(8000), tone])
    assert values["pause_count"] == 1
    assert 0.45 <= values["mean_pause_duration"] <= 0.5


def table(split="train", names=("a.wav", "b.wav")):
    descriptors = extract_features(np.zeros(400))
    return pd.DataFrame(
        [
            {"filename": name, "split": split, "duration_seconds": 1.0, **descriptors}
            for name in names
        ],
        columns=ARTIFACT_COLUMNS,
    )


def test_artifact_identity_validation(tmp_path):
    frame = table()
    expected = frame[["filename"]]
    validate_features(frame, expected, "train")
    path = tmp_path / "features.csv"
    frame.to_csv(path, index=False)
    pd.testing.assert_frame_equal(load_features(path, expected, "train"), frame)
    for bad in (
        frame.iloc[::-1],
        frame.assign(filename="a.wav"),
        frame.assign(split="test"),
        frame.assign(rms_mean=np.inf),
        frame.assign(label=1),
        frame.drop(columns="mfcc_01_mean"),
    ):
        with pytest.raises(ValueError):
            validate_features(bad, expected, "train")
    path.write_text("filename,filename\na,b\n")
    with pytest.raises(ValueError, match="schema"):
        load_features(path, expected, "train")


def test_shift_definition_and_constant_cases():
    train, test = table(), table("test")
    train["rms_mean"] = [1, 3]
    test["rms_mean"] = [2, 4]
    test["duration_seconds"] = 2
    audit = shift_audit(train, test).set_index("feature")
    assert audit.loc["rms_mean", "smd"] == pytest.approx(1)
    assert audit.loc["rms_mean", "shift"] == "large"
    assert np.isinf(audit.loc["duration_seconds", "smd"])
    assert not audit.loc["duration_seconds", "model_feature"]
    assert audit.loc["rms_std", "smd"] == 0
    test["rms_mean"] = [1.6, 3.6]
    assert (
        shift_audit(train, test).set_index("feature").loc["rms_mean", "shift"]
        == "moderate"
    )
