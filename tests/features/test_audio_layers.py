import wave

import numpy as np
import pytest

from grammar_scoring.features.audio_layers import (
    LayerStats,
    plan_chunks,
    read_wav,
    whisper_valid_frames,
)


def test_plan_chunks_covers_every_sample_and_merges_short_tail():
    assert plan_chunks(50, 20, 5) == [(0, 20), (20, 40), (40, 50)]
    assert plan_chunks(43, 20, 5) == [(0, 20), (20, 43)]
    assert plan_chunks(3, 20, 5) == [(0, 3)]
    with pytest.raises(ValueError):
        plan_chunks(0, 20, 5)


def test_whisper_valid_frames():
    assert whisper_valid_frames(16_000) == 50
    assert whisper_valid_frames(16_001) == 51
    assert whisper_valid_frames(16_000 * 45) == 1500


def test_layer_stats_match_direct_computation_across_chunks():
    rng = np.random.default_rng(0)
    frames = rng.normal(size=(3, 37, 4))  # [layers, frames, dim]
    stats = LayerStats()
    for start, end in [(0, 10), (10, 30), (30, 37)]:
        part = frames[:, start:end]
        stats.add(part.sum(1), (part**2).sum(1), end - start)
    mean, std = stats.finalize()
    assert np.allclose(mean, frames.mean(1))
    assert np.allclose(std, frames.std(1))
    assert stats.history == [10, 20, 7]


def test_layer_stats_validation():
    stats = LayerStats()
    with pytest.raises(ValueError, match="No frames"):
        stats.finalize()
    with pytest.raises(ValueError, match="at least one"):
        stats.add(np.zeros((1, 2)), np.zeros((1, 2)), 0)
    stats.add(np.zeros((1, 2)), np.zeros((1, 2)), 1)
    with pytest.raises(ValueError, match="disagree"):
        stats.add(np.zeros((2, 2)), np.zeros((2, 2)), 1)


def test_read_wav(tmp_path):
    path = tmp_path / "a.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(np.array([0, 16384, -32768], dtype="<i2").tobytes())
    assert read_wav(path).tolist() == [0.0, 0.5, -1.0]
    bad = tmp_path / "b.wav"
    with wave.open(str(bad), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(8_000)
        stream.writeframes(b"\x00\x00")
    with pytest.raises(ValueError, match="16 kHz"):
        read_wav(bad)
