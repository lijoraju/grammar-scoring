import struct
import wave
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from grammar_scoring.data.audio import collect_audio_metadata, read_audio_metadata


def write_wav(
    path: Path,
    channels: int = 1,
    sample_rate: int = 8000,
    sample_width: int = 2,
    frames: int = 80,
) -> Path:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(channels)
        audio.setframerate(sample_rate)
        audio.setsampwidth(sample_width)
        audio.writeframes(bytes(frames * channels * sample_width))
    return path


@pytest.mark.parametrize(
    "channels,sample_rate,sample_width,frames",
    [(1, 8000, 1, 80), (2, 16000, 2, 320)],
)
def test_valid_metadata(tmp_path, channels, sample_rate, sample_width, frames):
    path = write_wav(
        tmp_path / "audio.wav", channels, sample_rate, sample_width, frames
    )
    metadata = read_audio_metadata(path)
    assert metadata.path == path
    assert metadata.num_channels == channels
    assert metadata.sample_rate == sample_rate
    assert metadata.sample_width_bytes == sample_width
    assert metadata.num_frames == frames
    assert metadata.duration_seconds == pytest.approx(frames / sample_rate)
    with pytest.raises(FrozenInstanceError):
        metadata.num_frames = 1


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="WAV file does not exist"):
        read_audio_metadata(tmp_path / "missing.wav")


def test_directory(tmp_path):
    with pytest.raises(ValueError, match="not a file"):
        read_audio_metadata(tmp_path)


@pytest.mark.parametrize("contents", [b"not a WAV file", b"", b"RIFF"])
def test_malformed_file(tmp_path, contents):
    path = tmp_path / "invalid.wav"
    path.write_bytes(contents)
    with pytest.raises(ValueError, match="Cannot read WAV metadata"):
        read_audio_metadata(path)


def test_zero_frames(tmp_path):
    path = write_wav(tmp_path / "empty.wav", frames=0)
    with pytest.raises(ValueError, match="frame count"):
        read_audio_metadata(path)


@pytest.mark.parametrize("offset,format_code", [(24, "<I"), (22, "<H")])
def test_invalid_header_values(tmp_path, offset, format_code):
    path = write_wav(tmp_path / "invalid.wav")
    header = bytearray(path.read_bytes())
    struct.pack_into(format_code, header, offset, 0)
    path.write_bytes(header)
    with pytest.raises(ValueError, match="WAV"):
        read_audio_metadata(path)


def test_batch_preserves_order_and_duplicates(tmp_path):
    first = write_wav(tmp_path / "z.wav", frames=80)
    second = write_wav(tmp_path / "a.wav", channels=2, frames=160)
    paths = [first, second, first]
    frame = collect_audio_metadata(path for path in paths)
    assert frame["filename"].tolist() == ["z.wav", "a.wav", "z.wav"]
    assert frame["path"].tolist() == paths
    assert frame["num_frames"].tolist() == [80, 160, 80]
    assert frame["duration_seconds"].tolist() == pytest.approx([0.01, 0.02, 0.01])
    assert frame["sample_rate"].tolist() == [8000] * 3
    assert frame["num_channels"].tolist() == [1, 2, 1]
    assert frame["sample_width_bytes"].tolist() == [2] * 3


def test_empty_collection():
    frame = collect_audio_metadata(iter(()))
    assert frame.empty
    assert frame.columns.tolist() == [
        "filename",
        "path",
        "duration_seconds",
        "sample_rate",
        "num_channels",
        "sample_width_bytes",
        "num_frames",
    ]


def test_batch_invalid_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing.wav"):
        collect_audio_metadata([tmp_path / "missing.wav"])
