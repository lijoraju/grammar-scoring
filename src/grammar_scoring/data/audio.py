"""Inspect WAV headers without reading or processing audio samples."""

import wave
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_COLUMNS = [
    "filename",
    "path",
    "duration_seconds",
    "sample_rate",
    "num_channels",
    "sample_width_bytes",
    "num_frames",
]


@dataclass(frozen=True)
class AudioMetadata:
    """Immutable WAV header metadata, with duration measured in seconds.

    Attributes:
        path: Original filesystem path supplied by the caller.
        duration_seconds: Frame count divided by sample rate.
        sample_rate: Sampling frequency in hertz.
        num_channels: Number of audio channels.
        sample_width_bytes: Bytes per sample per channel.
        num_frames: Number of audio frames declared by the header.
    """

    path: Path
    duration_seconds: float
    sample_rate: int
    num_channels: int
    sample_width_bytes: int
    num_frames: int


def read_audio_metadata(path: Path) -> AudioMetadata:
    """Read and validate a WAV header without decoding audio samples.

    Args:
        path: Path to a WAV file supported by the standard-library wave module.

    Returns:
        Validated metadata with duration derived from the header.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the path is not a file or WAV metadata is invalid.
        OSError: If filesystem access prevents reading the file.
    """
    if not path.exists():
        raise FileNotFoundError(f"WAV file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Expected a WAV file, but path is not a file: {path}")
    try:
        with wave.open(str(path), "rb") as audio:
            sample_rate = audio.getframerate()
            num_channels = audio.getnchannels()
            sample_width_bytes = audio.getsampwidth()
            num_frames = audio.getnframes()
    except (wave.Error, EOFError) as exc:
        raise ValueError(
            f"Cannot read WAV metadata from {path}: invalid or unsupported WAV "
            f"header ({exc}). Check or replace the file."
        ) from exc
    except OSError as exc:
        raise OSError(
            f"Cannot read WAV file {path}: {exc}. Check file access permissions."
        ) from exc
    for name, value in (
        ("sample rate", sample_rate),
        ("frame count", num_frames),
        ("channel count", num_channels),
        ("sample width", sample_width_bytes),
    ):
        if value <= 0:
            raise ValueError(
                f"Invalid WAV {name} in {path}: expected a positive value, "
                f"got {value}. Check or replace the file."
            )
    return AudioMetadata(
        path=path,
        duration_seconds=num_frames / sample_rate,
        sample_rate=sample_rate,
        num_channels=num_channels,
        sample_width_bytes=sample_width_bytes,
        num_frames=num_frames,
    )


def collect_audio_metadata(paths: Iterable[Path]) -> pd.DataFrame:
    """Collect WAV metadata in the supplied iteration order.

    Args:
        paths: Explicit file paths; duplicates are preserved and no files are
            discovered automatically.

    Returns:
        One row per input path, with Path objects in the path column. Empty
        input returns an empty table with the same columns.

    Raises:
        FileNotFoundError: If an input file does not exist.
        ValueError: If an input path or WAV header is invalid.
        OSError: If an input file cannot be read.
    """
    rows: list[dict[str, object]] = []
    for path in paths:
        metadata = read_audio_metadata(path)
        rows.append(
            {
                "filename": metadata.path.name,
                "path": metadata.path,
                "duration_seconds": metadata.duration_seconds,
                "sample_rate": metadata.sample_rate,
                "num_channels": metadata.num_channels,
                "sample_width_bytes": metadata.sample_width_bytes,
                "num_frames": metadata.num_frames,
            }
        )
    return pd.DataFrame(rows, columns=_COLUMNS)
